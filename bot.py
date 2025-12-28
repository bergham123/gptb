import os
import re
import json
import hashlib
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from io import BytesIO
from urllib.parse import quote

import feedparser
from bs4 import BeautifulSoup
import requests
from PIL import Image, ImageOps

import telegram

TZ = ZoneInfo("Africa/Casablanca")

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Repo info (used to build raw URLs for media)
GITHUB_REPO_SLUG   = "bergham123/anime-news-bot"
GITHUB_REPO_BRANCH = "main"

# Your GitHub Pages site (optional)
SITE_BASE_URL = "https://bergham123.github.io/anime-news-bot"
ARTICLE_PAGE  = "article.html"

# =====================
# SOURCES
# =====================
# store=true  -> write to records + indexes + search + dedup maps
# store=false -> DO NOT write anything to json (only telegram), but still uses checkpoints to avoid re-sending.
SOURCES = [
    {"key": "crunchyroll", "rss": "https://cr-news-api-service.prd.crunchyrollsvc.com/v1/ar-SA/rss", "store": True},
    {"key": "youtube", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UC1WGYjPeHHc_3nRXqbW3OcQ", "store": False},
    # Add your 2 other sites:
    # {"key": "site3", "rss": "https://example.com/rss", "store": True},
    # {"key": "site4", "rss": "https://example.com/rss", "store": True},
]

# =====================
# PATHS
# =====================
RECORDS_DIR = Path("records")
INDEXES_DIR = Path("indexes")
STATE_DIR   = Path("state")
MEDIA_DIR   = Path("media")

GLOBAL_PAGES_DIR = INDEXES_DIR / "global" / "pages"
GLOBAL_PAGINATION = INDEXES_DIR / "global" / "pagination.json"
GLOBAL_STATS      = INDEXES_DIR / "global" / "stats.json"
MANIFEST_PATH     = INDEXES_DIR / "manifest.json"

SEARCH_TITLE_DIR  = INDEXES_DIR / "search" / "title"
SEARCH_TOKEN_MAP  = SEARCH_TITLE_DIR / "token_map.json"
SEARCH_SHARDS_DIR = SEARCH_TITLE_DIR / "shards"

DEDUP_DIR = INDEXES_DIR / "dedup"
DEDUP_STORY_DIR = DEDUP_DIR / "story" / "shards"
DEDUP_URL_DIR   = DEDUP_DIR / "url"   / "shards"

CHECKPOINTS_PATH = STATE_DIR / "checkpoints.json"

GLOBAL_PAGE_SIZE = 500

MAX_IMAGE_WIDTH  = 1280
MAX_IMAGE_HEIGHT = 1280
WEBP_QUALITY     = 85
HTTP_TIMEOUT     = 25

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def now_local() -> datetime:
    return datetime.now(TZ)

def iso_now() -> str:
    return now_local().isoformat()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def atomic_write_json(path: Path, obj):
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\u0600-\u06FF ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sha1_hex(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def shard_name(hex_str: str) -> str:
    return (hex_str[:2] if hex_str and len(hex_str) >= 2 else "00")

def build_raw_github_url(rel_path: str) -> str:
    return f"https://raw.githubusercontent.com/{GITHUB_REPO_SLUG}/{GITHUB_REPO_BRANCH}/{rel_path}"

def record_path_for(dt: datetime, rid: str) -> Path:
    return RECORDS_DIR / f"{dt.year}" / f"{dt.month:02d}" / f"{rid}.json"

def slugify(text: str, max_len: int = 70) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9\u0600-\u06FF\-]+", "", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:max_len] if text else "image"

# ====================
# Extractors
# ====================
def extract_full_text(entry) -> str:
    try:
        if hasattr(entry, "content") and entry.content and isinstance(entry.content, list):
            raw = entry.content[0].get("value") or ""
            if raw:
                return BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)
    except Exception:
        pass
    raw = getattr(entry, "description", "") or ""
    if raw:
        return BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)
    return ""

def extract_image(entry) -> str | None:
    if hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
        try:
            return entry.media_thumbnail[0].get("url") or entry.media_thumbnail[0]["url"]
        except Exception:
            pass
    raw = ""
    try:
        if hasattr(entry, "content") and entry.content and isinstance(entry.content, list):
            raw = entry.content[0].get("value") or ""
    except Exception:
        pass
    if not raw:
        raw = getattr(entry, "description", "") or ""
    if raw:
        soup = BeautifulSoup(raw, "html.parser")
        img = soup.find("img")
        if img and img.has_attr("src"):
            return img["src"]
    return None

def extract_categories(entry) -> list:
    cats = []
    tags = getattr(entry, "tags", None)
    if tags:
        for t in tags:
            term = getattr(t, "term", None)
            if term:
                cats.append(str(term))
    return cats

def entry_link(entry) -> str:
    return (getattr(entry, "link", "") or "").strip()

def entry_published_day(entry) -> str:
    for k in ["published", "updated"]:
        v = getattr(entry, k, None)
        if v:
            return str(v)[:10]
    return now_local().date().isoformat()

# ====================
# Dedup (stored sources only)
# ====================
def compute_story_fp(title: str, desc: str, pub_day: str) -> str:
    t = normalize_text(title)
    d = normalize_text(desc)[:160]
    return sha1_hex(f"{t}|{d}|{pub_day}")

def compute_url_hash(url: str) -> str:
    return sha1_hex((url or "").strip())

def load_shard_map(dir_path: Path, shard: str) -> dict:
    return read_json(dir_path / f"{shard}.json", default={})

def save_shard_map(dir_path: Path, shard: str, data: dict):
    ensure_dir(dir_path)
    atomic_write_json(dir_path / f"{shard}.json", data)

def dedup_lookup_story(story_hash: str) -> str | None:
    mp = load_shard_map(DEDUP_STORY_DIR, shard_name(story_hash))
    return mp.get(story_hash)

def dedup_put_story(story_hash: str, rid: str):
    sh = shard_name(story_hash)
    mp = load_shard_map(DEDUP_STORY_DIR, sh)
    mp[story_hash] = rid
    save_shard_map(DEDUP_STORY_DIR, sh, mp)

def dedup_lookup_url(url_hash: str) -> str | None:
    mp = load_shard_map(DEDUP_URL_DIR, shard_name(url_hash))
    return mp.get(url_hash)

def dedup_put_url(url_hash: str, rid: str):
    sh = shard_name(url_hash)
    mp = load_shard_map(DEDUP_URL_DIR, sh)
    mp[url_hash] = rid
    save_shard_map(DEDUP_URL_DIR, sh, mp)

# ====================
# Image processing for stored sources only
# ====================
def fetch_image(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        im = Image.open(BytesIO(r.content))
        im = ImageOps.exif_transpose(im)
        return im.convert("RGBA")
    except Exception as e:
        logging.warning(f"fetch_image failed: {e}")
        return None

def downscale_to_fit(im: Image.Image) -> Image.Image:
    w, h = im.size
    scale = min((MAX_IMAGE_WIDTH / w) if w else 1, (MAX_IMAGE_HEIGHT / h) if h else 1, 1)
    if scale < 1:
        im = im.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.LANCZOS)
    return im

def make_webp(url: str) -> BytesIO | None:
    im = fetch_image(url)
    if im is None:
        return None
    im = downscale_to_fit(im)
    out = BytesIO()
    im.convert("RGB").save(out, format="WEBP", quality=WEBP_QUALITY, method=6)
    out.seek(0)
    return out

def save_webp(stable_key: str, title: str, webp_bytes: BytesIO, dt: datetime) -> str:
    out_dir = MEDIA_DIR / f"{dt.year}" / f"{dt.month:02d}"
    ensure_dir(out_dir)
    filename = f"{slugify(title)}-{stable_key[:12]}.webp"
    p = out_dir / filename
    if not p.exists():
        p.write_bytes(webp_bytes.read())
    return build_raw_github_url(p.as_posix())

# ====================
# Records (stored sources only)
# ====================
def load_record(rid: str, dt_guess: datetime) -> dict | None:
    p = record_path_for(dt_guess, rid)
    if p.exists():
        return read_json(p, default=None)
    year_dir = RECORDS_DIR / f"{dt_guess.year}"
    if year_dir.exists():
        for month_dir in sorted(year_dir.glob("[0-1][0-9]")):
            cand = month_dir / f"{rid}.json"
            if cand.exists():
                return read_json(cand, default=None)
    return None

def save_record(rid: str, dt: datetime, rec: dict) -> Path:
    p = record_path_for(dt, rid)
    atomic_write_json(p, rec)
    return p

def merge_source(rec: dict, source_key: str, source_url: str) -> bool:
    sources = rec.get("sources") or []
    for s in sources:
        if s.get("source") == source_key and s.get("url") == source_url:
            return False
    sources.append({"source": source_key, "url": source_url, "added_at": iso_now()})
    rec["sources"] = sources
    rec["updated_at"] = iso_now()
    return True

# ====================
# Global indexes (stored sources only)
# ====================
def load_pagination() -> dict:
    return read_json(GLOBAL_PAGINATION, default={"total_articles": 0, "files": []})

def save_pagination(pag: dict):
    atomic_write_json(GLOBAL_PAGINATION, pag)

def save_stats(total: int, added: int):
    atomic_write_json(GLOBAL_STATS, {"total_articles": total, "added_today": added, "last_update": iso_now()})

def slim_card(rec: dict, record_path: str) -> dict:
    return {
        "id": rec.get("id"),
        "title": rec.get("title"),
        "image": rec.get("image"),
        "categories": rec.get("categories") or [],
        "created_at": rec.get("created_at"),
        "updated_at": rec.get("updated_at"),
        "record_path": record_path,
    }

def append_global_cards(cards: list):
    if not cards:
        return
    ensure_dir(GLOBAL_PAGES_DIR)
    pag = load_pagination()
    if not pag["files"]:
        atomic_write_json(GLOBAL_PAGES_DIR / "index_1.json", [])
        pag["files"].append("index_1.json")

    current_name = pag["files"][-1]
    current_file = GLOBAL_PAGES_DIR / current_name
    items = read_json(current_file, default=[])

    if len(items) >= GLOBAL_PAGE_SIZE:
        next_idx = len(pag["files"]) + 1
        current_name = f"index_{next_idx}.json"
        current_file = GLOBAL_PAGES_DIR / current_name
        items = []
        atomic_write_json(current_file, items)
        pag["files"].append(current_name)

    items.extend(cards)
    atomic_write_json(current_file, items)

    pag["total_articles"] = (pag.get("total_articles") or 0) + len(cards)
    save_pagination(pag)
    save_stats(pag["total_articles"], len(cards))

# ====================
# Search (stored sources only)
# ====================
def load_token_map() -> dict:
    return read_json(SEARCH_TOKEN_MAP, default={})

def save_token_map(mp: dict):
    atomic_write_json(SEARCH_TOKEN_MAP, mp)

def load_search_shard(sh: str) -> list:
    return read_json(SEARCH_SHARDS_DIR / f"{sh}.json", default=[])

def save_search_shard(sh: str, items: list):
    ensure_dir(SEARCH_SHARDS_DIR)
    atomic_write_json(SEARCH_SHARDS_DIR / f"{sh}.json", items)

def title_tokens(title: str, max_words: int = 6) -> set:
    t = normalize_text(title)
    words = [w for w in t.split(" ") if w]
    out = set()
    for w in words[:max_words]:
        if len(w) >= 2:
            out.add(w[:2])
    return out

def upsert_search_entry(rec: dict, record_path: str):
    rid = rec.get("id") or ""
    sh = shard_name(rid)
    items = load_search_shard(sh)
    for it in items:
        if it.get("id") == rid:
            it.update({"t": rec.get("title"), "d": rec.get("created_at"), "img": rec.get("image"), "cats": rec.get("categories") or [], "rp": record_path})
            save_search_shard(sh, items)
            return
    items.append({"id": rid, "t": rec.get("title"), "d": rec.get("created_at"), "img": rec.get("image"), "cats": rec.get("categories") or [], "rp": record_path})
    save_search_shard(sh, items)

def update_token_map(rec: dict):
    rid = rec.get("id") or ""
    shfile = f"{shard_name(rid)}.json"
    mp = load_token_map()
    for tok in title_tokens(rec.get("title") or ""):
        arr = mp.get(tok) or []
        if shfile not in arr:
            arr.append(shfile)
            mp[tok] = sorted(set(arr))
    save_token_map(mp)

# ====================
# Manifest (updated LAST)
# ====================
def save_manifest():
    pag = load_pagination()
    latest = None
    if pag.get("files"):
        latest = (GLOBAL_PAGES_DIR / pag["files"][-1]).as_posix()
    manifest = {
        "version": 1,
        "updated_at": iso_now(),
        "records_base": "records/",
        "global": {
            "stats_path": GLOBAL_STATS.as_posix(),
            "pagination_path": GLOBAL_PAGINATION.as_posix(),
            "latest_page_path": latest,
            "pages_base": GLOBAL_PAGES_DIR.as_posix() + "/",
        },
        "search": {
            "title": {
                "token_map_path": SEARCH_TOKEN_MAP.as_posix(),
                "shards_base": SEARCH_SHARDS_DIR.as_posix() + "/",
            }
        }
    }
    atomic_write_json(MANIFEST_PATH, manifest)

# ====================
# State checkpoints (all sources)
# ====================
def load_checkpoints() -> dict:
    return read_json(CHECKPOINTS_PATH, default={})

def save_checkpoints(cp: dict):
    atomic_write_json(CHECKPOINTS_PATH, cp)

def source_marker(entry) -> str:
    vid = getattr(entry, "id", None) or ""
    if vid:
        return str(vid)
    link = entry_link(entry)
    if link:
        return link
    title = getattr(entry, "title", "") or ""
    return sha1_hex(title)[:16]

# ====================
# Telegram send (FIXED + better logging)
# ====================
async def send_telegram_message(bot: telegram.Bot, title: str, description: str, image_url: str | None, link_url: str | None = None):
    if not TELEGRAM_CHAT_ID:
        logging.error("TELEGRAM_CHAT_ID is missing.")
        return

    text_parts = [title.strip() if title else ""]
    if description:
        text_parts.append(description.strip())
    if link_url:
        text_parts.append(f"🔗 {link_url}")
    caption = "\n\n".join([p for p in text_parts if p])

    try:
        if image_url:
            # Telegram can accept an https URL; if it fails, fallback to text
            try:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=image_url, caption=caption[:1024])
                return
            except Exception as e:
                logging.warning(f"send_photo failed, fallback to text. error={e}")

        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=caption[:4096])
    except Exception as e:
        logging.error(f"Telegram send failed: {e}")

async def send_record_to_telegram(bot: telegram.Bot, rec: dict):
    rid = rec.get("id") or ""
    article_url = f"{SITE_BASE_URL}/{ARTICLE_PAGE}?id={quote(rid)}" if SITE_BASE_URL else None
    await send_telegram_message(
        bot=bot,
        title=rec.get("title") or "",
        description=(rec.get("description_full") or "")[:700],
        image_url=rec.get("image"),
        link_url=article_url,
    )

# ====================
# Base folders
# ====================
def ensure_base_folders():
    for p in [RECORDS_DIR, INDEXES_DIR, STATE_DIR, MEDIA_DIR, GLOBAL_PAGES_DIR, SEARCH_SHARDS_DIR, DEDUP_STORY_DIR, DEDUP_URL_DIR]:
        ensure_dir(p)

# ====================
# Main processing
# ====================
def process_entry_store(source_key: str, entry) -> tuple[bool, str | None]:
    """Store + index + search + dedup + (optionally) telegram."""
    title = getattr(entry, "title", "") or ""
    desc  = extract_full_text(entry)
    img0  = extract_image(entry)
    cats  = extract_categories(entry)
    url   = entry_link(entry)
    pub_day = entry_published_day(entry)

    url_hash = compute_url_hash(url) if url else None
    if url_hash:
        ex = dedup_lookup_url(url_hash)
        if ex:
            rec = load_record(ex, now_local())
            if rec and merge_source(rec, source_key, url):
                save_record(ex, now_local(), rec)
            return False, ex

    story_hash = compute_story_fp(title, desc, pub_day)
    ex = dedup_lookup_story(story_hash)
    if ex:
        rec = load_record(ex, now_local())
        if rec and merge_source(rec, source_key, url or ""):
            save_record(ex, now_local(), rec)
        if url_hash:
            dedup_put_url(url_hash, ex)
        return False, ex

    rid = story_hash[:12]
    created_iso = iso_now()
    rec = {
        "id": rid,
        "title": title,
        "description_full": desc,
        "image": img0,
        "categories": cats or [],
        "created_at": created_iso,
        "updated_at": created_iso,
        "sources": [{"source": source_key, "url": url or "", "added_at": created_iso}],
    }

    # Only stored sources save webp to repo
    if img0:
        webp = make_webp(img0)
        if webp:
            rec["image"] = save_webp(rid, title, webp, now_local())
            rec["updated_at"] = iso_now()

    p = save_record(rid, now_local(), rec)
    rel_record_path = p.as_posix()

    dedup_put_story(story_hash, rid)
    if url_hash:
        dedup_put_url(url_hash, rid)

    append_global_cards([slim_card(rec, rel_record_path)])
    upsert_search_entry(rec, rel_record_path)
    update_token_map(rec)

    return True, rid

async def process_entry_telegram_only(source_key: str, entry, bot: telegram.Bot) -> bool:
    """Telegram-only: do NOT store anything in json/indexes."""
    title = getattr(entry, "title", "") or ""
    desc  = extract_full_text(entry)
    img   = extract_image(entry)
    url   = entry_link(entry)

    await send_telegram_message(
        bot=bot,
        title=title,
        description=desc[:700],
        image_url=img,
        link_url=url or None,
    )
    return True

async def run():
    ensure_base_folders()
    cp = load_checkpoints()

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram secrets missing. Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in repo secrets.")
    bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

    stored_new = 0
    for src in SOURCES:
        key = src["key"]
        store = bool(src.get("store", True))
        rss = src["rss"]

        feed = feedparser.parse(rss)
        if not feed.entries:
            logging.info(f"{key}: no entries")
            continue

        latest = feed.entries[0]
        marker = source_marker(latest)
        last_marker = (cp.get(key) or {}).get("last_marker")

        if last_marker and marker == last_marker:
            logging.info(f"{key}: skip (no new)")
            continue

        if store:
            created, rid = process_entry_store(key, latest)
            if created:
                stored_new += 1
                if bot:
                    rec = load_record(rid, now_local()) if rid else None
                    if rec:
                        await send_record_to_telegram(bot, rec)
        else:
            # Telegram-only (youtube): send even if not storing
            if bot:
                await process_entry_telegram_only(key, latest, bot)
            else:
                logging.warning(f"{key}: telegram-only source but TELEGRAM_TOKEN missing, cannot send.")

        cp[key] = {"last_marker": marker, "last_seen_at": iso_now()}

    save_checkpoints(cp)

    # Only update manifest if we have global index (stored sources). Safe to call always.
    save_manifest()  # LAST
    logging.info(f"Done. Stored new records: {stored_new}")

if __name__ == "__main__":
    asyncio.run(run())
