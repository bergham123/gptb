# Anime News Bot (v2 structure)

This version stores **one article per file** for safety + edits, and generates small **indexes** for fast website fetching + search.

## Folders

- `records/YYYY/MM/<id>.json`  
  Full article record (source of truth). Each record can contain multiple `sources[]` (when the same story appears on multiple sites).

- `indexes/manifest.json`  
  Entry-point for the website (updated **last** each run).

- `indexes/global/pages/index_N.json`  
  Feed pages (slim cards) for infinite scrolling.

- `indexes/search/title/`  
  Prefix search using `token_map.json` + 256 `shards/00..ff.json`.

- `indexes/dedup/`  
  Sharded maps to prevent duplicates:
  - `story/shards/00..ff.json` : `story_hash -> id`
  - `url/shards/00..ff.json`   : `url_hash -> id`

- `state/checkpoints.json`  
  Per-source last seen marker.

- `media/YYYY/MM/*.webp`  
  Optimized images.

## Website usage

1. Load `indexes/manifest.json`
2. Load `manifest.global.latest_page_path` for the homepage list
3. Search:
   - Load `indexes/search/title/token_map.json`
   - Fetch only the shards listed for the query token
4. Article page:
   - `article.html?id=<id>` -> fetch `records/.../<id>.json`
