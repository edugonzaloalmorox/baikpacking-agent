import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from baikpacking.scraper.browser import browser
from baikpacking.scraper.get_data import get_html, extract_article_links, parse_article

BASE = "https://dotwatcher.cc/features/bikes-of?page="

OUT_DIR = Path("data")
SNAP_DIR = OUT_DIR / "snapshots" / "raw"

OUT_JSONL = OUT_DIR / "dotwatcher_bikes_raw.jsonl"  # accumulated (append-only)
OUT_JSON = OUT_DIR / "dotwatcher_bikes_raw.json"    # regenerated snapshot (optional)

# Safety cap + early-stop in case there are no new articles
MAX_PAGES = 50


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_existing_urls(path: Path) -> set[str]:
    urls: set[str] = set()
    for row in _iter_jsonl(path):
        url = row.get("url")
        if url:
            urls.add(url)
    return urls


def main():
    OUT_DIR.mkdir(exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_jsonl = SNAP_DIR / f"dotwatcher_bikes_raw_new_{run_id}.jsonl"
    snap_json = SNAP_DIR / f"dotwatcher_bikes_raw_new_{run_id}.json"

    existing_urls = _load_existing_urls(OUT_JSONL)
    print(f"Existing articles in accumulated JSONL: {len(existing_urls)}")

    all_links: list[str] = []
    new_articles: list[dict] = []

    with browser() as page:
        # 1) Crawl index pages (early-stop when no new links appear)
        for page_num in range(MAX_PAGES):
            index_url = BASE + str(page_num)
            print(f"Fetching index page: {index_url}")

            index_html = get_html(page, index_url)
            links = extract_article_links(index_html)

            if not links:
                print("  No feature links found on this page, stopping crawl.")
                break

            # how many are actually new vs already scraped
            new_links_on_page = [u for u in links if u not in existing_urls]
            print(
                f"  Found {len(links)} feature links "
                f"({len(new_links_on_page)} new)"
            )

            all_links.extend(links)

            # 🔑 EARLY STOP: once a page yields 0 new links, the next pages are older
            if not new_links_on_page:
                print("  No new articles on this page, stopping crawl.")
                break

        # 2) Deduplicate links while preserving order
        all_links = list(dict.fromkeys(all_links))
        print(f"\nTotal unique feature links discovered: {len(all_links)}")

        # 3) Only scrape NEW urls
        to_scrape = [u for u in all_links if u not in existing_urls]
        print(f"New feature links to scrape: {len(to_scrape)}\n")

        for url in to_scrape:
            print(f"Scraping NEW article: {url}")
            html = get_html(page, url)
            data = parse_article(html)
            data["url"] = url
            new_articles.append(data)
            print("  →", data.get("title", ""))

    if not new_articles:
        print("No new articles found. Snapshot not created. Accumulated JSONL unchanged.")
        return

    # 4) Write snapshot containing ONLY new articles
    with snap_jsonl.open("w", encoding="utf-8") as f:
        for art in new_articles:
            json.dump(art, f, ensure_ascii=False)
            f.write("\n")

    with snap_json.open("w", encoding="utf-8") as f:
        json.dump(new_articles, f, ensure_ascii=False, indent=2)

    # 5) Append new articles to accumulated JSONL (append-only)
    with OUT_JSONL.open("a", encoding="utf-8") as f:
        for art in new_articles:
            json.dump(art, f, ensure_ascii=False)
            f.write("\n")

    # 6) Regenerate full JSON snapshot for inspection (optional)
    all_articles = list(_iter_jsonl(OUT_JSONL))
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print(f"\nAdded {len(new_articles)} new articles.")
    print(f"Snapshot JSONL (new only): {snap_jsonl}")
    print(f"Snapshot JSON  (new only): {snap_json}")
    print(f"Accumulated JSONL updated: {OUT_JSONL}")
    print(f"Full JSON snapshot updated: {OUT_JSON}")


if __name__ == "__main__":
    main()