import json
from pathlib import Path

from baikpacking.scraper.browser import browser
from baikpacking.scraper.get_data import get_html, extract_article_links, parse_article




BASE = "https://dotwatcher.cc/features/bikes-of?page="
OUT_DIR = Path("data")
OUT_JSONL = OUT_DIR / "dotwatcher_bikes_raw.jsonl"
OUT_JSON = OUT_DIR / "dotwatcher_bikes_raw.json"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    all_links = []
    articles = []

    with browser() as page:
        # 1) Crawl index pages
        for page_num in range(10): 
            index_url = BASE + str(page_num)
            print(f"Fetching index page: {index_url}")

            index_html = get_html(page, index_url)
            links = extract_article_links(index_html)

            print(f"  Found {len(links)} feature links on this page")
            all_links.extend(links)

        # 2) Deduplicate all links
        all_links = list(dict.fromkeys(all_links))
        print(f"\nTotal unique feature links: {len(all_links)}\n")

        # 3) For each article: fetch + parse
        for url in all_links:
            print(f"Scraping article: {url}")
            html = get_html(page, url)
            data = parse_article(html)

            data["url"] = url  # keep origin
            articles.append(data)

            print("  →", data["title"])

    print(f"\nScraped {len(articles)} articles. Saving to JSON...")

    # 4) Save as JSONL (one article per line) 
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for art in articles:
            json.dump(art, f, ensure_ascii=False)
            f.write("\n")

    # 5) Also save as a single JSON array (handy for inspection)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"Saved JSONL to: {OUT_JSONL}")
    print(f"Saved JSON  to: {OUT_JSON}")


if __name__ == "__main__":
    main()
