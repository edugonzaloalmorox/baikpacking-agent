import hashlib
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup



BASE_URL = "https://dotwatcher.cc"


def compute_hash(title: str, body: str) -> str:
    """Deterministic content hash used for incremental updates."""
    payload = (title.strip() + "\n" + body.strip()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_html(page, url: str) -> str:
    """Fetch HTML with Playwright page."""
    page.goto(url, wait_until="networkidle")
    return page.content()


def _normalize_feature_url(url: str) -> str:
    """Remove querystring/fragment to avoid duplicates and keep stable IDs."""
    parsed = urlparse(url)
    # keep scheme, netloc, path; drop params/query/fragment
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def extract_article_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        # allow both relative and absolute urls
        if href.startswith("/feature/") or href.startswith(f"{BASE_URL}/feature/"):
            full = urljoin(BASE_URL, href)
            links.append(_normalize_feature_url(full))

    # Deduplicate while preserving order
    return list(dict.fromkeys(links))


def parse_article(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1")
    article_tag = soup.find("article")

    title = title_tag.get_text(strip=True) if title_tag else ""
    body = article_tag.get_text("\n", strip=True) if article_tag else ""

    return {
        "title": title,
        "body": body,
        "content_hash": compute_hash(title, body),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }