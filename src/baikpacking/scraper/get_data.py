from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://dotwatcher.cc"


def get_html(page, url):
    page.goto(url, wait_until="networkidle")
    return page.content()


def extract_article_links(html):
    soup = BeautifulSoup(html, "html.parser")
    links = []

    # DotWatcher feature URLs always look like: /feature/<slug>
    for a in soup.find_all("a", href=True):
        href = a["href"]

        # keep only links that start with /feature/
        if href.startswith("/feature/"):
            full = urljoin(BASE_URL, href)
            links.append(full)
            
            
    

    return list(dict.fromkeys(links))


def parse_article(html):
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1")
    article_tag = soup.find("article")

    return {
        "title": title_tag.get_text(strip=True) if title_tag else "",
        "body": article_tag.get_text("\n", strip=True) if article_tag else "",
    }
