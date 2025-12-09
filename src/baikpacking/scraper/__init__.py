


if __name__ == "__main__":
    html = fetch_html_with_playwright(URL)
    soup = BeautifulSoup(html, "html.parser")

    # Quick sanity check: print title and a few links
    print("Page title:", soup.title.string if soup.title else "No title")

    for a in soup.find_all("a", href=True)[:10]:
        print(a.get_text(strip=True), "->", a["href"])
        