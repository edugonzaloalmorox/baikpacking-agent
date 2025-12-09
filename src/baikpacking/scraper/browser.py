
from contextlib import contextmanager
from playwright.sync_api import sync_playwright

@contextmanager
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        try:
            yield page
        finally:
            browser.close()
