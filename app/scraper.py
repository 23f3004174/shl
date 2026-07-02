from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
import re
import time

BASE = "https://www.shl.com"

START_URL = BASE + "/products/assessments/"


visited = set()
results = []


def clean(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_page(page, url):

    soup = BeautifulSoup(page.content(), "html.parser")

    title = clean(soup.title.text if soup.title else "")

    h1 = soup.find("h1")
    name = clean(h1.text) if h1 else title

    paragraphs = []

    for p in soup.find_all("p"):
        txt = clean(p.get_text())
        if len(txt) > 20:
            paragraphs.append(txt)

    description = " ".join(paragraphs[:5])

    page_text = clean(soup.get_text(" "))

    duration = ""

    m = re.search(r"(\d+\s*(minutes|minute|min))", page_text, re.I)

    if m:
        duration = m.group(1)

    results.append(
        {
            "name": name,
            "url": url,
            "description": description,
            "duration": duration,
            "content": page_text
        }
    )


def crawl(page, url, depth=0):
    print("Visiting:", url)

    page.goto(url, wait_until="domcontentloaded")

    page.wait_for_timeout(5000)

    print("\nTITLE:", page.title())

    links = page.locator("a")

    print("Total links:", links.count())

    for i in range(min(50, links.count())):
        try:
            href = links.nth(i).get_attribute("href")
            text = links.nth(i).inner_text()

            print(i, text[:40], "->", href)

        except Exception:
            pass


def main():

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page()

        page.goto(START_URL)

        try:
            page.get_by_role(
                "button",
                name="Allow all cookies"
            ).click(timeout=3000)
        except:
            pass

        crawl(page, START_URL)

        browser.close()

    with open(
        "data/assessments.json",
        "w",
        encoding="utf8"
    ) as f:

        json.dump(results, f, indent=4, ensure_ascii=False)

    print()

    print("=" * 60)

    print("Pages scraped:", len(results))

    print("=" * 60)


if __name__ == "__main__":
    main()