"""
Module 1 - Data Pipeline: Scraping
Scrapes books.toscrape.com across at least 3 categories, producing >= 60 rows.
Fields captured: title, price (GBP, as listed), star_rating (text), availability (text), category.

Run:
    python scrape.py
Output:
    raw_books.csv  (in this folder)
"""
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE = "https://books.toscrape.com/"
HEADERS = {"User-Agent": "Mozilla/5.0 (capstone-scraper/1.0)"}


def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def list_categories(min_categories: int = 3):
    """Return a list of (category_name, category_url) tuples."""
    soup = get_soup(BASE)
    nav = soup.select("div.side_categories ul li ul li a")
    cats = []
    for a in nav:
        name = a.get_text(strip=True)
        href = BASE + a["href"]
        cats.append((name, href))
    # Prefer categories that actually contain books; skip empty edge cases.
    return cats[:max(min_categories, 3)] if len(cats) >= min_categories else cats


def scrape_category(name: str, url: str, min_books: int = 20):
    """Paginate through a category, scraping every book until pages run out
    or we have collected at least `min_books` (whichever comes first is fine,
    we just keep going until pagination ends)."""
    rows = []
    next_url = url
    while next_url:
        soup = get_soup(next_url)
        for article in soup.select("article.product_pod"):
            title = article.h3.a["title"]
            price_text = article.select_one("p.price_color").get_text(strip=True)
            star_rating = article.p["class"][1]  # e.g. "Three"
            availability = article.select_one("p.instock.availability").get_text(strip=True)
            rows.append(
                {
                    "title": title,
                    "price": price_text,       # e.g. "£53.74"
                    "star_rating": star_rating,  # e.g. "Three"
                    "availability": availability,  # e.g. "In stock"
                    "category": name,
                }
            )
        next_link = soup.select_one("li.next a")
        if next_link:
            next_url = next_url.rsplit("/", 1)[0] + "/" + next_link["href"]
        else:
            next_url = None
        time.sleep(0.3)  # polite delay
    return rows


def scrape_all(min_total: int = 60, min_categories: int = 3) -> pd.DataFrame:
    categories = list_categories(min_categories)
    all_rows = []
    for name, url in categories:
        rows = scrape_category(name, url)
        all_rows.extend(rows)
        print(f"  scraped {len(rows):3d} books from category '{name}'")
        if len(all_rows) >= min_total and len(set(r["category"] for r in all_rows)) >= min_categories:
            break
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    print("Scraping books.toscrape.com ...")
    df = scrape_all(min_total=60, min_categories=3)
    print(f"Total rows scraped: {len(df)} across {df['category'].nunique()} categories")
    df.to_csv("raw_books.csv", index=False)
    print("Saved raw_books.csv")
