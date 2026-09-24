# Module 1 — Data Pipeline

## Run
```bash
pip install -r requirements.txt
python scrape.py     # -> raw_books.csv (>=60 rows, >=3 categories)
python pipeline.py   # -> books.db, queries_output.txt
```

## Design decisions
- **Scraping**: `requests` + `BeautifulSoup` walk each category's paginated
  listing on books.toscrape.com until the "next" link disappears, so no
  category is truncated mid-page.
- **Currency conversion**: fixed baseline **1 GBP = 105.50 INR** (a
  project-defined constant, not a live rate — no network call, no API key).
- **Missing/unparseable data**: numeric fields (`price_gbp`) are
  median-imputed if a row fails to parse; rows with an unparseable
  categorical `rating` are dropped, since a rating can't be meaningfully
  median-imputed as a label. Both choices and their counts are logged to
  stdout when `pipeline.py` runs.
- **Schema**: two tables, `categories(category_id PK, category_name)` and
  `books(book_id PK, title, price_gbp, price_inr, rating, in_stock,
  category_id FK -> categories)`.
- **Queries**: 5 queries in `pipeline.py::QUERIES` cover
  `SELECT/WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `IN`/`BETWEEN`, and one
  `JOIN`. Output is written to `queries_output.txt`.
- **Cross-check**: the JOIN query's result is reproduced with
  `pd.merge` on in-memory DataFrames pulled back via `pd.read_sql`, and the
  two are compared for equality (also logged to `queries_output.txt`).
