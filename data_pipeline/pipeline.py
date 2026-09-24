"""
Module 1 - Data Pipeline: Clean -> Convert -> Load -> Query
Reads raw_books.csv (produced by scrape.py), cleans/types the fields,
converts GBP -> INR using the project's fixed baseline rate, builds a
normalized two-table SQLite schema, loads the data, runs >= 5 SQL queries
(including a JOIN), and cross-checks one query result against an
equivalent pandas merge.

Run:
    python scrape.py        # produces raw_books.csv
    python pipeline.py      # produces books.db + queries_output.txt

Fixed baseline conversion rate (required, keyless, project-defined constant):
    1 GBP = 105.50 INR
"""
import re
import sqlite3
import pandas as pd

GBP_TO_INR = 105.50  # fixed project baseline; NOT a live/historical market rate

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

RAW_CSV = "raw_books.csv"
DB_PATH = "books.db"
QUERY_LOG = "queries_output.txt"


# ---------------------------------------------------------------------------
# 1. Clean scraped fields into proper types
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # price -> price_gbp (float)
    def parse_price(p):
        try:
            return float(re.sub(r"[^\d.]", "", str(p)))
        except ValueError:
            return None

    df["price_gbp"] = df["price"].apply(parse_price)

    # star_rating text -> rating int (1-5)
    df["rating"] = df["star_rating"].map(RATING_MAP)

    # availability text -> in_stock bool
    df["in_stock"] = df["availability"].str.contains("In stock", case=False, na=False)

    # Handle rows that failed to parse: median-impute numeric, drop if rating
    # (categorical/ordinal, can't be meaningfully median-imputed as a category
    # label) is unparseable.
    n_before = len(df)
    if df["price_gbp"].isna().any():
        median_price = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(median_price)
        print(f"  imputed {df['price_gbp'].isna().sum()} missing price_gbp values with median {median_price:.2f}")

    bad_rating = df["rating"].isna()
    if bad_rating.any():
        print(f"  dropping {bad_rating.sum()} rows with unparseable star_rating")
        df = df[~bad_rating]

    print(f"  rows: {n_before} -> {len(df)} after cleaning")

    # 2. Fixed-rate currency conversion (required, keyless baseline)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(int)  # SQLite has no native bool

    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


# ---------------------------------------------------------------------------
# 3. Normalized SQLite schema (two tables, PK/FK)
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER REFERENCES categories(category_id)
);
"""


def load_to_sqlite(df: pd.DataFrame, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript("DROP TABLE IF EXISTS books; DROP TABLE IF EXISTS categories;")
    cur.executescript(SCHEMA)

    categories = df["category"].unique().tolist()
    cat_rows = [(i + 1, name) for i, name in enumerate(categories)]
    cur.executemany("INSERT INTO categories (category_id, category_name) VALUES (?, ?)", cat_rows)
    cat_id_map = {name: i + 1 for i, name in enumerate(categories)}

    book_rows = [
        (
            row.title,
            row.price_gbp,
            row.price_inr,
            row.rating,
            row.in_stock,
            cat_id_map[row.category],
        )
        for row in df.itertuples(index=False)
    ]
    cur.executemany(
        """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        book_rows,
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# 5. Required SQL queries (>= 5, covering SELECT/WHERE, ORDER BY, LIMIT,
#    DISTINCT, IN/BETWEEN, plus >= 1 JOIN)
# ---------------------------------------------------------------------------
QUERIES = {
    "q1_cheap_instock": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE in_stock = 1 AND price_gbp < 20
        ORDER BY price_gbp ASC
        LIMIT 10;
    """,
    "q2_distinct_categories": """
        SELECT DISTINCT category_name
        FROM categories;
    """,
    "q3_rating_between": """
        SELECT title, rating, price_gbp
        FROM books
        WHERE rating BETWEEN 4 AND 5
        ORDER BY rating DESC, price_gbp ASC;
    """,
    "q4_specific_ratings_in": """
        SELECT title, rating
        FROM books
        WHERE rating IN (1, 5)
        ORDER BY rating;
    """,
    "q5_top_rated_per_category_join": """
        SELECT c.category_name, b.title, b.rating, b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name, b.rating DESC, b.price_gbp DESC
        LIMIT 10;
    """,
}


def run_queries(conn: sqlite3.Connection):
    outputs = {}
    with open(QUERY_LOG, "w", encoding="utf-8") as f:
        for name, sql in QUERIES.items():
            df_result = pd.read_sql(sql, conn)
            outputs[name] = df_result
            f.write(f"--- {name} ---\n{sql.strip()}\n\n{df_result.to_string(index=False)}\n\n")
            print(f"  {name}: {len(df_result)} rows")
    return outputs


# ---------------------------------------------------------------------------
# 6. Cross-check: pd.read_sql vs pd.merge for the JOIN query
# ---------------------------------------------------------------------------
def cross_check_join(conn: sqlite3.Connection, sql_join_result: pd.DataFrame):
    books = pd.read_sql("SELECT * FROM books", conn)
    categories = pd.read_sql("SELECT * FROM categories", conn)

    merged = pd.merge(books, categories, on="category_id")
    merged_sorted = (
        merged[["category_name", "title", "rating", "price_gbp"]]
        .sort_values(["category_name", "rating", "price_gbp"], ascending=[True, False, False])
        .head(10)
        .reset_index(drop=True)
    )
    sql_sorted = sql_join_result.reset_index(drop=True)

    match = merged_sorted.equals(sql_sorted)
    with open(QUERY_LOG, "a", encoding="utf-8") as f:
        f.write("--- cross-check: pd.read_sql JOIN vs pd.merge ---\n")
        f.write(f"Equivalent output: {match}\n\n")
        f.write("SQL/read_sql result:\n" + sql_sorted.to_string(index=False) + "\n\n")
        f.write("pd.merge result:\n" + merged_sorted.to_string(index=False) + "\n")
    print(f"  pd.read_sql vs pd.merge equivalent: {match}")
    return match


if __name__ == "__main__":
    print("Loading raw_books.csv ...")
    raw = pd.read_csv(RAW_CSV)

    print("Cleaning ...")
    clean_df = clean(raw)

    print("Loading into SQLite (books.db) ...")
    connection = load_to_sqlite(clean_df)

    print("Running SQL queries ...")
    results = run_queries(connection)

    print("Cross-checking JOIN query against pandas merge ...")
    cross_check_join(connection, results["q5_top_rated_per_category_join"])

    connection.close()
    print("Done. See books.db and queries_output.txt")
