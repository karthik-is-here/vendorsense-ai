"""
database.py — SQLite setup, schema, seed data, and query helpers for VendorSense AI.

This module handles ONLY persistence: creating the database, inserting rows, and
reading them back out (as raw rows or as a pandas DataFrame for analysis). It
deliberately knows nothing about the LLM or about generating insights — that
"thinking" logic lives entirely in agent.py.

Why this separation matters for the pitch: the database is the agent's MEMORY.
Keeping memory separate from reasoning means agent.py can re-read the full
sales history and re-decide what matters after every single write, without
being coupled to how the data is stored.
"""

import random
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

DB_PATH = "vendorsense.db"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Create the sales table if it doesn't already exist. Safe to call every run."""
    conn = get_connection(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_text TEXT NOT NULL,
            item TEXT NOT NULL,
            quantity REAL,
            unit TEXT,
            price REAL NOT NULL,
            is_estimate INTEGER NOT NULL DEFAULT 0,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def insert_sale(
    item: str,
    quantity: float,
    unit: str,
    price: float,
    raw_text: str,
    is_estimate: bool = False,
    timestamp: str | None = None,
    db_path: str = DB_PATH,
) -> None:
    """Insert one parsed sale. Called by app.py right after the LLM parses a
    vendor's free-text entry — this is the agent 'writing to memory'."""
    conn = get_connection(db_path)
    ts = timestamp or datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """INSERT INTO sales (raw_text, item, quantity, unit, price, is_estimate, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (raw_text, item, quantity, unit, price, int(is_estimate), ts),
    )
    conn.commit()
    conn.close()


def get_sales_df(db_path: str = DB_PATH) -> pd.DataFrame:
    """Return the full sales history as a pandas DataFrame, newest first.
    This is what agent.py re-reads every time it re-runs analysis."""
    conn = get_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM sales ORDER BY timestamp DESC", conn)
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def has_data(db_path: str = DB_PATH) -> bool:
    conn = get_connection(db_path)
    count = conn.execute("SELECT COUNT(*) AS c FROM sales").fetchone()["c"]
    conn.close()
    return count > 0


def seed_sample_data(db_path: str = DB_PATH, force: bool = False) -> None:
    """
    Populate the DB with ~20 realistic sample sales so the dashboard looks
    alive for a demo/pitch without needing to type entries live on stage.

    Design choices that matter for the demo:
    - Onions appear often across the last 3 days -> triggers the low-stock
      frequency inference.
    - Rice appears once, ~7 days ago, and never again -> triggers the
      "hasn't sold in N days, consider a discount" stale-item insight.
    - Prices/quantities are spread so today vs. yesterday and this-week vs.
      last-week revenue trends are non-trivial (not flat, not always up).
    """
    if has_data(db_path) and not force:
        return

    now = datetime.now()

    # (days_ago, hour, item, quantity, unit, price_total_rupees, raw_text)
    sample_entries = [
        (9, 9, "onions", 4, "kg", 160, "sold 4kg onions for 160 rupees"),
        (8, 10, "tomatoes", 3, "kg", 90, "sold 3kg tomatoes for 90"),
        (8, 17, "onions", 3, "kg", 120, "sold 3kg onions for 120 rupees"),
        (7, 9, "rice", 5, "kg", 250, "sold 5kg rice for 250"),
        (7, 12, "tomatoes", 2, "kg", 60, "sold 2kg tomatoes for 60 rupees"),
        (6, 8, "onions", 5, "kg", 200, "sold 5kg onions for 200"),
        (6, 16, "potatoes", 4, "kg", 100, "sold 4kg potatoes for 100 rupees"),
        (5, 9, "tomatoes", 3, "kg", 90, "sold 3kg tomatoes for 90 rupees"),
        (5, 11, "onions", 2, "kg", 80, "sold 2kg onions for 80"),
        (5, 18, "chillies", 1, "kg", 60, "sold 1kg chillies for 60 rupees"),
        (4, 9, "potatoes", 3, "kg", 75, "sold 3kg potatoes for 75"),
        (4, 14, "tomatoes", 4, "kg", 120, "sold 4kg tomatoes for 120 rupees"),
        (3, 9, "onions", 4, "kg", 160, "sold 4kg onions for 160"),
        (3, 13, "onions", 3, "kg", 120, "sold 3kg onions for 120 rupees"),
        (2, 10, "tomatoes", 5, "kg", 150, "sold 5kg tomatoes for 150"),
        (2, 15, "chillies", 1, "kg", 55, "sold 1kg chillies for 55 rupees"),
        (1, 9, "onions", 3, "kg", 130, "sold 3kg onions for 130 rupees"),
        (1, 17, "tomatoes", 3, "kg", 95, "sold 3kg tomatoes for 95"),
        (0, 9, "onions", 5, "kg", 210, "sold 5kg onions for 210 rupees"),
        (0, 11, "potatoes", 2, "kg", 50, "sold 2kg potatoes for 50 rupees"),
    ]

    for days_ago, hour, item, qty, unit, price, raw in sample_entries:
        ts = (now - timedelta(days=days_ago)).replace(
            hour=hour, minute=random.randint(0, 59), second=0, microsecond=0
        )
        insert_sale(
            item=item,
            quantity=qty,
            unit=unit,
            price=price,
            raw_text=raw,
            is_estimate=False,
            timestamp=ts.isoformat(timespec="seconds"),
            db_path=db_path,
        )


if __name__ == "__main__":
    # Quick manual check: `python database.py` sets up and prints the seeded DB.
    init_db()
    seed_sample_data()
    print(get_sales_df().to_string())
