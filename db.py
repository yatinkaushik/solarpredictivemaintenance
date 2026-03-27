"""
db.py — SQLite persistence layer for Solar AI
Appends only rows after the latest stored DATE (indexed range query).
Avoids full table scan and string-based date comparison.
"""
import sqlite3
import pandas as pd
from config import DB_PATH, DB_TABLE, DB_DATE_COL
from logger import get_logger

log = get_logger("solar_ai.db")

_CREATE_INDEX = f"""
CREATE INDEX IF NOT EXISTS idx_{DB_TABLE}_date ON {DB_TABLE} ({DB_DATE_COL});
"""

def save_to_db(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        log.warning("save_to_db called with empty DataFrame — skipping.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # ── Get max stored date using indexed query (no full scan) ────────────
        try:
            cursor.execute(f"SELECT MAX({DB_DATE_COL}) FROM {DB_TABLE}")
            row = cursor.fetchone()
            max_stored = pd.to_datetime(row[0]) if row and row[0] else None
        except sqlite3.OperationalError:
            max_stored = None   # Table doesn't exist yet

        df_copy = df.copy()
        df_copy[DB_DATE_COL] = pd.to_datetime(df_copy[DB_DATE_COL])

        # ── Append only rows strictly after last stored date ──────────────────
        if max_stored is not None:
            new_rows = df_copy[df_copy[DB_DATE_COL] > max_stored]
        else:
            new_rows = df_copy

        if new_rows.empty:
            log.info("DB up to date — no new rows to append.")
        else:
            new_rows.to_sql(DB_TABLE, conn, if_exists="append", index=False)
            # Ensure index exists for fast future lookups
            conn.execute(_CREATE_INDEX)
            conn.commit()
            log.info(f"Appended {len(new_rows):,} new rows to '{DB_TABLE}' in {DB_PATH}")

        conn.close()

    except sqlite3.Error as e:
        log.error(f"Database error in save_to_db: {e}")
    except Exception as e:
        log.error(f"Unexpected error in save_to_db: {e}")


def load_from_db() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            f"SELECT * FROM {DB_TABLE} ORDER BY {DB_DATE_COL}",
            conn, parse_dates=[DB_DATE_COL]
        )
        conn.close()
        log.info(f"Loaded {len(df):,} rows from '{DB_TABLE}'")
        return df
    except Exception as e:
        log.warning(f"Could not load from DB: {e} — returning empty DataFrame")
        return pd.DataFrame()