#!/usr/bin/env python3
"""Import Elliptic++ wallet labels into SQLite."""
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "ellipticpp_labels.db"
CSV_PATH = Path(__file__).parent.parent / "data" / "ellipticpp" / "wallets_classes.csv"

def import_labels():
    # Read CSV
    df = pd.read_csv(CSV_PATH)

    # Filter class 1 (illicit) and 2 (licit), drop 3 (unknown)
    df = df[df['class'].isin([1, 2])]

    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ellipticpp_labels (
            address TEXT PRIMARY KEY,
            class INTEGER NOT NULL
        )
    """)

    # Clear existing data
    cursor.execute("DELETE FROM ellipticpp_labels")

    # Insert
    df.to_sql('ellipticpp_labels', conn, if_exists='replace', index=False)

    # Index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_address ON ellipticpp_labels(address)")

    conn.commit()

    # Counts
    illicit = len(df[df['class'] == 1])
    licit = len(df[df['class'] == 2])
    print(f"Imported {illicit} illicit, {licit} licit addresses.")

    conn.close()

if __name__ == "__main__":
    import_labels()
