#!/usr/bin/env python3
"""
Prepare Elliptic++ dataset: CSV → Parquet, aggregated and deduplicated.
Uses DuckDB for low-memory processing.
"""
import duckdb
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "ellipticpp"
OUTPUT_DIR = DATA_DIR

def main():
    con = duckdb.connect(":memory:")

    print("Processing wallets...")
    # 1. wallets.parquet: one row per address, take latest Time step
    con.execute(f"""
        COPY (
            WITH ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (PARTITION BY address ORDER BY "Time step" DESC) as rn
                FROM read_csv_auto('{DATA_DIR / "wallets_features.csv"}')
            ),
            latest AS (
                SELECT * EXCLUDE (rn) FROM ranked WHERE rn = 1
            ),
            minmax AS (
                SELECT
                    address,
                    MIN("Time step") as first_ts,
                    MAX("Time step") as last_ts
                FROM read_csv_auto('{DATA_DIR / "wallets_features.csv"}')
                GROUP BY address
            )
            SELECT
                l.*,
                c.class,
                m.first_ts,
                m.last_ts
            FROM latest l
            LEFT JOIN read_csv_auto('{DATA_DIR / "wallets_classes.csv"}') c USING (address)
            LEFT JOIN minmax m USING (address)
        ) TO '{OUTPUT_DIR / "wallets.parquet"}' (FORMAT PARQUET)
    """)

    print("✓ wallets.parquet created")

    # 2. tx_edges.parquet: AddrTx ⋈ TxAddr → (src, dst, txId), aggregated
    print("Processing tx_edges...")
    con.execute(f"""
        COPY (
            SELECT DISTINCT
                a.input_address as src,
                b.output_address as dst,
                a.txId,
                t."Time step" as time_step,
                tc.class as tx_class,
                t.total_BTC,
                t.fees
            FROM read_csv_auto('{DATA_DIR / "AddrTx_edgelist.csv"}') a
            JOIN read_csv_auto('{DATA_DIR / "TxAddr_edgelist.csv"}') b ON a.txId = b.txId
            LEFT JOIN read_csv_auto('{DATA_DIR / "txs_features.csv"}') t ON a.txId = t.txId
            LEFT JOIN read_csv_auto('{DATA_DIR / "txs_classes.csv"}') tc ON a.txId = tc.txId
            WHERE a.input_address != b.output_address
        ) TO '{OUTPUT_DIR / "tx_edges.parquet"}' (FORMAT PARQUET)
    """)
    print("✓ tx_edges.parquet created")

    # 3. addr_edges.parquet: GNN graph (AddrAddr_edgelist.csv)
    print("Processing addr_edges...")
    con.execute(f"""
        COPY (
            SELECT input_address, output_address
            FROM read_csv_auto('{DATA_DIR / "AddrAddr_edgelist.csv"}')
        ) TO '{OUTPUT_DIR / "addr_edges.parquet"}' (FORMAT PARQUET)
    """)
    print("✓ addr_edges.parquet created")

    # 4. tx_features.parquet: transaction features (for tx model)
    print("Processing tx_features...")
    # Read model_meta.json to get tx_features list
    import json
    meta_path = OUTPUT_DIR / "model" / "model_meta.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        tx_features = meta.get("tx_features", [])
        if tx_features:
            feature_cols = ", ".join([f'"{f}"' for f in ["txId"] + tx_features])
            con.execute(f"""
                COPY (
                    SELECT {feature_cols}
                    FROM read_csv_auto('{DATA_DIR / "txs_features.csv"}')
                ) TO '{OUTPUT_DIR / "tx_features.parquet"}' (FORMAT PARQUET)
            """)
            print(f"✓ tx_features.parquet created ({len(tx_features)} features)")
        else:
            print("⚠ No tx_features in model_meta.json, skipping tx_features.parquet")
    else:
        print("⚠ model_meta.json not found, skipping tx_features.parquet")

    con.close()
    print(f"✓ Complete: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
