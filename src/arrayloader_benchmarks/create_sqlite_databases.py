from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("/dss/mcmlscratch/04/di93zer/benchmarks.sqlite")


def create_store_table(db_path: Path):
    """Create SQLite database with a table to store benchmark results."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmarks
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_path TEXT NOT NULL,
            store_type TEXT NOT NULL,
            gene_space TEXT NOT NULL,
            zarr_chunk_size INTEGER NOT NULL,
            zarr_shard_size INTEGER NOT NULL,
            anndata_shard_size INTEGER NOT NULL,
            creation_time REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def create_results_table(db_path: Path):
    """Create SQLite database with a table to store benchmark results."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_path TEXT NOT NULL,
            store_type TEXT NOT NULL,
            n_samples INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            use_torch_loader BOOLEAN NOT NULL,
            chunk_size INTEGER NOT NULL,
            preload_nchunks INTEGER NOT NULL,
            num_workers INTEGER NOT NULL,
            samples_per_sec REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    if not DB_PATH.exists():
        print(f"Database {DB_PATH} does not exist. Creating it...")
        create_store_table(DB_PATH)
        create_results_table(DB_PATH)
    else:
        print(f"Database {DB_PATH} already exists.")
