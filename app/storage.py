"""SQLite-backed persistence for registered graphs and the shared registry.

Two pieces of state survive restarts:

* the shared sensitive-column registry (entries keyed by ``(source, column)``)
  together with its monotonically increasing ``generation`` (0 on a fresh
  database, incremented once per successful whole-list replacement);
* every registered graph: its original ``nodes``/``export`` submission and the
  review result produced under the registry generation the graph was last
  reviewed at.

All reads and writes of the in-process state go through one lock, and a
replacement persists the new entries, the new generation and every graph's new
review result in a single SQLite transaction: either the whole change commits
or nothing at all changes, so a registry can never be observed together with
stale conclusions.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Optional

# Default database location inside the container; docker-compose mounts a named
# volume there. Local runs create the file next to the working directory.
DEFAULT_DB_PATH = os.environ.get("RISK_API_DB", "data/export-risk.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sensitive_entries (
    source_id TEXT NOT NULL,
    column_name TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (source_id, column_name)
);
CREATE TABLE IF NOT EXISTS graphs (
    id TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL,
    nodes TEXT NOT NULL,
    export TEXT NOT NULL,
    review TEXT NOT NULL,
    review_generation INTEGER NOT NULL
);
"""


class Store:
    """A single SQLite database guarded by one process-wide lock."""

    def __init__(self, path: Optional[str] = DEFAULT_DB_PATH):
        self._path = path or ":memory:"
        self.lock = threading.RLock()
        if self._path != ":memory:":
            parent = os.path.dirname(os.path.abspath(self._path))
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(
            self._path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)
            if self._path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
        self._init_meta()

    def close(self) -> None:
        with self.lock:
            self._conn.close()

    # --------------------------------------------------------------- meta --

    def _init_meta(self) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES "
                "('generation', '0')"
            )

    def get_generation(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'generation'"
        ).fetchone()
        return int(row["value"])

    def set_generation(self, generation: int) -> None:
        self._conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'generation'",
            (str(generation),),
        )

    # ---------------------------------------------------------- registry --

    def get_entries(self) -> list[tuple[str, str]]:
        """Return the registry entries in submission (ordinal) order."""
        rows = self._conn.execute(
            "SELECT source_id, column_name FROM sensitive_entries "
            "ORDER BY ordinal"
        ).fetchall()
        return [(row["source_id"], row["column_name"]) for row in rows]

    def replace_entries(
        self,
        entries: list[tuple[str, str]],
        generation: int,
        graph_reviews: dict[str, dict[str, object]],
    ) -> None:
        """Persist entries + generation + every graph's review atomically.

        ``graph_reviews`` maps every registered graph id to its review under
        the *new* registry. The caller has already computed the reviews, so the
        transaction itself never runs review logic.
        """
        with self._conn:  # commits on success, rolls back on any exception
            self._conn.execute("DELETE FROM sensitive_entries")
            self._conn.executemany(
                "INSERT INTO sensitive_entries "
                "(source_id, column_name, ordinal) VALUES (?, ?, ?)",
                [
                    (source_id, column_name, ordinal)
                    for ordinal, (source_id, column_name) in enumerate(entries)
                ],
            )
            self.set_generation(generation)
            for graph_id, review in graph_reviews.items():
                self._conn.execute(
                    "UPDATE graphs SET review = ?, review_generation = ? "
                    "WHERE id = ?",
                    (json.dumps(review, ensure_ascii=False), generation, graph_id),
                )

    # ------------------------------------------------------------- graphs --

    def list_graph_rows(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, nodes, export, review, review_generation "
            "FROM graphs ORDER BY ordinal"
        ).fetchall()

    def get_graph_row(self, graph_id: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, nodes, export, review, review_generation "
            "FROM graphs WHERE id = ?",
            (graph_id,),
        ).fetchone()

    def insert_graph(
        self,
        graph_id: str,
        nodes: list[dict[str, object]],
        export: str,
        review: dict[str, object],
        generation: int,
    ) -> None:
        next_ordinal = self._conn.execute(
            "SELECT COALESCE(MAX(ordinal) + 1, 0) FROM graphs"
        ).fetchone()[0]
        with self._conn:
            self._conn.execute(
                "INSERT INTO graphs "
                "(id, ordinal, nodes, export, review, review_generation) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    graph_id,
                    next_ordinal,
                    json.dumps(nodes, ensure_ascii=False),
                    export,
                    json.dumps(review, ensure_ascii=False),
                    generation,
                ),
            )


# The shared process-wide store, created lazily so importing the app never
# touches disk (tests and scripts stay free to rebind ``store`` first).
store: Optional[Store] = None
_store_lock = threading.Lock()


def get_store() -> Store:
    global store
    if store is None:
        with _store_lock:
            if store is None:
                store = Store()
    return store


def reset_store(path: Optional[str] = DEFAULT_DB_PATH) -> Store:
    """Replace the process-wide store (tests use a throwaway database)."""
    global store
    with _store_lock:
        if store is not None:
            store.close()
        store = Store(path)
    return store
