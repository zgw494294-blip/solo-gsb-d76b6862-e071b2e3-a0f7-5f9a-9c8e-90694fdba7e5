"""Registered-graph and shared-registry operations on top of the engine.

Every operation here runs under the store's single lock, so registration,
reads, registered previews and registry replacements take effect in one
observable total order. Within that order each decision uses the *same*
generation for the registry, the stored review conclusions and the review a
preview is gated on: a preview either precedes a replacement (old registry plus
old conclusion) or follows it (new registry plus freshly computed, newly stored
conclusion), never a mixture of the two.
"""

from __future__ import annotations

import json
import uuid

from .engine import preview as run_preview
from .engine import review as run_review
from .models import PreviewRequest, ReviewRequest
from .storage import get_store


class GraphNotFound(Exception):
    """No registered graph carries the given id (HTTP 404)."""


class StaleGeneration(Exception):
    """The replacement's expected generation is outdated (HTTP 409)."""

    def __init__(self, expected: int, current: int):
        super().__init__(
            f"expected generation {expected} does not match current "
            f"generation {current}"
        )
        self.expected = expected
        self.current = current


def _registry_snapshot() -> tuple[frozenset[tuple[str, str]], int]:
    db = get_store()
    return frozenset(db.get_entries()), db.get_generation()


def register_graph(
    nodes: list[dict[str, object]], export: str
) -> dict[str, object]:
    """Validate a whole submitted graph, review it under the current registry.

    The graph must pass exactly the same whole-graph validation as
    ``/api/review`` (an invalid graph is rejected, nothing is registered). Its
    sensitivity is the OR of the in-graph ``sensitive`` flags and the current
    shared registry. The original submitted ``nodes`` are stored verbatim.
    """
    # Parse/validate before taking the lock so model-level errors behave as on
    # the other endpoints; the engine re-validates under the registry snapshot.
    ReviewRequest.model_validate({"nodes": nodes, "export": export})

    db = get_store()
    with db.lock:
        extra_sensitive, generation = _registry_snapshot()
        request = ReviewRequest.model_validate(
            {"nodes": nodes, "export": export}
        )
        review_result = run_review(request, extra_sensitive)
        graph_id = uuid.uuid4().hex
        db.insert_graph(graph_id, nodes, export, review_result, generation)
        return {
            "id": graph_id,
            "nodes": nodes,
            "export": export,
            "generation": generation,
            "review": review_result,
        }


def read_graph(graph_id: str) -> dict[str, object]:
    """Return the original submission plus the current generation and review."""
    db = get_store()
    with db.lock:
        row = db.get_graph_row(graph_id)
        if row is None:
            raise GraphNotFound(graph_id)
        return {
            "id": row["id"],
            "nodes": _loads(row["nodes"]),
            "export": row["export"],
            "generation": row["review_generation"],
            "review": _loads(row["review"]),
        }


def preview_graph(
    graph_id: str,
    rows: dict[str, list[dict[str, object]]],
    conditions: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Preview a registered graph from sample inputs and the current registry.

    The stored graph supplies ``nodes``/``export``; the caller supplies only
    ``rows``/``conditions`` and all original preview rules apply unchanged.
    The graph is re-evaluated under the registry snapshot taken under the same
    lock as the stored review conclusions, so the block decision can never be
    based on a different registry than the conclusion returned.
    """
    db = get_store()
    with db.lock:
        row = db.get_graph_row(graph_id)
        if row is None:
            raise GraphNotFound(graph_id)
        extra_sensitive, generation = _registry_snapshot()
        request = PreviewRequest.model_validate(
            {
                "nodes": _loads(row["nodes"]),
                "export": row["export"],
                "rows": rows,
                "conditions": conditions,
            }
        )
        result = run_preview(request, extra_sensitive)
        result["id"] = graph_id
        result["generation"] = generation
        return result


def get_registry() -> dict[str, object]:
    db = get_store()
    with db.lock:
        return {
            "generation": db.get_generation(),
            "entries": [
                {"source": source_id, "column": column_name}
                for source_id, column_name in db.get_entries()
            ],
        }


def replace_registry(
    entries: list[tuple[str, str]], expected_generation: int
) -> dict[str, object]:
    """Replace the whole registry and re-review every registered graph.

    Stale ``expected_generation`` is rejected as :class:`StaleGeneration`
    without touching anything. Otherwise every registered graph is re-reviewed
    under the OR of its in-graph flags and the new entries, and entries, the
    generation increment and all new conclusions are committed in one
    transaction; any failure rolls the whole replacement back.
    """
    db = get_store()
    with db.lock:
        current = db.get_generation()
        if expected_generation != current:
            raise StaleGeneration(expected_generation, current)

        extra_sensitive = frozenset(entries)
        new_reviews: dict[str, dict[str, object]] = {}
        for row in db.list_graph_rows():
            request = ReviewRequest.model_validate(
                {"nodes": json.loads(row["nodes"]), "export": row["export"]}
            )
            new_reviews[row["id"]] = run_review(request, extra_sensitive)

        next_generation = current + 1
        # Entries, generation and conclusions commit together or not at all.
        db.replace_entries(entries, next_generation, new_reviews)
        return {
            "generation": next_generation,
            "entries": [
                {"source": source_id, "column": column_name}
                for source_id, column_name in entries
            ],
            "reviews": {graph_id: review for graph_id, review in new_reviews.items()},
        }


def _loads(value: str) -> object:
    return json.loads(value)
