"""
Reciprocal Rank Fusion (RRF) — merges ranked lists from multiple retrievers.

RRF was introduced in the paper "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" (Cormack et al., 2009) and has become
the standard way to combine vector and keyword search results.

How it works:
    For each document d, across each ranked list L:
        rrf_score(d) = Σ  1 / (k + rank(d, L))

    where k=60 is a constant that dampens the impact of very high ranks.
    Documents not appearing in a list are simply skipped (not penalised).

Why it beats score normalisation:
    Vector scores (cosine) and BM25 scores live on completely different scales.
    Normalising them is fragile. RRF sidesteps this by only caring about rank
    position, not the raw score value — making it robust to any combination
    of retrievers.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    k: int = 60,
    id_key: str = "chunk_id",
) -> list[dict]:
    """
    Merge multiple ranked result lists into one using RRF.

    Args:
        *ranked_lists: Any number of ranked lists. Each list is a list of
                       dicts that must contain `id_key` and any other fields
                       you want carried through to the final result.
        k:             RRF constant (default 60, from the original paper).
                       Higher k → less weight to top-ranked items.
        id_key:        The dict key used to identify unique documents.

    Returns:
        A single merged list sorted by descending RRF score.
        Each item has all original fields plus "rrf_score" and "rrf_rank".

    Example:
        vector_results = [{"chunk_id": "a", ...}, {"chunk_id": "b", ...}]
        bm25_results   = [{"chunk_id": "b", ...}, {"chunk_id": "a", ...}]
        fused = reciprocal_rank_fusion(vector_results, bm25_results)
    """
    # Accumulate RRF scores and keep payload of each unique document
    scores:   dict[str, float] = {}
    payloads: dict[str, dict]  = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            doc_id = item[id_key]
            scores[doc_id]   = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            payloads[doc_id] = item   # last-write wins for payload fields

    # Sort by descending RRF score
    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)

    results = []
    for final_rank, doc_id in enumerate(sorted_ids, start=1):
        entry = {**payloads[doc_id], "rrf_score": scores[doc_id], "rrf_rank": final_rank}
        results.append(entry)

    return results


def explain_rrf(
    *ranked_lists: list[dict],
    k: int = 60,
    id_key: str = "chunk_id",
    top_k: int = 5,
) -> None:
    """
    Print a human-readable breakdown of how RRF scored each document.
    Useful for debugging and understanding why a result ranked where it did.

    Example output:
        RRF Explanation (k=60, 2 lists)
        ─────────────────────────────────────────────
        #1 chunk_id=abc  rrf=0.01639  (list1: rank 1 → 0.01639, list2: rank 3 → 0.01587)
        #2 chunk_id=def  rrf=0.01613  (list1: rank 2 → 0.01613, list2: -)
    """
    # Build rank lookup: doc_id → rank per list
    rank_lookup: dict[str, list[int | None]] = {}
    for list_idx, ranked_list in enumerate(ranked_lists):
        for rank, item in enumerate(ranked_list, start=1):
            doc_id = item[id_key]
            if doc_id not in rank_lookup:
                rank_lookup[doc_id] = [None] * len(ranked_lists)
            rank_lookup[doc_id][list_idx] = rank

    fused = reciprocal_rank_fusion(*ranked_lists, k=k, id_key=id_key)[:top_k]

    print(f"\nRRF Explanation (k={k}, {len(ranked_lists)} lists)")
    print("─" * 56)
    for item in fused:
        doc_id = item[id_key]
        ranks  = rank_lookup.get(doc_id, [None] * len(ranked_lists))
        parts  = []
        for i, r in enumerate(ranks):
            if r is not None:
                contribution = 1.0 / (k + r)
                parts.append(f"list{i+1}: rank {r} → {contribution:.5f}")
            else:
                parts.append(f"list{i+1}: -")
        detail = ",  ".join(parts)
        title  = item.get("title", "")[:30]
        print(
            f"  #{item['rrf_rank']:2d}  {doc_id}  "
            f"rrf={item['rrf_score']:.5f}  [{title}]  ({detail})"
        )
    print()