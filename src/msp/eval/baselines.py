from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank_chunks_tfidf(question: str, chunks: list[dict[str, str]], top_k: int = 5) -> list[str]:
    if not chunks:
        return []

    texts = [question] + [chunk["text"] for chunk in chunks]
    vectors = TfidfVectorizer().fit_transform(texts)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    ranked = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
    return [chunk["chunk_id"] for chunk, _ in ranked[:top_k]]


def rank_chunks_bm25(question: str, chunks: list[dict[str, str]], top_k: int = 5) -> list[str]:
    try:
        from rank_bm25 import BM25Okapi  # pyright: ignore[reportMissingImports]
    except ImportError:
        return rank_chunks_tfidf(question, chunks, top_k=top_k)

    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    scores = BM25Okapi(tokenized).get_scores(question.lower().split())
    ranked = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
    return [chunk["chunk_id"] for chunk, _ in ranked[:top_k]]


def bm25_baseline_record(example: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    pred_chunks = rank_chunks_bm25(example["question"], example["chunks"], top_k=top_k)
    return {
        "id": example["id"],
        "prediction_text": "",
        "pred_chunks": pred_chunks,
        "gold_support_chunks": example["gold_support_chunks"],
        "gold_slots": example.get("gold_slots", []),
        "valid_chunk_ids": example["valid_chunk_ids"],
    }
