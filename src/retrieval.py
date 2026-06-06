from dataclasses import dataclass
from typing import List, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import TextChunk


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved text chunk and its similarity score."""

    chunk: TextChunk
    score: float


class DenseRetriever:
    """
    Dense vector retriever using SentenceTransformers and FAISS.

    This class embeds text chunks, builds a FAISS index, and retrieves
    the most relevant chunks for a query using inner-product similarity.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: List[TextChunk] = []

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine-similarity style search."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vectors / norms

    def build_index(self, chunks: Sequence[TextChunk]) -> None:
        """
        Build a FAISS index over text chunks.

        Args:
            chunks: Text chunks to index.
        """
        if not chunks:
            raise ValueError("Cannot build index from an empty chunk list.")

        self.chunks = list(chunks)
        texts = [chunk.text for chunk in self.chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")

        embeddings = self._normalize(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve top-k chunks for a query.

        Args:
            query: User query.
            top_k: Number of chunks to return.

        Returns:
            Ranked retrieval results.
        """
        if self.index is None:
            raise RuntimeError("Index has not been built. Call build_index first.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
        ).astype("float32")

        query_embedding = self._normalize(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results: List[RetrievalResult] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                )
            )

        return results
