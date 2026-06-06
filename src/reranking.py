from dataclasses import dataclass
from typing import List, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from retrieval import RetrievalResult


@dataclass(frozen=True)
class RerankedResult:
    """A re-ranked retrieval result with a ColBERT-style score."""

    result: RetrievalResult
    rerank_score: float


class ColBERTStyleReranker:
    """
    ColBERT-style token-level re-ranker.

    This module encodes the query and candidate passages with BERT,
    then scores each candidate using token-level max-sim interactions.
    It is intended as a lightweight educational implementation of the
    core idea behind late-interaction retrieval.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _encode(self, text: str) -> torch.Tensor:
        """
        Encode text into token embeddings.

        Args:
            text: Input text.

        Returns:
            Tensor of shape [num_tokens, hidden_dim].
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state.squeeze(0)

        # Remove [CLS] and [SEP] when possible.
        if token_embeddings.shape[0] > 2:
            token_embeddings = token_embeddings[1:-1]

        return torch.nn.functional.normalize(token_embeddings, p=2, dim=1)

    def score(self, query: str, passage: str) -> float:
        """
        Compute a ColBERT-style max-sim score between a query and passage.

        Args:
            query: User query.
            passage: Candidate passage.

        Returns:
            Late-interaction similarity score.
        """
        query_embeddings = self._encode(query)
        passage_embeddings = self._encode(passage)

        similarity = torch.matmul(query_embeddings, passage_embeddings.T)

        # For each query token, keep the best matching passage token.
        max_sim_per_query_token = similarity.max(dim=1).values

        return float(max_sim_per_query_token.sum().item())

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
    ) -> List[RerankedResult]:
        """
        Re-rank dense retrieval results using token-level interactions.

        Args:
            query: User query.
            results: Initial dense retrieval results.

        Returns:
            Results sorted by re-ranking score in descending order.
        """
        reranked: List[RerankedResult] = []

        for result in results:
            rerank_score = self.score(query, result.chunk.text)
            reranked.append(
                RerankedResult(
                    result=result,
                    rerank_score=rerank_score,
                )
            )

        return sorted(reranked, key=lambda item: item.rerank_score, reverse=True)
