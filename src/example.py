from chunking import chunk_documents
from retrieval import DenseRetriever
from reranking import ColBERTStyleReranker


def main() -> None:
    documents = [
        (
            "meeting_001",
            """
            The team discussed latency issues in the retrieval system.
            Engineers proposed improving embedding quality, reducing chunk size,
            and adding a re-ranking stage to improve search relevance.
            """,
        ),
        (
            "meeting_002",
            """
            The product team reviewed user feedback about the dashboard.
            Customers wanted faster filtering, clearer visualizations, and
            better export options for reports.
            """,
        ),
        (
            "meeting_003",
            """
            The machine learning team evaluated several model deployment options.
            The discussion covered inference latency, batching, monitoring,
            and rollback strategies for production systems.
            """,
        ),
    ]

    query = "How can we improve retrieval quality?"

    chunks = chunk_documents(documents, chunk_size=300, overlap=50)

    retriever = DenseRetriever()
    retriever.build_index(chunks)

    initial_results = retriever.search(query, top_k=3)

    print("Initial dense retrieval results:")
    for rank, result in enumerate(initial_results, start=1):
        print(f"{rank}. {result.chunk.document_id} | score={result.score:.4f}")
        print(result.chunk.text)
        print()

    reranker = ColBERTStyleReranker()
    reranked_results = reranker.rerank(query, initial_results)

    print("ColBERT-style re-ranked results:")
    for rank, result in enumerate(reranked_results, start=1):
        print(
            f"{rank}. {result.result.chunk.document_id} "
            f"| dense_score={result.result.score:.4f} "
            f"| rerank_score={result.rerank_score:.4f}"
        )
        print(result.result.chunk.text)
        print()


if __name__ == "__main__":
    main()
