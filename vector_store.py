import numpy as np
from sentence_transformers import SentenceTransformer

# Load a lightweight local model — no API key needed for embeddings
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def build_vector_store(chunks: list[str]) -> np.ndarray:
    """
    Embed all chunks and return the embedding matrix.
    Shape: (num_chunks, embedding_dim)
    """
    model = _get_model()
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


def search_similar_chunks(
    query: str,
    embeddings_store: np.ndarray,
    chunks: list[str],
    top_k: int = 4
) -> list[str]:
    """
    Find the top_k most relevant chunks for the query using cosine similarity.
    No external vector DB needed — pure numpy.
    """
    model = _get_model()
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Cosine similarity
    norms_store = np.linalg.norm(embeddings_store, axis=1, keepdims=True)
    norms_query = np.linalg.norm(query_embedding)
    similarities = (embeddings_store @ query_embedding.T).flatten() / (
        norms_store.flatten() * norms_query + 1e-10
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]
