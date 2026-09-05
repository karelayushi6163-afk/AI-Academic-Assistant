"""
utils/vector_store.py
The RAG (Retrieval-Augmented Generation) core: document chunking, embedding,
vector storage, and similarity-based retrieval.

Primary path (used when dependencies are installed -- i.e. in the deployed
app after `pip install -r requirements.txt`):
    - Chunking:   LangChain's RecursiveCharacterTextSplitter
    - Embeddings: HuggingFace sentence-transformers (all-MiniLM-L6-v2) --
                   runs locally, no API key required, genuine semantic vectors
    - Vector DB:  FAISS (in-memory, per-session)

Fallback path (used automatically if the above packages are unavailable, so
the app never crashes on a missing optional dependency):
    - Chunking:   a lightweight recursive character splitter re-implemented
                   with no external dependency
    - Embeddings: a pure numpy TF-IDF vectorizer
    - Vector DB:  an in-memory numpy matrix with cosine-similarity search

Both paths expose the exact same public interface (`VectorStore`), so the
rest of the app (agents/chat_agent.py) never needs to know which one is
active.
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> List[str]:
    """
    Split text into overlapping chunks. Tries LangChain's
    RecursiveCharacterTextSplitter first (splits on paragraph -> sentence ->
    word boundaries, which produces cleaner chunks); falls back to a
    dependency-free character-window splitter with the same chunk_size /
    chunk_overlap semantics if LangChain's splitter isn't installed.
    """
    text = (text or "").strip()
    if not text:
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)
    except ImportError:
        return _fallback_chunk_text(text, chunk_size, chunk_overlap)


def _fallback_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Dependency-free recursive-ish splitter: paragraphs -> fixed windows."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
            continue
        start = 0
        while start < len(para):
            end = start + chunk_size
            chunks.append(para[start:end])
            if end >= len(para):
                break
            start = end - chunk_overlap
    return [c for c in chunks if c.strip()]


# --------------------------------------------------------------------------
# Embedding backends
# --------------------------------------------------------------------------
class _SemanticEmbeddingBackend:
    """Real sentence-transformer embeddings (used when available)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def embed(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


class _TfidfEmbeddingBackend:
    """
    Pure-numpy TF-IDF fallback. Not semantic, but requires zero extra
    dependencies beyond numpy (already used by pandas/plotly in this app),
    so the RAG pipeline degrades gracefully instead of breaking outright.
    """

    name = "tfidf-fallback (numpy)"

    def __init__(self):
        self.vocab = {}
        self.idf = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9\+\#\.\-]{2,}", text.lower())

    def fit(self, corpus: List[str]):
        import numpy as np

        doc_freq = {}
        tokenized_docs = [self._tokenize(doc) for doc in corpus]
        for tokens in tokenized_docs:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self.vocab = {token: i for i, token in enumerate(sorted(doc_freq.keys()))}
        n_docs = max(len(corpus), 1)
        self.idf = np.zeros(len(self.vocab))
        for token, idx in self.vocab.items():
            self.idf[idx] = math.log((1 + n_docs) / (1 + doc_freq[token])) + 1.0

    def embed(self, texts: List[str]):
        import numpy as np

        vectors = np.zeros((len(texts), len(self.vocab)))
        for row, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                continue
            term_counts = {}
            for token in tokens:
                if token in self.vocab:
                    term_counts[token] = term_counts.get(token, 0) + 1
            max_count = max(term_counts.values()) if term_counts else 1
            for token, count in term_counts.items():
                idx = self.vocab[token]
                tf = 0.5 + 0.5 * (count / max_count)
                vectors[row, idx] = tf * self.idf[idx]
        # L2-normalize so cosine similarity == dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------
class VectorStore:
    """
    Unified RAG vector store: add_documents() chunks + embeds + indexes;
    similarity_search() retrieves the top-k most relevant chunks for a
    query. Automatically uses real semantic embeddings + FAISS when those
    packages are installed, otherwise transparently falls back to a
    numpy TF-IDF index.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Chunk] = []
        self._faiss_index = None
        self._embeddings_matrix = None
        self.backend_name = "not built yet"
        self.mode = None  # "semantic" or "tfidf"

    def add_documents(self, documents: List[dict]) -> None:
        """
        documents: list of {"text": str, "source": str, "metadata": dict}
        Chunks every document and rebuilds the index from scratch (simple
        and sufficient for a single-session career-assessment chat).
        """
        self.chunks = []
        for doc in documents:
            pieces = chunk_text(doc["text"], self.chunk_size, self.chunk_overlap)
            for piece in pieces:
                self.chunks.append(Chunk(
                    text=piece,
                    source=doc.get("source", "unknown"),
                    metadata=doc.get("metadata", {}),
                ))

        if not self.chunks:
            self._embeddings_matrix = None
            return

        self._build_index()

    def _build_index(self) -> None:
        texts = [c.text for c in self.chunks]

        try:
            backend = _SemanticEmbeddingBackend()
            vectors = backend.embed(texts)
            self.backend_name = f"semantic ({backend.name})"
            self.mode = "semantic"

            try:
                import faiss
                import numpy as np
                dim = vectors.shape[1]
                index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs == cosine
                index.add(np.asarray(vectors, dtype="float32"))
                self._faiss_index = index
                self._embeddings_matrix = np.asarray(vectors, dtype="float32")
                return
            except ImportError:
                # sentence-transformers available but faiss isn't -- keep
                # the semantic vectors, do similarity search with numpy.
                import numpy as np
                self._embeddings_matrix = np.asarray(vectors, dtype="float32")
                self._faiss_index = None
                return
        except ImportError:
            pass

        # Full fallback: TF-IDF + numpy cosine similarity.
        backend = _TfidfEmbeddingBackend()
        backend.fit(texts)
        self._embeddings_matrix = backend.embed(texts)
        self._tfidf_backend = backend
        self.backend_name = backend.name
        self.mode = "tfidf"

    def similarity_search(self, query: str, k: int = 4) -> List[dict]:
        """Return the top-k most relevant chunks for `query`."""
        import numpy as np

        if not self.chunks or self._embeddings_matrix is None:
            return []

        k = min(k, len(self.chunks))

        if self.mode == "semantic":
            backend = _SemanticEmbeddingBackend()
            query_vec = backend.embed([query])[0]

            if self._faiss_index is not None:
                scores, idxs = self._faiss_index.search(
                    np.asarray([query_vec], dtype="float32"), k
                )
                idxs, scores = idxs[0], scores[0]
            else:
                sims = self._embeddings_matrix @ query_vec
                idxs = np.argsort(-sims)[:k]
                scores = sims[idxs]
        else:
            query_vec = self._tfidf_backend.embed([query])[0]
            sims = self._embeddings_matrix @ query_vec
            idxs = np.argsort(-sims)[:k]
            scores = sims[idxs]

        results = []
        for idx, score in zip(idxs, scores):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk.text,
                "source": chunk.source,
                "metadata": chunk.metadata,
                "score": float(score),
            })
        return results

    def stats(self) -> dict:
        return {
            "num_chunks": len(self.chunks),
            "backend": self.backend_name,
            "mode": self.mode,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
