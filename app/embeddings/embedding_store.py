from typing import Any, Dict, List, Tuple
import hashlib
import os
import pickle
import re

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from app.utils.settings import settings

# Embedding dimension for local fallback (no native DLLs)
_LOCAL_EMBED_DIM = 384


class _SimpleLocalEmbeddings(Embeddings):
    """Pure-Python embedding fallback when no API key and no native libs (e.g. ONNX) work."""

    def _embed_one(self, text: str) -> List[float]:
        tokens = re.findall(r"\b\w+\b", (text or "").lower())
        vec = [0.0] * _LOCAL_EMBED_DIM
        for t in tokens:
            h = int(hashlib.sha256(t.encode("utf-8")).hexdigest(), 16) % _LOCAL_EMBED_DIM
            vec[h] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class _SimpleVectorStore:
    """In-memory vector store with pickle persistence. No Chroma/ONNX dependency."""

    def __init__(self, persist_directory: str, collection_name: str, embedding_function: Embeddings) -> None:
        self._embeddings = embedding_function
        self._path = os.path.join(persist_directory, f"{collection_name}.pkl")
        self._docs: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.isfile(self._path):
            try:
                with open(self._path, "rb") as f:
                    self._docs = pickle.load(f)
            except Exception:
                self._docs = []

    def _save(self) -> None:
        with open(self._path, "wb") as f:
            pickle.dump(self._docs, f)

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        vectors = self._embeddings.embed_documents(texts)
        for text, meta, vec in zip(texts, metadatas, vectors):
            self._docs.append({"page_content": text, "metadata": meta or {}, "embedding": vec})
        self._save()

    def similarity_search_with_relevance_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if not self._docs:
            return []
        q_vec = self._embeddings.embed_query(query)
        scored = [(doc, _cosine_sim(q_vec, doc["embedding"])) for doc in self._docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]
        return [
            (Document(page_content=doc["page_content"], metadata=doc["metadata"]), score)
            for doc, score in top
        ]


def _get_embeddings() -> Embeddings:
    """Use OpenAI if key is set; else pure-Python local embeddings (no DLLs)."""
    if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip():
        from langchain_community.embeddings import OpenAIEmbeddings
        return OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL,
        )
    return _SimpleLocalEmbeddings()


def _use_chroma() -> bool:
    """Use Chroma only when OpenAI key is set; otherwise avoid Chroma (and its ONNX default)."""
    return bool(settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip())


class EmbeddingStore:
    def __init__(self) -> None:
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        self._embeddings = _get_embeddings()
        if _use_chroma():
            from langchain_community.vectorstores import Chroma
            self._store = Chroma(
                collection_name=settings.VECTOR_COLLECTION_NAME,
                embedding_function=self._embeddings,
                persist_directory=settings.VECTOR_DB_PATH,
            )
        else:
            self._store = _SimpleVectorStore(
                settings.VECTOR_DB_PATH,
                settings.VECTOR_COLLECTION_NAME,
                self._embeddings,
            )

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        self._store.add_texts(texts=texts, metadatas=metadatas)
        if hasattr(self._store, "persist"):
            self._store.persist()

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[Document, float]]:
        return self._store.similarity_search_with_relevance_scores(query, k=k)

