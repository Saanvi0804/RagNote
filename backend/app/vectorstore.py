# backend/app/vectorstore.py
import os
from typing import List, Any
import numpy as np

# sentence-transformers for local embeddings
from sentence_transformers import SentenceTransformer

# faiss for vector index
try:
    import faiss
except Exception as exc:
    raise ImportError("faiss not installed. Install faiss-cpu (or an appropriate wheel) in your venv.") from exc


# -------------------------
# Local embeddings wrapper
# -------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class LocalEmbeddings:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        # lazy load model on instantiation
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of texts to embeddings (list of lists).
        """
        embs = self.model.encode(texts, show_progress_bar=False)
        # convert to python lists
        return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]

    def embed_query(self, text: str) -> List[float]:
        v = self.model.encode([text], show_progress_bar=False)[0]
        return v.tolist() if hasattr(v, "tolist") else list(v)


# -------------------------
# Utility paths
# -------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _meta_path(store_path: str) -> str:
    return os.path.join(store_path, "metadata.txt")


def _index_path(store_path: str) -> str:
    return os.path.join(store_path, "index.faiss")


def _vectors_path(store_path: str) -> str:
    return os.path.join(store_path, "vectors.npy")


# -------------------------
# Robust retriever builder
# -------------------------
def _make_retriever(index_path: str, meta_path: str, embedder: LocalEmbeddings, default_k: int = 4):
    """
    Returns a callable retriever(query, k_override=None) -> list[dict(page_content, metadata)]
    """
    # Ensure index exists
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    # load index
    index = faiss.read_index(index_path)

    # read metadata lines once
    try:
        with open(meta_path, "r", encoding="utf8") as mf:
            # remove newline characters only — keep content
            meta_lines = [ln.rstrip("\n\r") for ln in mf.readlines()]
    except FileNotFoundError:
        meta_lines = []

    def retriever_func(query: str, k_override: int = None):
        kq = k_override or default_k
        vec = np.array(embedder.embed_query(query)).astype("float32").reshape(1, -1)
        D, I = index.search(vec, kq)

        docs = []
        for idx in I[0]:
            if idx < 0:
                # FAISS returns -1 for empty result slots
                continue
            preview = ""
            if 0 <= idx < len(meta_lines):
                line = meta_lines[idx].strip()
                if not line:
                    preview = ""
                else:
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        preview = parts[1]
                    else:
                        # fallback: take entire line as preview
                        preview = parts[0]
            else:
                preview = ""
            docs.append({"page_content": preview, "metadata": {"source_idx": int(idx)}})
        return docs

    return retriever_func


# -------------------------
# Create embeddings and store
# -------------------------
def create_embeddings_and_store(text: str, store_path: str = "faiss_store"):
    """
    Splits text into chunks, computes embeddings using sentence-transformers,
    and stores them in a FAISS index (persisted on disk). Also saves metadata.
    Returns a DB-like object exposing .as_retriever(search_kwargs={"k": ...})
    """
    _ensure_dir(store_path)

    # Simple deterministic splitter: chunk_size chars with overlap
    chunk_size = 1000
    overlap = 150
    chunks: List[str] = []
    i = 0
    text_len = len(text)
    while i < text_len:
        chunk = text[i: i + chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap

    embedder = LocalEmbeddings()
    vectors = embedder.embed_documents(chunks)  # list[list[float]]
    xb = np.array(vectors).astype("float32")

    # build faiss index
    d = xb.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    faiss.write_index(index, _index_path(store_path))

    # save vectors (numpy) as fallback
    np.save(_vectors_path(store_path), xb)

    # write metadata lines: one per chunk; format "idx\tpreview"
    with open(_meta_path(store_path), "w", encoding="utf8") as mf:
        for idx, chunk in enumerate(chunks):
            preview = chunk.replace("\n", " ").strip()
            # ensure no tabs in preview (we'll treat first tab-separated part as index)
            preview = preview.replace("\t", " ")
            mf.write(f"{idx}\t{preview}\n")

    # return a simple DB-like object
    class _DB:
        def __init__(self, index_path, meta_path, embedder, default_k=4):
            self._index_path = index_path
            self._meta_path = meta_path
            self._embedder = embedder
            self._default_k = default_k
            # Note: we don't load the faiss index here to keep memory use low;
            # _make_retriever will read index when called.
        def as_retriever(self, search_kwargs=None):
            sk = search_kwargs or {}
            k = sk.get("k", self._default_k)
            retr = _make_retriever(self._index_path, self._meta_path, self._embedder, default_k=k)
            return lambda query: retr(query, k_override=k)

    return _DB(_index_path(store_path), _meta_path(store_path), embedder, default_k=4)


# -------------------------
# Load existing vectorstore
# -------------------------
def load_vectorstore(store_path: str = "faiss_store"):
    """
    Loads the existing FAISS index and metadata and returns a DB-like object with .as_retriever(...)
    """
    if not os.path.exists(store_path):
        return None

    index_path = _index_path(store_path)
    meta_path = _meta_path(store_path)
    vectors_path = _vectors_path(store_path)

    if not os.path.exists(index_path):
        # maybe user used another format; return None
        return None

    # create embeddings object
    embedder = LocalEmbeddings()

    class _DB:
        def __init__(self, index_path, meta_path, embedder, default_k=4):
            self._index_path = index_path
            self._meta_path = meta_path
            self._embedder = embedder
            self._default_k = default_k

        def as_retriever(self, search_kwargs=None):
            sk = search_kwargs or {}
            k = sk.get("k", self._default_k)
            retr = _make_retriever(self._index_path, self._meta_path, self._embedder, default_k=k)
            return lambda query: retr(query, k_override=k)

    return _DB(index_path, meta_path, embedder, default_k=4)
