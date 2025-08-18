# retriever.py
import sqlite3
from utils import load_config, setup_logger
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from extractors.basic import extract_text
#from chunker import chunk_text
from tqdm import tqdm
from pathlib import Path
from extractors.images import extract_text_from_image, IMAGE_EXTS
cfg = load_config()
logger = setup_logger(cfg.get("LOG_LEVEL", "INFO"))
from sentence_transformers.cross_encoder import CrossEncoder
DB_PATH = cfg.get("SQLITE_DB", "index.db")
INDEX_DIR = cfg.get("INDEX_DIR", "./indices")
EMBED_MODEL = cfg.get("EMBED_MODEL", "all-MiniLM-L6-v2")
CROSS = cfg.get("CROSS_ENCODER", "cross-encoder/ms-marco-TinyBERT-L-2-v2")
BM25_TOPK = cfg.get("BM25_TOPK", 200)
DENSE_TOPK = cfg.get("DENSE_TOPK", 50)
RERANK_TOPK = cfg.get("RERANK_TOPK", 5)

class Retriever:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        # Load FAISS
        idx_path = f"{INDEX_DIR}/chunks_ivfpq.index"
        if not Path(idx_path).exists():
            raise FileNotFoundError(f"FAISS index not found at {idx_path}")
        self.index = faiss.read_index(idx_path)
        # Load cross encoder lazily
        try:
            
            self.cross = CrossEncoder(CROSS)
        except Exception:
            self.cross = None

    def _load_all_chunk_texts(self):
        c = self.conn.cursor()
        rows = c.execute("SELECT id, chunk_text FROM chunks").fetchall()
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        return ids, texts

    def bm25_prefilter(self, query, topk=BM25_TOPK):
        ids, texts = self._load_all_chunk_texts()
        tokenized = [t.split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:topk]
        pre_ids = [ids[i] for i in top_idx if scores[i] > 0]
        return pre_ids

    def dense_search(self, query, candidate_ids=None, topk=DENSE_TOPK):
        qv = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(qv, topk)
        hits = list(I[0])
        return hits, list(D[0])

    def search(self, query, topk=RERANK_TOPK):
        # Two-pass: BM25 -> Dense -> Rerank
        pre_ids = None
        try:
            pre_ids = self.bm25_prefilter(query)
        except Exception:
            logger.info("BM25 prefilter failed; skipping")

        hits, dists = self.dense_search(query, candidate_ids=pre_ids, topk=DENSE_TOPK)
        # Map hit ids to chunk ids stored in SQLite - note: FAISS idx id mapping depends on insertion order
        # We stored emb_id sequentially starting at 0 mapping to chunk.id primary key.
        # Retrieve chunk rows for hit indices
        c = self.conn.cursor()
        results = []
        for emb_id, dist in zip(hits, dists):
            row = c.execute("SELECT id, file_id, chunk_text, summary FROM chunks WHERE emb_id=?", (int(emb_id),)).fetchone()
            if row:
                chunk_id, file_id, text, summary = row
                file_row = c.execute("SELECT path FROM files WHERE id=?", (file_id,)).fetchone()
                path = file_row[0] if file_row else ""
                results.append({"chunk_id": chunk_id, "file_id": file_id, "path": path, "text": text, "score": float(dist), "summary": summary})
        # Re-rank with cross-encoder if available
        if self.cross and results:
            pairs = [[query, r["text"][:512]] for r in results[:RERANK_TOPK*2]]
            scores = self.cross.predict(pairs)
            for i, s in enumerate(scores):
                results[i]["cross_score"] = float(s)
            results.sort(key=lambda r: r.get("cross_score", r["score"]), reverse=True)
        else:
            results = sorted(results, key=lambda r: r["score"], reverse=True)
        return results[:topk]
