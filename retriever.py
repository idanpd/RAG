"""
Advanced retrieval system with hybrid search and reranking capabilities.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from utils import ConfigManager

# Optional imports with fallbacks
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers.cross_encoder import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a search result with metadata."""
    
    def __init__(self, chunk_id: int, file_id: int, path: str, text: str, 
                 summary: str, chunk_type: str, score: float, method: str = 'dense'):
        self.chunk_id = chunk_id
        self.file_id = file_id
        self.path = path
        self.text = text
        self.summary = summary
        self.chunk_type = chunk_type
        self.score = score
        self.method = method
        self.cross_score: Optional[float] = None
        self.final_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'chunk_id': self.chunk_id,
            'file_id': self.file_id,
            'path': self.path,
            'text': self.text,
            'summary': self.summary,
            'chunk_type': self.chunk_type,
            'score': self.score,
            'method': self.method,
            'cross_score': self.cross_score,
            'final_score': self.final_score or self.score
        }
    
    def __repr__(self):
        return f"SearchResult(chunk_id={self.chunk_id}, path='{self.path}', score={self.score:.4f})"


class DatabaseRetriever:
    """Handles database operations for retrieval."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
    
    def get_chunk_by_emb_id(self, emb_id: int) -> Optional[Dict[str, Any]]:
        """Get chunk information by embedding ID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT c.id, c.file_id, c.chunk_text, c.summary, c.chunk_type,
                   f.path, f.filename, f.file_type
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE c.emb_id = ?
        """, (emb_id,))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_all_chunks(self) -> List[Tuple[int, str]]:
        """Get all chunks with their embedding IDs."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT emb_id, chunk_text FROM chunks WHERE emb_id IS NOT NULL ORDER BY emb_id")
        return [(row['emb_id'], row['chunk_text']) for row in cursor.fetchall()]
    
    def get_chunks_by_emb_ids(self, emb_ids: List[int]) -> List[Dict[str, Any]]:
        """Get multiple chunks by their embedding IDs."""
        if not emb_ids:
            return []
        
        cursor = self.conn.cursor()
        placeholders = ','.join('?' * len(emb_ids))
        cursor.execute(f"""
            SELECT c.id, c.file_id, c.chunk_text, c.summary, c.chunk_type,
                   f.path, f.filename, f.file_type, c.emb_id
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE c.emb_id IN ({placeholders})
        """, emb_ids)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def log_search(self, query: str, results_count: int):
        """Log search query for analytics."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO search_history (query, results_count)
            VALUES (?, ?)
        """, (query, results_count))
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        self.conn.close()


class BM25Retriever:
    """BM25-based sparse retrieval for prefiltering."""
    
    def __init__(self, db_retriever: DatabaseRetriever):
        self.db_retriever = db_retriever
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[int] = []
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of BM25 index."""
        if self._initialized or not BM25_AVAILABLE:
            return
        
        logger.info("Initializing BM25 index...")
        chunks = self.db_retriever.get_all_chunks()
        
        if not chunks:
            logger.warning("No chunks found for BM25 initialization")
            return
        
        self.chunk_ids = [chunk_id for chunk_id, _ in chunks]
        texts = [text for _, text in chunks]
        
        # Tokenize texts for BM25
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        self._initialized = True
        
        logger.info(f"BM25 index initialized with {len(chunks)} chunks")
    
    def search(self, query: str, top_k: int = 200) -> List[int]:
        """Perform BM25 search and return chunk IDs (not embedding IDs)."""
        if not BM25_AVAILABLE:
            return []
        
        self._initialize()
        if not self._initialized or self.bm25 is None:
            return []
        
        try:
            # Get all chunks from database (like original)
            chunks = self.db_retriever.get_all_chunks()
            if not chunks:
                return []
            
            ids = [chunk_id for chunk_id, _ in chunks]
            texts = [text for _, text in chunks]
            
            # Tokenize texts for BM25
            tokenized = [text.split() for text in texts]
            bm25 = BM25Okapi(tokenized)
            
            # Get BM25 scores
            scores = bm25.get_scores(query.split())
            
            # Get top results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # Filter out zero scores and return chunk IDs (not embedding IDs)
            pre_ids = [ids[i] for i in top_indices if scores[i] > 0]
            return pre_ids
            
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []


class DenseRetriever:
    """Dense vector retrieval using FAISS."""
    
    def __init__(self, config: ConfigManager, index_dir: Path):
        self.config = config
        self.index_dir = index_dir
        
        # Initialize sentence transformer
        model_name = self.config.get('EMBED_MODEL', 'all-MiniLM-L6-v2')
        self.embedder = SentenceTransformer(model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Load FAISS index
        self.index = self._load_index()
        
        logger.info(f"Dense retriever initialized with model: {model_name}")
    
    def _load_index(self) -> Optional[faiss.Index]:
        """Load FAISS index from disk."""
        # Try different possible index file names
        possible_names = ["chunks_index.faiss", "chunks_ivfpq.index"]
        
        for name in possible_names:
            index_path = self.index_dir / name
            if index_path.exists():
                try:
                    index = faiss.read_index(str(index_path))
                    logger.info(f"Loaded FAISS index from {index_path} with {index.ntotal} vectors")
                    return index
                except Exception as e:
                    logger.error(f"Failed to load FAISS index from {index_path}: {e}")
        
        logger.error(f"No FAISS index found in {self.index_dir}")
        return None
    
    def search(self, query: str, top_k: int = 50, 
               candidate_ids: Optional[List[int]] = None) -> Tuple[List[int], List[float]]:
        """Perform dense vector search."""
        if self.index is None:
            return [], []
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        
        # Perform search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert to lists
        hit_indices = indices[0].tolist()
        hit_distances = distances[0].tolist()
        
        # Filter out invalid indices
        valid_results = [(idx, dist) for idx, dist in zip(hit_indices, hit_distances) 
                        if idx >= 0 and idx < self.index.ntotal]
        
        if not valid_results:
            return [], []
        
        indices, distances = zip(*valid_results)
        return list(indices), list(distances)


class CrossEncoderReranker:
    """Cross-encoder based reranking for improved relevance."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model_name = self.config.get('CROSS_ENCODER', 'cross-encoder/ms-marco-TinyBERT-L-2-v2')
        self.cross_encoder: Optional[CrossEncoder] = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of cross-encoder."""
        if self._initialized or not CROSS_ENCODER_AVAILABLE:
            return
        
        try:
            self.cross_encoder = CrossEncoder(self.model_name)
            self._initialized = True
            logger.info(f"Cross-encoder initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
    
    def rerank(self, query: str, results: List[SearchResult], 
               top_k: Optional[int] = None) -> List[SearchResult]:
        """Rerank search results using cross-encoder."""
        if not CROSS_ENCODER_AVAILABLE or not results:
            return results
        
        self._initialize()
        if not self._initialized or self.cross_encoder is None:
            return results
        
        # Prepare input pairs for cross-encoder
        pairs = []
        for result in results:
            # Truncate text to avoid token limits
            text = result.text[:512] if len(result.text) > 512 else result.text
            pairs.append([query, text])
        
        try:
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Update results with cross-encoder scores
            for result, score in zip(results, scores):
                result.cross_score = float(score)
                result.final_score = float(score)
            
            # Sort by cross-encoder score
            results.sort(key=lambda r: r.cross_score or r.score, reverse=True)
            
            # Return top_k results if specified
            if top_k:
                results = results[:top_k]
            
            logger.debug(f"Reranked {len(results)} results using cross-encoder")
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
        
        return results


class SemanticRetriever:
    """Main retriever class with hybrid search capabilities and conversation support."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        
        # Initialize components
        db_path = self.config.get('SQLITE_DB', 'index.db')
        index_dir = Path(self.config.get('INDEX_DIR', './indices'))
        
        self.db_retriever = DatabaseRetriever(db_path)
        self.bm25_retriever = BM25Retriever(self.db_retriever)
        self.dense_retriever = DenseRetriever(self.config, index_dir)
        self.reranker = CrossEncoderReranker(self.config)
        
        # Configuration
        self.bm25_top_k = self.config.get('BM25_TOPK', 200)
        self.dense_top_k = self.config.get('DENSE_TOPK', 50)
        self.rerank_top_k = self.config.get('RERANK_TOPK', 5)
        
        # Conversation settings
        self.doc_search_k = self.config.get('DOC_SEARCH_K', 8)
        self.memory_search_k = self.config.get('MEMORY_SEARCH_K', 6)
        self.confidence_threshold = self.config.get('CONFIDENCE_THRESHOLD', 0.7)
        
        logger.info("SemanticRetriever initialized")
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Perform hybrid search with optional reranking - using original algorithm."""
        if top_k is None:
            top_k = self.rerank_top_k
        
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        
        # Step 1: BM25 prefiltering (optional) - like original
        pre_ids = None
        if BM25_AVAILABLE:
            try:
                pre_ids = self.bm25_retriever.search(query, self.bm25_top_k)
                logger.debug(f"BM25 prefilter returned {len(pre_ids)} candidates")
            except Exception:
                logger.info("BM25 prefilter failed; skipping")
        
        # Step 2: Dense retrieval - exactly like original
        query_vector = self.dense_retriever.embedder.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.dense_retriever.index.search(query_vector, self.dense_top_k)
        
        hits = list(indices[0])
        dists = list(distances[0])
        
        if not hits:
            logger.warning("No results found from dense retrieval")
            return []
        
        logger.debug(f"Dense retrieval returned {len(hits)} results")
        
        # Step 3: Map hit indices to database records - exactly like original
        cursor = self.db_retriever.conn.cursor()
        results = []
        
        for emb_id, dist in zip(hits, dists):
            # Original mapping: emb_id is the FAISS index position
            row = cursor.execute("""
                SELECT id, file_id, chunk_text, summary, chunk_type 
                FROM chunks WHERE emb_id=?
            """, (int(emb_id),)).fetchone()
            
            if row:
                chunk_id, file_id, text, summary, chunk_type = row
                file_row = cursor.execute("SELECT path FROM files WHERE id=?", (file_id,)).fetchone()
                path = file_row[0] if file_row else ""
                
                results.append({
                    "chunk_id": chunk_id,
                    "file_id": file_id, 
                    "path": path,
                    "text": text,
                    "score": float(dist),
                    "summary": summary,
                    "chunk_type": chunk_type
                })
        
        # Step 4: Re-rank with cross-encoder if available - exactly like original
        if CROSS_ENCODER_AVAILABLE and self.reranker.cross_encoder and results:
            try:
                pairs = [[query, r["text"][:512]] for r in results[:self.rerank_top_k*2]]
                scores = self.reranker.cross_encoder.predict(pairs)
                
                for i, s in enumerate(scores):
                    if i < len(results):
                        results[i]["cross_score"] = float(s)
                
                # Sort by cross-encoder score (higher = better)
                results.sort(key=lambda r: r.get("cross_score", r["score"]), reverse=True)
            except Exception as e:
                logger.error(f"Cross-encoder reranking failed: {e}")
                # Fallback: sort by distance (lower = better) - like original
                results = sorted(results, key=lambda r: r["score"])
        else:
            # Sort by distance (lower = better) - like original
            results = sorted(results, key=lambda r: r["score"])
        
        # Step 5: Convert to SearchResult objects and return top results
        search_results = []
        for r in results[:top_k]:
            result = SearchResult(
                chunk_id=r["chunk_id"],
                file_id=r["file_id"],
                path=r["path"],
                text=r["text"],
                summary=r["summary"],
                chunk_type=r.get("chunk_type", "content"),
                score=r["score"],
                method='dense'
            )
            if "cross_score" in r:
                result.cross_score = r["cross_score"]
                result.final_score = r["cross_score"]
            search_results.append(result)
        
        # Log search
        try:
            self.db_retriever.log_search(query, len(search_results))
        except Exception:
            pass
        
        logger.info(f"Search completed: {len(search_results)} results returned")
        return search_results
    
    def get_similar_chunks(self, chunk_id: int, top_k: int = 5) -> List[SearchResult]:
        """Find chunks similar to a given chunk."""
        cursor = self.db_retriever.conn.cursor()
        cursor.execute("SELECT chunk_text FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        
        if not row:
            return []
        
        chunk_text = row['chunk_text']
        return self.search(chunk_text, top_k)
    
    def get_context_chunks(self, chunk_id: int, context_size: int = 2) -> List[SearchResult]:
        """Get surrounding chunks for context."""
        cursor = self.db_retriever.conn.cursor()
        
        # Get the chunk and its file_id
        cursor.execute("SELECT file_id FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        if not row:
            return []
        
        file_id = row['file_id']
        
        # Get chunks from the same file around the target chunk
        cursor.execute("""
            SELECT c.id, c.file_id, c.chunk_text, c.summary, c.chunk_type,
                   f.path
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE c.file_id = ?
            ORDER BY c.id
        """, (file_id,))
        
        chunks = cursor.fetchall()
        
        # Find the target chunk index
        target_idx = None
        for i, chunk in enumerate(chunks):
            if chunk['id'] == chunk_id:
                target_idx = i
                break
        
        if target_idx is None:
            return []
        
        # Get context chunks
        start_idx = max(0, target_idx - context_size)
        end_idx = min(len(chunks), target_idx + context_size + 1)
        
        results = []
        for i in range(start_idx, end_idx):
            chunk = chunks[i]
            result = SearchResult(
                chunk_id=chunk['id'],
                file_id=chunk['file_id'],
                path=chunk['path'],
                text=chunk['chunk_text'],
                summary=chunk['summary'],
                chunk_type=chunk['chunk_type'],
                score=1.0,  # Context chunks have uniform score
                method='context'
            )
            results.append(result)
        
        return results
    
    def search_docs(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Search documents only (type='doc' or legacy content types)."""
        if top_k is None:
            top_k = self.doc_search_k
        
        # Use the existing search but filter for document types
        all_results = self.search(query, top_k * 2)  # Get more to filter
        
        # Filter for document chunks
        doc_results = []
        for result in all_results:
            # Include chunks that are documents (not conversation messages)
            if (hasattr(result, 'chunk_type') and 
                result.chunk_type in ['doc', 'content', 'metadata']) or \
               (isinstance(result, dict) and 
                result.get('chunk_type') in ['doc', 'content', 'metadata']):
                doc_results.append(result)
                if len(doc_results) >= top_k:
                    break
        
        return doc_results
    
    def search_memory(self, query: str, conversation_id: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Search conversation memory only."""
        if top_k is None:
            top_k = self.memory_search_k
        
        if not conversation_id:
            return []
        
        # Search for conversation messages
        cursor = self.db_retriever.conn.cursor()
        cursor.execute("""
            SELECT c.id, c.file_id, c.chunk_text, c.summary, c.chunk_type,
                   c.conversation_id, c.timestamp, c.confidence, c.citations,
                   COALESCE(f.path, 'conversation') as path
            FROM chunks c
            LEFT JOIN files f ON c.file_id = f.id
            WHERE c.conversation_id = ? 
            AND c.chunk_type IN ('assistant_msg', 'user_msg', 'summary')
            AND c.archived = FALSE
            AND (c.chunk_type != 'assistant_msg' OR c.confidence >= ? OR c.citations IS NOT NULL)
            ORDER BY c.timestamp DESC
            LIMIT ?
        """, (conversation_id, self.confidence_threshold, top_k))
        
        results = []
        for row in cursor.fetchall():
            # Parse citations
            citations = []
            if row[8]:  # citations column
                try:
                    citations = json.loads(row[8])
                except:
                    citations = []
            
            result = SearchResult(
                chunk_id=row[0],
                file_id=row[1] or -1,
                path=row[9],
                text=row[2],
                summary=row[3],
                chunk_type=row[4],
                score=1.0,  # Memory items have uniform score for now
                method='memory'
            )
            results.append(result)
        
        return results
    
    def search_dual(self, query: str, conversation_id: Optional[str] = None, 
                   doc_k: Optional[int] = None, memory_k: Optional[int] = None) -> Tuple[List[SearchResult], List[SearchResult]]:
        """
        Dual retrieval: Search both documents and conversation memory.
        This is the key method for conversational RAG.
        """
        doc_k = doc_k or self.doc_search_k
        memory_k = memory_k or self.memory_search_k
        
        # Search documents (all indexed documents from DATA_PATH)
        doc_results = self.search_docs(query, doc_k)
        
        # Search conversation memory if conversation_id provided
        memory_results = []
        if conversation_id:
            memory_results = self.search_memory(query, conversation_id, memory_k)
        
        logger.info(f"Dual search: {len(doc_results)} docs + {len(memory_results)} memory items")
        return doc_results, memory_results
    
    def close(self):
        """Clean up resources."""
        self.db_retriever.close()


# Legacy compatibility class - exactly like original
class Retriever:
    """Legacy retriever class for backward compatibility - matches original exactly."""
    
    def __init__(self):
        from utils import ConfigManager
        
        config = ConfigManager()
        db_path = config.get("SQLITE_DB", "index.db")
        index_dir = config.get("INDEX_DIR", "./indices")
        embed_model = config.get("EMBED_MODEL", "all-MiniLM-L6-v2")
        cross_model = config.get("CROSS_ENCODER", "cross-encoder/ms-marco-TinyBERT-L-2-v2")
        
        self.conn = sqlite3.connect(db_path)
        self.embedder = SentenceTransformer(embed_model)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        
        # Load FAISS - try different possible names
        idx_path = f"{index_dir}/chunks_ivfpq.index"
        if not Path(idx_path).exists():
            idx_path = f"{index_dir}/chunks_index.faiss"
        
        if not Path(idx_path).exists():
            raise FileNotFoundError(f"FAISS index not found at {idx_path}")
        
        self.index = faiss.read_index(idx_path)
        
        # Load cross encoder lazily
        try:
            self.cross = CrossEncoder(cross_model)
        except Exception:
            self.cross = None
    
    def _load_all_chunk_texts(self):
        """Load all chunk texts - like original."""
        c = self.conn.cursor()
        rows = c.execute("SELECT id, chunk_text FROM chunks").fetchall()
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        return ids, texts
    
    def bm25_prefilter(self, query, topk=200):
        """BM25 prefiltering - like original."""
        if not BM25_AVAILABLE:
            return []
        
        ids, texts = self._load_all_chunk_texts()
        tokenized = [t.split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:topk]
        pre_ids = [ids[i] for i in top_idx if scores[i] > 0]
        return pre_ids
    
    def dense_search(self, query, candidate_ids=None, topk=50):
        """Dense search - like original."""
        qv = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(qv, topk)
        hits = list(I[0])
        return hits, list(D[0])
    
    def search(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        """Search method - exactly like original."""
        # Two-pass: BM25 -> Dense -> Rerank
        pre_ids = None
        try:
            pre_ids = self.bm25_prefilter(query)
        except Exception:
            logger.info("BM25 prefilter failed; skipping")
        
        hits, dists = self.dense_search(query, candidate_ids=pre_ids, topk=50)
        
        # Map hit ids to chunk ids stored in SQLite
        c = self.conn.cursor()
        results = []
        for emb_id, dist in zip(hits, dists):
            row = c.execute("SELECT id, file_id, chunk_text, summary FROM chunks WHERE emb_id=?", (int(emb_id),)).fetchone()
            if row:
                chunk_id, file_id, text, summary = row
                file_row = c.execute("SELECT path FROM files WHERE id=?", (file_id,)).fetchone()
                path = file_row[0] if file_row else ""
                results.append({
                    "chunk_id": chunk_id, 
                    "file_id": file_id, 
                    "path": path, 
                    "text": text, 
                    "score": float(dist), 
                    "summary": summary
                })
        
        # Re-rank with cross-encoder if available
        if self.cross and results:
            pairs = [[query, r["text"][:512]] for r in results[:topk*2]]
            scores = self.cross.predict(pairs)
            for i, s in enumerate(scores):
                results[i]["cross_score"] = float(s)
            results.sort(key=lambda r: r.get("cross_score", r["score"]), reverse=True)
        else:
            results = sorted(results, key=lambda r: r["score"])
        
        return results[:topk]