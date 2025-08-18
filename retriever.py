"""
Advanced retrieval system with hybrid search and reranking capabilities.
"""

import sqlite3
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
        """Perform BM25 search and return chunk embedding IDs."""
        if not BM25_AVAILABLE:
            return []
        
        self._initialize()
        if not self._initialized or self.bm25 is None:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Filter out zero scores
        filtered_indices = [idx for idx in top_indices if scores[idx] > 0]
        
        # Return corresponding chunk IDs
        return [self.chunk_ids[idx] for idx in filtered_indices]


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
    """Main retriever class with hybrid search capabilities."""
    
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
        
        logger.info("SemanticRetriever initialized")
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Perform hybrid search with optional reranking."""
        if top_k is None:
            top_k = self.rerank_top_k
        
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        
        # Step 1: BM25 prefiltering (optional)
        candidate_ids = None
        if BM25_AVAILABLE:
            try:
                candidate_ids = self.bm25_retriever.search(query, self.bm25_top_k)
                logger.debug(f"BM25 prefilter returned {len(candidate_ids)} candidates")
            except Exception as e:
                logger.warning(f"BM25 prefiltering failed: {e}")
        
        # Step 2: Dense retrieval
        hit_indices, distances = self.dense_retriever.search(
            query, self.dense_top_k, candidate_ids
        )
        
        if not hit_indices:
            logger.warning("No results found from dense retrieval")
            return []
        
        logger.debug(f"Dense retrieval returned {len(hit_indices)} results")
        
        # Step 3: Convert to SearchResult objects
        chunk_data = self.db_retriever.get_chunks_by_emb_ids(hit_indices)
        chunk_lookup = {chunk['emb_id']: chunk for chunk in chunk_data}
        
        results = []
        for idx, distance in zip(hit_indices, distances):
            chunk = chunk_lookup.get(idx)
            if chunk:
                # Convert distance to similarity score (lower distance = higher similarity)
                score = 1.0 / (1.0 + distance) if distance >= 0 else 0.0
                
                result = SearchResult(
                    chunk_id=chunk['id'],
                    file_id=chunk['file_id'],
                    path=chunk['path'],
                    text=chunk['chunk_text'],
                    summary=chunk['summary'],
                    chunk_type=chunk['chunk_type'],
                    score=score,
                    method='dense'
                )
                results.append(result)
        
        # Step 4: Reranking (optional)
        if CROSS_ENCODER_AVAILABLE and len(results) > 1:
            results = self.reranker.rerank(query, results, top_k * 2)  # Get more for reranking
        
        # Step 5: Final filtering and sorting
        results = results[:top_k]
        
        # Log search
        self.db_retriever.log_search(query, len(results))
        
        logger.info(f"Search completed: {len(results)} results returned")
        return results
    
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
    
    def close(self):
        """Clean up resources."""
        self.db_retriever.close()


# Legacy compatibility class
class Retriever(SemanticRetriever):
    """Legacy retriever class for backward compatibility."""
    
    def __init__(self):
        super().__init__()
    
    def search(self, query: str, topk: Optional[int] = None) -> List[Dict[str, Any]]:
        """Legacy search method returning dictionaries."""
        if topk is None:
            topk = self.rerank_top_k
        
        results = super().search(query, topk)
        return [result.to_dict() for result in results]