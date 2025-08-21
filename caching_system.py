"""
Caching system for embeddings and responses to avoid recomputation.
"""

import sqlite3
import hashlib
import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

from utils import ConfigManager

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.db_path = config.get('SQLITE_DB', 'index.db')
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._ensure_cache_tables()
    
    def _ensure_cache_tables(self):
        """Create cache tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Embedding cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE NOT NULL,
                model_name TEXT NOT NULL,
                content_preview TEXT,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Response cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                response TEXT NOT NULL,
                confidence REAL,
                citations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(query_hash, context_hash, model_name)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embedding_cache_hash ON embedding_cache(content_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embedding_cache_model ON embedding_cache(model_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_response_cache_query ON response_cache(query_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_response_cache_model ON response_cache(model_name)")
        
        self.conn.commit()
        logger.info("Cache tables initialized")
    
    def get_embedding(self, content: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding if available."""
        content_hash = self._hash_content(content)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT embedding FROM embedding_cache 
            WHERE content_hash = ? AND model_name = ?
        """, (content_hash, model_name))
        
        result = cursor.fetchone()
        if result:
            try:
                embedding = pickle.loads(result[0])
                logger.debug(f"Cache hit for embedding: {content_hash[:8]}")
                return embedding
            except Exception as e:
                logger.warning(f"Failed to deserialize cached embedding: {e}")
        
        return None
    
    def store_embedding(self, content: str, model_name: str, embedding: np.ndarray):
        """Store embedding in cache."""
        content_hash = self._hash_content(content)
        content_preview = content[:100] + "..." if len(content) > 100 else content
        
        try:
            embedding_blob = pickle.dumps(embedding)
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO embedding_cache 
                (content_hash, model_name, content_preview, embedding)
                VALUES (?, ?, ?, ?)
            """, (content_hash, model_name, content_preview, embedding_blob))
            
            self.conn.commit()
            logger.debug(f"Cached embedding: {content_hash[:8]}")
            
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def get_response(self, query: str, context: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        query_hash = self._hash_content(query)
        context_hash = self._hash_content(context)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT response, confidence, citations FROM response_cache 
            WHERE query_hash = ? AND context_hash = ? AND model_name = ?
        """, (query_hash, context_hash, model_name))
        
        result = cursor.fetchone()
        if result:
            try:
                citations = json.loads(result[2]) if result[2] else []
                logger.debug(f"Cache hit for response: {query_hash[:8]}")
                return {
                    'response': result[0],
                    'confidence': result[1],
                    'citations': citations,
                    'cached': True
                }
            except Exception as e:
                logger.warning(f"Failed to deserialize cached response: {e}")
        
        return None
    
    def store_response(self, query: str, context: str, model_name: str, 
                      response: str, confidence: float, citations: List[str]):
        """Store response in cache."""
        query_hash = self._hash_content(query)
        context_hash = self._hash_content(context)
        
        try:
            citations_json = json.dumps(citations) if citations else None
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO response_cache 
                (query_hash, context_hash, model_name, response, confidence, citations)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (query_hash, context_hash, model_name, response, confidence, citations_json))
            
            self.conn.commit()
            logger.debug(f"Cached response: {query_hash[:8]}")
            
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    def _hash_content(self, content: str) -> str:
        """Create hash of content for caching."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def clear_cache(self, older_than_days: int = 30):
        """Clear old cache entries."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            DELETE FROM embedding_cache 
            WHERE created_at < datetime('now', '-{} days')
        """.format(older_than_days))
        
        cursor.execute("""
            DELETE FROM response_cache 
            WHERE created_at < datetime('now', '-{} days')
        """.format(older_than_days))
        
        self.conn.commit()
        logger.info(f"Cleared cache entries older than {older_than_days} days")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM embedding_cache")
        embedding_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM response_cache")
        response_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT 
                SUM(LENGTH(embedding)) / (1024.0 * 1024.0) as embedding_mb,
                SUM(LENGTH(response)) / (1024.0 * 1024.0) as response_mb
            FROM embedding_cache ec
            LEFT JOIN response_cache rc ON 1=1
        """)
        
        size_result = cursor.fetchone()
        embedding_mb = size_result[0] or 0
        response_mb = size_result[1] or 0
        
        return {
            'embedding_count': embedding_count,
            'response_count': response_count,
            'embedding_size_mb': round(embedding_mb, 2),
            'response_size_mb': round(response_mb, 2),
            'total_size_mb': round(embedding_mb + response_mb, 2)
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()