"""
Conversational Memory System for Industry-Grade RAG
Implements episodic and semantic memory with proper threading and metadata.
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import ConfigManager
from chunker import SemanticChunker

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""
    id: str
    type: str  # 'user_msg', 'assistant_msg', 'summary', 'doc'
    conversation_id: str
    prev_id: Optional[str]
    topic: Optional[str]
    timestamp: str
    source: Optional[str]
    confidence: float
    citations: List[str]
    content: str
    token_count: int = 0
    archived: bool = False
    embedding_id: Optional[int] = None


class ConversationMemoryDB:
    """Enhanced database for conversational RAG with memory."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.db_path = config.get('SQLITE_DB', 'index.db')
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Initialize embedder for memory
        embed_model = config.get('EMBED_MODEL', 'all-MiniLM-L6-v2')
        self.embedder = SentenceTransformer(embed_model)
        
        # Token counting
        self.chunker = SemanticChunker(
            chunk_size=config.get('CHUNK_SIZE', 500),
            overlap=config.get('CHUNK_OVERLAP', 100)
        )
        
        self._ensure_conversation_tables()
    
    def _ensure_conversation_tables(self):
        """Create conversation memory tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Enhanced messages table with full metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,  -- 'user_msg', 'assistant_msg', 'summary', 'doc'
                conversation_id TEXT NOT NULL,
                prev_id TEXT,
                topic TEXT,
                timestamp TEXT NOT NULL,
                source TEXT,
                confidence REAL DEFAULT 0.0,
                citations TEXT,  -- JSON array of citation IDs
                content TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                archived BOOLEAN DEFAULT FALSE,
                embedding_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prev_id) REFERENCES conversation_messages (id),
                FOREIGN KEY (embedding_id) REFERENCES message_embeddings (id)
            )
        """)
        
        # Separate embeddings table for memory items
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (message_id) REFERENCES conversation_messages (id)
            )
        """)
        
        # Conversation metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                last_summary_at TIMESTAMP,
                archived BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Enhanced search history with conversation context
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                query TEXT NOT NULL,
                query_type TEXT,  -- 'doc_search', 'memory_search', 'hybrid'
                results_count INTEGER DEFAULT 0,
                doc_results_count INTEGER DEFAULT 0,
                memory_results_count INTEGER DEFAULT 0,
                latency_ms REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_msg_conv_id ON conversation_messages(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_msg_type ON conversation_messages(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_msg_timestamp ON conversation_messages(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_msg_prev_id ON conversation_messages(prev_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_msg_archived ON conversation_messages(archived)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_embeddings_msg_id ON message_embeddings(message_id)")
        
        self.conn.commit()
        logger.info("Conversation memory database schema initialized")
    
    def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation and return its ID."""
        conversation_id = str(uuid.uuid4())
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (
            conversation_id,
            title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat()
        ))
        
        self.conn.commit()
        logger.info(f"Created new conversation: {conversation_id}")
        return conversation_id
    
    def add_message(self, 
                   message_type: str,
                   conversation_id: str,
                   content: str,
                   prev_id: Optional[str] = None,
                   topic: Optional[str] = None,
                   source: Optional[str] = None,
                   confidence: float = 0.0,
                   citations: Optional[List[str]] = None) -> ConversationMessage:
        """Add a message to the conversation and generate embedding."""
        
        message_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        citations = citations or []
        
        # Count tokens
        token_count = self.chunker.token_counter.count_tokens(content)
        
        # Create message object
        message = ConversationMessage(
            id=message_id,
            type=message_type,
            conversation_id=conversation_id,
            prev_id=prev_id,
            topic=topic,
            timestamp=timestamp,
            source=source,
            confidence=confidence,
            citations=citations,
            content=content,
            token_count=token_count
        )
        
        # Generate and store embedding
        embedding = self.embedder.encode([content], convert_to_numpy=True)[0]
        embedding_blob = embedding.astype(np.float32).tobytes()
        
        cursor = self.conn.cursor()
        
        # Insert embedding
        cursor.execute("""
            INSERT INTO message_embeddings (message_id, embedding)
            VALUES (?, ?)
        """, (message_id, embedding_blob))
        
        embedding_id = cursor.lastrowid
        message.embedding_id = embedding_id
        
        # Insert message
        cursor.execute("""
            INSERT INTO conversation_messages 
            (id, type, conversation_id, prev_id, topic, timestamp, source, 
             confidence, citations, content, token_count, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message_id, message_type, conversation_id, prev_id, topic, 
            timestamp, source, confidence, json.dumps(citations), 
            content, token_count, embedding_id
        ))
        
        # Update conversation metadata
        cursor.execute("""
            UPDATE conversations 
            SET updated_at = ?, message_count = message_count + 1
            WHERE id = ?
        """, (timestamp, conversation_id))
        
        self.conn.commit()
        
        logger.info(f"Added {message_type} message to conversation {conversation_id}")
        return message
    
    def get_conversation_messages(self, 
                                conversation_id: str,
                                message_types: Optional[List[str]] = None,
                                limit: Optional[int] = None,
                                include_archived: bool = False) -> List[ConversationMessage]:
        """Retrieve messages from a conversation."""
        
        cursor = self.conn.cursor()
        
        # Build query
        where_conditions = ["conversation_id = ?"]
        params = [conversation_id]
        
        if message_types:
            placeholders = ','.join('?' * len(message_types))
            where_conditions.append(f"type IN ({placeholders})")
            params.extend(message_types)
        
        if not include_archived:
            where_conditions.append("archived = FALSE")
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
            SELECT * FROM conversation_messages 
            WHERE {where_clause}
            ORDER BY timestamp ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        messages = []
        for row in rows:
            citations = json.loads(row['citations']) if row['citations'] else []
            message = ConversationMessage(
                id=row['id'],
                type=row['type'],
                conversation_id=row['conversation_id'],
                prev_id=row['prev_id'],
                topic=row['topic'],
                timestamp=row['timestamp'],
                source=row['source'],
                confidence=row['confidence'],
                citations=citations,
                content=row['content'],
                token_count=row['token_count'],
                archived=row['archived'],
                embedding_id=row['embedding_id']
            )
            messages.append(message)
        
        return messages
    
    def search_memory(self, 
                     query: str,
                     conversation_id: str,
                     message_types: Optional[List[str]] = None,
                     top_k: int = 10,
                     confidence_threshold: float = 0.0) -> List[Tuple[ConversationMessage, float]]:
        """Search conversation memory using semantic similarity."""
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        
        # Get candidate messages
        messages = self.get_conversation_messages(
            conversation_id, 
            message_types, 
            include_archived=False
        )
        
        if not messages:
            return []
        
        # Filter by confidence for assistant messages
        if message_types is None or 'assistant_msg' in message_types:
            filtered_messages = []
            for msg in messages:
                if msg.type == 'assistant_msg':
                    # Apply confidence gating for assistant messages
                    if msg.confidence >= confidence_threshold or msg.citations:
                        filtered_messages.append(msg)
                else:
                    filtered_messages.append(msg)
            messages = filtered_messages
        
        if not messages:
            return []
        
        # Get embeddings for all messages
        cursor = self.conn.cursor()
        embedding_ids = [msg.embedding_id for msg in messages if msg.embedding_id]
        
        if not embedding_ids:
            return []
        
        placeholders = ','.join('?' * len(embedding_ids))
        cursor.execute(f"""
            SELECT message_id, embedding 
            FROM message_embeddings 
            WHERE id IN ({placeholders})
        """, embedding_ids)
        
        embedding_data = {row['message_id']: np.frombuffer(row['embedding'], dtype=np.float32) 
                         for row in cursor.fetchall()}
        
        # Calculate similarities
        results = []
        for msg in messages:
            if msg.id in embedding_data:
                msg_embedding = embedding_data[msg.id]
                similarity = np.dot(query_embedding, msg_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(msg_embedding)
                )
                results.append((msg, float(similarity)))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_sliding_window(self, 
                          conversation_id: str, 
                          window_size: int = 6) -> List[ConversationMessage]:
        """Get the most recent N messages for sliding window context."""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM conversation_messages 
            WHERE conversation_id = ? AND type IN ('user_msg', 'assistant_msg')
            AND archived = FALSE
            ORDER BY timestamp DESC
            LIMIT ?
        """, (conversation_id, window_size))
        
        rows = cursor.fetchall()
        
        messages = []
        for row in reversed(rows):  # Reverse to get chronological order
            citations = json.loads(row['citations']) if row['citations'] else []
            message = ConversationMessage(
                id=row['id'],
                type=row['type'],
                conversation_id=row['conversation_id'],
                prev_id=row['prev_id'],
                topic=row['topic'],
                timestamp=row['timestamp'],
                source=row['source'],
                confidence=row['confidence'],
                citations=citations,
                content=row['content'],
                token_count=row['token_count'],
                archived=row['archived'],
                embedding_id=row['embedding_id']
            )
            messages.append(message)
        
        return messages
    
    def should_summarize(self, conversation_id: str, turn_threshold: int = 10) -> bool:
        """Check if conversation should be summarized."""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT message_count, last_summary_at 
            FROM conversations 
            WHERE id = ?
        """, (conversation_id,))
        
        row = cursor.fetchone()
        if not row:
            return False
        
        message_count = row['message_count']
        last_summary_at = row['last_summary_at']
        
        # Count messages since last summary
        if last_summary_at:
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM conversation_messages 
                WHERE conversation_id = ? AND timestamp > ? AND archived = FALSE
            """, (conversation_id, last_summary_at))
            
            messages_since_summary = cursor.fetchone()['count']
        else:
            messages_since_summary = message_count
        
        return messages_since_summary >= turn_threshold
    
    def archive_messages(self, message_ids: List[str]):
        """Mark messages as archived."""
        
        if not message_ids:
            return
        
        cursor = self.conn.cursor()
        placeholders = ','.join('?' * len(message_ids))
        cursor.execute(f"""
            UPDATE conversation_messages 
            SET archived = TRUE 
            WHERE id IN ({placeholders})
        """, message_ids)
        
        self.conn.commit()
        logger.info(f"Archived {len(message_ids)} messages")
    
    def update_conversation_summary_timestamp(self, conversation_id: str):
        """Update the last summary timestamp for a conversation."""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE conversations 
            SET last_summary_at = ? 
            WHERE id = ?
        """, (datetime.now(timezone.utc).isoformat(), conversation_id))
        
        self.conn.commit()
    
    def get_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of conversations."""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, title, created_at, updated_at, message_count, archived
            FROM conversations 
            WHERE archived = FALSE
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def log_search(self, 
                  conversation_id: Optional[str],
                  query: str,
                  query_type: str,
                  results_count: int,
                  doc_results_count: int = 0,
                  memory_results_count: int = 0,
                  latency_ms: float = 0.0):
        """Log search operation with detailed metrics."""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO enhanced_search_history 
            (conversation_id, query, query_type, results_count, 
             doc_results_count, memory_results_count, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            conversation_id, query, query_type, results_count,
            doc_results_count, memory_results_count, latency_ms
        ))
        
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        self.conn.close()