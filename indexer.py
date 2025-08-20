"""
Multi-modal semantic indexer with advanced chunking and embedding generation.
"""

import os
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from dataclasses import dataclass
import faiss
from sentence_transformers import SentenceTransformer

from utils import ConfigManager, ensure_dir, file_sha256
from chunker import SemanticChunker, TextChunk
from extractors import  TextExtractor, ImageExtractor, VideoExtractor
from extractors.base import MultiExtractor

logger = logging.getLogger(__name__)


@dataclass
class FileRecord:
    """Represents a file record in the database."""
    id: Optional[int]
    path: str
    filename: str
    file_extension: str
    file_type: str
    sha256: str
    size: int
    processed: bool = False


@dataclass
class ChunkRecord:
    """Represents a chunk record in the database."""
    id: Optional[int]
    file_id: int
    chunk_text: str
    summary: str
    chunk_type: str
    token_count: int
    prev_id: Optional[int] = None
    next_id: Optional[int] = None
    emb_id: Optional[int] = None


class DatabaseManager:
    """Manages SQLite database operations for the indexer."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_extension TEXT,
                file_type TEXT,
                sha256 TEXT,
                size INTEGER,
                processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                summary TEXT,
                chunk_type TEXT DEFAULT 'content',
                token_count INTEGER DEFAULT 0,
                prev_id INTEGER,
                next_id INTEGER,
                emb_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES files (id) ON DELETE CASCADE
            )
        """)
        
        # Search history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                results_count INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_type ON files(file_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_emb_id ON chunks(emb_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type)")
        
        self.conn.commit()
    
    def insert_file(self, file_record: FileRecord) -> int:
        """Insert or update file record and return the file ID."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO files 
            (path, filename, file_extension, file_type, sha256, size, processed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            file_record.path,
            file_record.filename,
            file_record.file_extension,
            file_record.file_type,
            file_record.sha256,
            file_record.size,
            file_record.processed
        ))
        
        self.conn.commit()
        
        # Get the file ID
        cursor.execute("SELECT id FROM files WHERE path = ?", (file_record.path,))
        result = cursor.fetchone()
        return result['id'] if result else cursor.lastrowid
    
    def insert_chunks(self, chunks: List[ChunkRecord]) -> List[int]:
        """Insert chunk records and return their IDs."""
        cursor = self.conn.cursor()
        chunk_ids = []
        
        for chunk in chunks:
            cursor.execute("""
                INSERT INTO chunks 
                (file_id, chunk_text, summary, chunk_type, token_count, prev_id, emb_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.file_id,
                chunk.chunk_text,
                chunk.summary,
                chunk.chunk_type,
                chunk.token_count,
                chunk.prev_id,
                chunk.emb_id
            ))
            chunk_ids.append(cursor.lastrowid)
        
        self.conn.commit()
        return chunk_ids
    
    def update_chunk_links(self, chunk_ids: List[int]):
        """Update prev_id and next_id links for chunks."""
        cursor = self.conn.cursor()
        
        for i, chunk_id in enumerate(chunk_ids):
            prev_id = chunk_ids[i - 1] if i > 0 else None
            next_id = chunk_ids[i + 1] if i < len(chunk_ids) - 1 else None
            
            cursor.execute("""
                UPDATE chunks SET prev_id = ?, next_id = ? WHERE id = ?
            """, (prev_id, next_id, chunk_id))
        
        self.conn.commit()
    
    def get_all_chunk_texts(self) -> List[Tuple[int, str]]:
        """Get all chunk texts with their embedding IDs."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT emb_id, chunk_text FROM chunks WHERE emb_id IS NOT NULL ORDER BY emb_id")
        return [(row['emb_id'], row['chunk_text']) for row in cursor.fetchall()]
    
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
        if row:
            return dict(row)
        return None
    
    def clear_all_data(self):
        """Clear all data from the database."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chunks")
        cursor.execute("DELETE FROM files")
        cursor.execute("DELETE FROM search_history")
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        self.conn.close()


class EmbeddingManager:
    """Manages embedding generation and FAISS index operations."""
    
    def __init__(self, config: ConfigManager, index_dir: Path):
        self.config = config
        self.index_dir = ensure_dir(index_dir)
        
        # Initialize sentence transformer
        model_name = self.config.get('EMBED_MODEL', 'all-MiniLM-L6-v2')
        self.embedder = SentenceTransformer(model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        logger.info(f"Initialized embedder: {model_name} (dim={self.embedding_dim})")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.astype(np.float32)
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index from embeddings."""
        num_vectors = embeddings.shape[0]
        
        if num_vectors == 0:
            logger.warning("No embeddings to index")
            return faiss.IndexFlatL2(self.embedding_dim)
        
        logger.info(f"Building FAISS index for {num_vectors} vectors...")
        
        # Choose index type based on dataset size
        if num_vectors < 100:
            # Small dataset - use flat index
            logger.info("Using IndexFlatL2 for small dataset")
            index = faiss.IndexFlatL2(self.embedding_dim)
            index.add(embeddings)
        else:
            # Larger dataset - use IVF+PQ
            nlist = max(8, min(
                int(np.sqrt(num_vectors)),
                self.config.get('FAISS_NLIST', 100),
                num_vectors // 4
            ))
            
            # Choose PQ parameters
            m = self.config.get('FAISS_PQ_M', 8)
            while m > 1 and self.embedding_dim % m != 0:
                m -= 1
            if m <= 1:
                m = max(2, self.embedding_dim // 8)
            
            logger.info(f"Using IndexIVFPQ with nlist={nlist}, m={m}")
            
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, 8)
            
            # Train the index
            logger.info("Training FAISS index...")
            index.train(embeddings)
            index.add(embeddings)
        
        return index
    
    def save_index(self, index: faiss.Index, filename: str = "chunks_index.faiss"):
        """Save FAISS index to disk."""
        index_path = self.index_dir / filename
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        return index_path
    
    def load_index(self, filename: str = "chunks_index.faiss") -> Optional[faiss.Index]:
        """Load FAISS index from disk."""
        index_path = self.index_dir / filename
        if not index_path.exists():
            return None
        
        try:
            index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index from {index_path}")
            return index
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return None


class SemanticIndexer:
    """Main indexer class for multi-modal semantic search."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        
        # Initialize components
        self.data_path = Path(self.config.get('DATA_PATH', './data'))
        self.index_dir = Path(self.config.get('INDEX_DIR', './indices'))
        
        ensure_dir(self.index_dir)
        
        # Initialize database
        db_path = self.config.get('SQLITE_DB', 'index.db')
        self.db_manager = DatabaseManager(db_path)
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(self.config, self.index_dir)
        
        # Initialize chunker
        self.chunker = SemanticChunker(
            chunk_size=self.config.get('CHUNK_SIZE', 500),
            overlap=self.config.get('CHUNK_OVERLAP', 100)
        )
        
        # Initialize extractors
        self.extractor = MultiExtractor([
            TextExtractor(self.config.config),
            ImageExtractor(self.config.config),
            VideoExtractor(self.config.config)
        ])
        
        logger.info(f"Initialized SemanticIndexer with data path: {self.data_path}")
    
    def get_file_type(self, path: Path) -> str:
        """Determine file type based on extension."""
        ext = path.suffix.lower()
        
        if ext in {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}:
            return 'video'
        elif ext in {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'}:
            return 'audio'
        elif ext in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif'}:
            return 'image'
        elif ext in {'.pdf', '.docx', '.doc', '.txt', '.rtf', '.odt'}:
            return 'document'
        elif ext in {'.xlsx', '.xls', '.csv', '.ods'}:
            return 'spreadsheet'
        elif ext in {'.pptx', '.ppt', '.odp'}:
            return 'presentation'
        else:
            return 'other'
    
    def create_metadata_chunk(self, file_path: Path, file_type: str) -> str:
        """Create metadata chunk for file discovery."""
        metadata_lines = [
            f"File: {file_path.name}",
            f"Type: {file_type}",
            f"Extension: {file_path.suffix}",
            f"Directory: {file_path.parent.name}",
            f"Full path: {file_path}",
            f"Summary: This is a {file_type} file named {file_path.name}"
        ]
        
        # Add type-specific keywords
        keywords = {
            'video': "video file, movie, clip, recording, media, visual content, footage",
            'image': "image file, picture, photo, graphic, visual, screenshot",
            'audio': "audio file, music, sound, recording, voice, song",
            'document': "document, text file, pdf, word document, report, paper",
            'spreadsheet': "spreadsheet, excel, data, table, numbers, calculations",
            'presentation': "presentation, slides, powerpoint, deck"
        }
        
        if file_type in keywords:
            metadata_lines.append(f"Keywords: {keywords[file_type]}")
        
        return "\n".join(metadata_lines)
    
    def process_file(self, file_path: Path) -> Tuple[int, List[ChunkRecord]]:
        """Process a single file and return file ID and chunks."""
        try:
            # Get file info
            file_type = self.get_file_type(file_path)
            file_stat = file_path.stat()
            file_hash = file_sha256(file_path)
            
            # Create file record
            file_record = FileRecord(
                id=None,
                path=str(file_path),
                filename=file_path.name,
                file_extension=file_path.suffix.lower(),
                file_type=file_type,
                sha256=file_hash,
                size=file_stat.st_size,
                processed=False
            )
            
            # Insert file record
            file_id = self.db_manager.insert_file(file_record)
            
            # Extract content
            extraction_result = self.extractor.extract(file_path)
            
            chunks = []
            
            # Always add metadata chunk
            metadata_text = self.create_metadata_chunk(file_path, file_type)
            metadata_chunk = ChunkRecord(
                id=None,
                file_id=file_id,
                chunk_text=metadata_text,
                summary=f"Metadata for {file_path.name}",
                chunk_type='metadata',
                token_count=len(metadata_text.split()),
                emb_id=None
            )
            chunks.append(metadata_chunk)
            
            # Add content chunks if extraction was successful
            if extraction_result.success and extraction_result.content.strip():
                text_chunks = self.chunker.chunk_text(
                    extraction_result.content,
                    source_info={'file_path': str(file_path), 'file_type': file_type}
                )
                
                for chunk in text_chunks:
                    summary = self._create_chunk_summary(chunk.text)
                    chunk_record = ChunkRecord(
                        id=None,
                        file_id=file_id,
                        chunk_text=chunk.text,
                        summary=summary,
                        chunk_type='content',
                        token_count=chunk.token_count,
                        emb_id=None
                    )
                    chunks.append(chunk_record)
            
            logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
            return file_id, chunks
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return -1, []
    
    def _create_chunk_summary(self, text: str, max_length: int = 200) -> str:
        """Create a summary for a chunk."""
        if len(text) <= max_length:
            return text
        
        # Try to find a good break point
        sentences = text.split('.')
        if sentences and len(sentences[0]) <= max_length:
            return sentences[0].strip() + '.'
        
        # Fallback to simple truncation
        return text[:max_length - 3].strip() + '...'
    
    def build_index(self, rebuild: bool = False) -> bool:
        """Build the semantic index from all files in the data directory."""
        if rebuild:
            logger.info("Rebuilding index from scratch...")
            self.clear_index()
        
        # Find all files
        if not self.data_path.exists():
            logger.error(f"Data path does not exist: {self.data_path}")
            return False
        
        files = [p for p in self.data_path.rglob("*") if p.is_file()]
        if not files:
            logger.warning("No files found to index")
            return False
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process files and collect chunks
        all_chunks = []
        all_chunk_texts = []
        
        for file_path in files:
            file_id, chunks = self.process_file(file_path)
            if file_id > 0 and chunks:
                all_chunks.extend(chunks)
                all_chunk_texts.extend([chunk.chunk_text for chunk in chunks])
        
        if not all_chunks:
            logger.warning("No chunks generated from files")
            return False
        
        logger.info(f"Generated {len(all_chunks)} total chunks")
        
        # Generate embeddings
        embeddings = self.embedding_manager.create_embeddings(all_chunk_texts)
        
        # Build FAISS index
        index = self.embedding_manager.build_faiss_index(embeddings)
        
        # Save FAISS index
        self.embedding_manager.save_index(index)
        
        # Update chunk records with embedding IDs
        for i, chunk in enumerate(all_chunks):
            chunk.emb_id = i
        
        # Insert chunks into database
        chunk_ids = self.db_manager.insert_chunks(all_chunks)
        
        # Update chunk links (prev/next relationships)
        file_chunks = {}
        for chunk_id, chunk in zip(chunk_ids, all_chunks):
            if chunk.file_id not in file_chunks:
                file_chunks[chunk.file_id] = []
            file_chunks[chunk.file_id].append(chunk_id)
        
        for file_id, file_chunk_ids in file_chunks.items():
            self.db_manager.update_chunk_links(file_chunk_ids)
        
        logger.info(f"Successfully built index with {len(all_chunks)} chunks")
        return True
    
    def clear_index(self):
        """Clear all index data."""
        logger.info("Clearing existing index data...")
        
        # Clear database
        self.db_manager.clear_all_data()
        
        # Remove FAISS index files
        try:
            for index_file in self.index_dir.glob("*.faiss"):
                index_file.unlink()
            for index_file in self.index_dir.glob("*.index"):
                index_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove some index files: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        cursor = self.db_manager.conn.cursor()
        
        # File statistics
        cursor.execute("SELECT COUNT(*) as total_files FROM files")
        total_files = cursor.fetchone()['total_files']
        
        cursor.execute("SELECT file_type, COUNT(*) as count FROM files GROUP BY file_type")
        files_by_type = {row['file_type']: row['count'] for row in cursor.fetchall()}
        
        # Chunk statistics
        cursor.execute("SELECT COUNT(*) as total_chunks FROM chunks")
        total_chunks = cursor.fetchone()['total_chunks']
        
        cursor.execute("SELECT chunk_type, COUNT(*) as count FROM chunks GROUP BY chunk_type")
        chunks_by_type = {row['chunk_type']: row['count'] for row in cursor.fetchall()}
        
        cursor.execute("SELECT AVG(token_count) as avg_tokens FROM chunks WHERE token_count > 0")
        avg_tokens = cursor.fetchone()['avg_tokens'] or 0
        
        return {
            'total_files': total_files,
            'files_by_type': files_by_type,
            'total_chunks': total_chunks,
            'chunks_by_type': chunks_by_type,
            'average_tokens_per_chunk': round(avg_tokens, 2),
            'embedding_dimension': self.embedding_manager.embedding_dim
        }
    
    def close(self):
        """Clean up resources."""
        self.db_manager.close()


# Legacy compatibility class
class Indexer(SemanticIndexer):
    """Legacy indexer class for backward compatibility."""
    
    def __init__(self, data_path: str):
        config = ConfigManager()
        config.set('DATA_PATH', data_path)
        super().__init__(config)
    
    def index_all(self):
        """Legacy method for building index."""
        return self.build_index(rebuild=True)
    
    def delete_index(self):
        """Legacy method for clearing index."""
        self.clear_index()