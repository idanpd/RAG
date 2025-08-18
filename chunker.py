import re
import logging
from typing import List, Optional, Union
from dataclasses import dataclass

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    text: str
    start_pos: int = 0
    end_pos: int = 0
    token_count: int = 0
    chunk_id: Optional[str] = None
    source_info: Optional[dict] = None


class TokenCounter:
    """Token counter with fallback to word-based approximation."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.encoder = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.encoding_for_model(model_name)
            except KeyError:
                try:
                    self.encoder = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    logger.warning("Failed to load tiktoken encoder, using word approximation")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception:
                pass
        
        # Fallback: approximate tokens as words * 1.3
        return int(len(text.split()) * 1.3)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to tokens."""
        if self.encoder:
            try:
                return self.encoder.encode(text)
            except Exception:
                pass
        return []
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text."""
        if self.encoder:
            try:
                return self.encoder.decode(tokens)
            except Exception:
                pass
        return ""


class SemanticChunker:
    """Advanced chunker with semantic awareness and token-based splitting."""
    
    def __init__(self, 
                 chunk_size: int = 500, 
                 overlap: int = 100,
                 min_chunk_size: int = 50,
                 max_chunk_size: int = 1000,
                 token_model: str = "gpt-3.5-turbo"):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size in tokens
            max_chunk_size: Maximum chunk size in tokens
            token_model: Model name for tokenization
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.token_counter = TokenCounter(token_model)
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure."""
        # First split by paragraphs
        paragraphs = self.paragraph_breaks.split(text)
        sentences = []
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Split paragraph into sentences
            para_sentences = self.sentence_endings.split(para)
            for i, sent in enumerate(para_sentences):
                if sent.strip():
                    # Add back sentence ending except for last sentence
                    if i < len(para_sentences) - 1:
                        sentences.append(sent.strip() + '.')
                    else:
                        sentences.append(sent.strip())
            
            # Add paragraph break marker
            if len(sentences) > 0 and not sentences[-1].endswith('\n\n'):
                sentences[-1] += '\n\n'
        
        return [s for s in sentences if s.strip()]
    
    def _merge_small_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Merge chunks that are too small with adjacent chunks."""
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # If current chunk is too small, try to merge with next
            if (current.token_count < self.min_chunk_size and 
                i + 1 < len(chunks)):
                next_chunk = chunks[i + 1]
                
                # Check if merged chunk would be reasonable size
                merged_text = current.text + " " + next_chunk.text
                merged_tokens = self.token_counter.count_tokens(merged_text)
                
                if merged_tokens <= self.max_chunk_size:
                    merged_chunk = TextChunk(
                        text=merged_text,
                        start_pos=current.start_pos,
                        end_pos=next_chunk.end_pos,
                        token_count=merged_tokens,
                        chunk_id=f"merged_{current.chunk_id}_{next_chunk.chunk_id}"
                    )
                    merged.append(merged_chunk)
                    i += 2  # Skip next chunk as it's been merged
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def chunk_text(self, text: str, source_info: Optional[dict] = None) -> List[TextChunk]:
        """
        Chunk text into semantic chunks with token awareness.
        
        Args:
            text: Text to chunk
            source_info: Optional metadata about the source
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Clean text
        text = text.strip()
        
        # If text is small enough, return as single chunk
        total_tokens = self.token_counter.count_tokens(text)
        if total_tokens <= self.chunk_size:
            return [TextChunk(
                text=text,
                start_pos=0,
                end_pos=len(text),
                token_count=total_tokens,
                chunk_id="single",
                source_info=source_info
            )]
        
        # Split into sentences
        sentences = self._split_by_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        chunk_counter = 0
        
        for sentence in sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            
            # If single sentence is too large, split it further
            if sentence_tokens > self.max_chunk_size:
                # If we have accumulated sentences, save them first
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences).strip()
                    chunks.append(TextChunk(
                        text=chunk_text,
                        token_count=current_tokens,
                        chunk_id=f"chunk_{chunk_counter}",
                        source_info=source_info
                    ))
                    chunk_counter += 1
                    current_chunk_sentences = []
                    current_tokens = 0
                
                # Split large sentence by words
                words = sentence.split()
                word_chunk = []
                word_tokens = 0
                
                for word in words:
                    word_token_count = self.token_counter.count_tokens(word + " ")
                    
                    if word_tokens + word_token_count > self.chunk_size and word_chunk:
                        chunk_text = " ".join(word_chunk).strip()
                        chunks.append(TextChunk(
                            text=chunk_text,
                            token_count=word_tokens,
                            chunk_id=f"chunk_{chunk_counter}",
                            source_info=source_info
                        ))
                        chunk_counter += 1
                        
                        # Handle overlap
                        if self.overlap > 0:
                            overlap_words = word_chunk[-min(len(word_chunk), self.overlap // 4):]
                            word_chunk = overlap_words + [word]
                            word_tokens = self.token_counter.count_tokens(" ".join(word_chunk))
                        else:
                            word_chunk = [word]
                            word_tokens = word_token_count
                    else:
                        word_chunk.append(word)
                        word_tokens += word_token_count
                
                # Add remaining words
                if word_chunk:
                    current_chunk_sentences = [" ".join(word_chunk)]
                    current_tokens = word_tokens
                
                continue
            
            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk_sentences:
                # Save current chunk
                chunk_text = " ".join(current_chunk_sentences).strip()
                chunks.append(TextChunk(
                    text=chunk_text,
                    token_count=current_tokens,
                    chunk_id=f"chunk_{chunk_counter}",
                    source_info=source_info
                ))
                chunk_counter += 1
                
                # Handle overlap
                if self.overlap > 0 and len(current_chunk_sentences) > 1:
                    overlap_count = min(len(current_chunk_sentences), 
                                      max(1, self.overlap // 100))  # Rough sentence overlap
                    overlap_sentences = current_chunk_sentences[-overlap_count:]
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_tokens = self.token_counter.count_tokens(" ".join(current_chunk_sentences))
                else:
                    current_chunk_sentences = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            chunks.append(TextChunk(
                text=chunk_text,
                token_count=current_tokens,
                chunk_id=f"chunk_{chunk_counter}",
                source_info=source_info
            ))
        
        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)
        
        # Update positions
        text_position = 0
        for chunk in chunks:
            chunk.start_pos = text_position
            chunk.end_pos = text_position + len(chunk.text)
            text_position = chunk.end_pos
        
        return chunks


class Chunker:
    """Legacy chunker for backward compatibility."""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 120):
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.semantic_chunker = SemanticChunker(
            chunk_size=chunk_size,
            overlap=overlap
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text and return list of strings (legacy interface)."""
        chunks = self.semantic_chunker.chunk_text(text)
        return [chunk.text for chunk in chunks]