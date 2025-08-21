"""
Conversational RAG Orchestrator
Industry-grade conversational AI with memory, token budgeting, and continuous interaction.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Generator
from dataclasses import dataclass
import logging
import json

from utils import ConfigManager
from conversation_memory import ConversationMemoryDB, ConversationMessage
from prompt_manager import PromptAssembler
from retriever import SemanticRetriever
from rag import RAGSystem, LocalLLMManager
from model_api import LocalModelAPI

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a complete conversation turn with metadata."""
    conversation_id: str
    user_message: ConversationMessage
    assistant_message: Optional[ConversationMessage]
    doc_results: List[Dict[str, Any]]
    memory_results: List[Tuple[ConversationMessage, float]]
    prompt_metadata: Dict[str, Any]
    generation_time: float
    total_time: float
    confidence: float = 0.0
    citations: List[str] = None


class ConversationalRAG:
    """Main orchestrator for conversational RAG with memory."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        
        # Initialize components
        self.memory_db = ConversationMemoryDB(self.config)
        self.prompt_assembler = PromptAssembler(self.config)
        self.retriever = SemanticRetriever(self.config)
        self.llm_manager = LocalLLMManager(self.config)
        
        # Configuration
        self.doc_search_k = self.config.get('DOC_SEARCH_K', 12)
        self.memory_search_k = self.config.get('MEMORY_SEARCH_K', 8)
        self.rerank_k = self.config.get('RERANK_K', 5)
        self.sliding_window_size = self.config.get('SLIDING_WINDOW_SIZE', 6)
        self.summarization_threshold = self.config.get('SUMMARIZATION_THRESHOLD', 10)
        self.confidence_threshold = self.config.get('CONFIDENCE_THRESHOLD', 0.7)
        
        # Session state
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        
        logger.info("ConversationalRAG initialized")
    
    def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation session."""
        conversation_id = self.memory_db.create_conversation(title)
        
        # Initialize session state
        self.active_conversations[conversation_id] = {
            'created_at': time.time(),
            'turn_count': 0,
            'last_activity': time.time(),
            'topic': None,
            'summary_count': 0
        }
        
        logger.info(f"Created conversation session: {conversation_id}")
        return conversation_id
    
    def process_user_message(self, 
                           conversation_id: str, 
                           user_input: str,
                           stream_response: bool = False) -> ConversationTurn:
        """Process a user message and generate response with full RAG pipeline."""
        
        start_time = time.time()
        
        # Update session state
        if conversation_id not in self.active_conversations:
            self.active_conversations[conversation_id] = {
                'created_at': time.time(),
                'turn_count': 0,
                'last_activity': time.time(),
                'topic': None,
                'summary_count': 0
            }
        
        session = self.active_conversations[conversation_id]
        session['last_activity'] = time.time()
        session['turn_count'] += 1
        
        # Step 1: Persist user message
        prev_id = self._get_last_message_id(conversation_id)
        user_message = self.memory_db.add_message(
            message_type='user_msg',
            conversation_id=conversation_id,
            content=user_input,
            prev_id=prev_id,
            source='user_interface'
        )
        
        logger.info(f"Processing user message in conversation {conversation_id}")
        
        # Step 2: Retrieve documents and memory
        doc_results, memory_results = self._retrieve_context(
            query=user_input,
            conversation_id=conversation_id
        )
        
        # Step 3: Get sliding window
        sliding_window = self.memory_db.get_sliding_window(
            conversation_id, 
            self.sliding_window_size
        )
        
        # Step 4: Assemble prompt with token budgeting
        prompt, prompt_metadata = self.prompt_assembler.assemble_prompt(
            query=user_input,
            sliding_window=sliding_window,
            memory_items=memory_results,
            doc_results=doc_results,
            conversation_id=conversation_id
        )
        
        # Step 5: Generate response
        generation_start = time.time()
        
        if stream_response:
            # For streaming, we'll return a generator
            response_generator = self._generate_streaming_response(
                prompt, conversation_id, user_message.id
            )
            
            # Create partial turn object for streaming
            turn = ConversationTurn(
                conversation_id=conversation_id,
                user_message=user_message,
                assistant_message=None,  # Will be set when streaming completes
                doc_results=doc_results,
                memory_results=memory_results,
                prompt_metadata=prompt_metadata,
                generation_time=0.0,
                total_time=0.0
            )
            
            return turn, response_generator
        
        else:
            # Synchronous generation
            response = self.llm_manager.generate(prompt)
            generation_time = time.time() - generation_start
            
            # Step 6: Extract citations and calculate confidence
            citations, confidence = self._analyze_response(response, doc_results)
            
            # Step 7: Persist assistant message
            assistant_message = self.memory_db.add_message(
                message_type='assistant_msg',
                conversation_id=conversation_id,
                content=response,
                prev_id=user_message.id,
                confidence=confidence,
                citations=citations
            )
            
            # Step 8: Check if summarization is needed
            if self.memory_db.should_summarize(conversation_id, self.summarization_threshold):
                self._create_summary(conversation_id)
            
            total_time = time.time() - start_time
            
            # Create turn object
            turn = ConversationTurn(
                conversation_id=conversation_id,
                user_message=user_message,
                assistant_message=assistant_message,
                doc_results=doc_results,
                memory_results=memory_results,
                prompt_metadata=prompt_metadata,
                generation_time=generation_time,
                total_time=total_time,
                confidence=confidence,
                citations=citations
            )
            
            # Log the interaction
            self._log_turn(turn)
            
            logger.info(f"Completed turn in {total_time:.2f}s (generation: {generation_time:.2f}s)")
            return turn
    
    def _retrieve_context(self, 
                         query: str, 
                         conversation_id: str) -> Tuple[List[Dict[str, Any]], List[Tuple[ConversationMessage, float]]]:
        """Retrieve both document and memory context."""
        
        # Document retrieval
        doc_results = self.retriever.search(query, top_k=self.doc_search_k)
        
        # Convert SearchResult objects to dictionaries if needed
        if doc_results and hasattr(doc_results[0], 'to_dict'):
            doc_results = [result.to_dict() for result in doc_results]
        
        # Memory retrieval
        memory_results = self.memory_db.search_memory(
            query=query,
            conversation_id=conversation_id,
            message_types=['assistant_msg', 'user_msg', 'summary'],
            top_k=self.memory_search_k,
            confidence_threshold=self.confidence_threshold
        )
        
        logger.debug(f"Retrieved {len(doc_results)} docs, {len(memory_results)} memory items")
        return doc_results, memory_results
    
    def _generate_streaming_response(self, 
                                   prompt: str, 
                                   conversation_id: str, 
                                   user_message_id: str) -> Generator[str, None, ConversationMessage]:
        """Generate streaming response (placeholder for future streaming implementation)."""
        # For now, implement as regular generation
        # In a full implementation, this would use streaming APIs
        
        response = self.llm_manager.generate(prompt)
        
        # Simulate streaming by yielding chunks
        chunk_size = 10
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            yield chunk
            time.sleep(0.01)  # Simulate streaming delay
        
        # After streaming is complete, persist the message
        citations, confidence = self._analyze_response(response, [])
        
        assistant_message = self.memory_db.add_message(
            message_type='assistant_msg',
            conversation_id=conversation_id,
            content=response,
            prev_id=user_message_id,
            confidence=confidence,
            citations=citations
        )
        
        return assistant_message
    
    def _analyze_response(self, 
                         response: str, 
                         doc_results: List[Dict[str, Any]]) -> Tuple[List[str], float]:
        """Analyze response to extract citations and calculate confidence."""
        
        citations = []
        confidence = 0.5  # Base confidence
        
        # Extract citations from response
        # Look for patterns like [doc:filename] or [memory:turn]
        import re
        
        doc_citations = re.findall(r'\[doc:([^\]]+)\]', response)
        memory_citations = re.findall(r'\[memory:([^\]]+)\]', response)
        
        citations.extend([f"doc:{cite}" for cite in doc_citations])
        citations.extend([f"memory:{cite}" for cite in memory_citations])
        
        # Calculate confidence based on citations and content analysis
        if citations:
            confidence += 0.3  # Boost for having citations
        
        # Check if response uses document content
        if doc_results:
            doc_texts = [result.get('text', '') for result in doc_results]
            
            # Simple heuristic: check for overlapping phrases
            response_words = set(response.lower().split())
            doc_words = set()
            for doc_text in doc_texts:
                doc_words.update(doc_text.lower().split())
            
            overlap = len(response_words.intersection(doc_words))
            if overlap > 5:  # Arbitrary threshold
                confidence += 0.2
        
        # Ensure confidence is in valid range
        confidence = min(1.0, max(0.0, confidence))
        
        return citations, confidence
    
    def _create_summary(self, conversation_id: str):
        """Create a rolling summary of the conversation."""
        
        logger.info(f"Creating summary for conversation {conversation_id}")
        
        # Get messages to summarize (not archived)
        messages = self.memory_db.get_conversation_messages(
            conversation_id,
            message_types=['user_msg', 'assistant_msg'],
            include_archived=False
        )
        
        if len(messages) < 4:  # Need at least 2 exchanges
            return
        
        # Build summary prompt
        conversation_text = []
        for msg in messages[-10:]:  # Last 10 messages
            if msg.type == 'user_msg':
                conversation_text.append(f"User: {msg.content}")
            elif msg.type == 'assistant_msg':
                conversation_text.append(f"Assistant: {msg.content}")
        
        summary_prompt = f"""Summarize the following conversation concisely, focusing on:
- Key topics discussed
- Important decisions or conclusions
- Open questions or TODOs
- Relevant document references

Conversation:
{chr(10).join(conversation_text)}

Summary (max 400 tokens):"""
        
        # Generate summary
        summary_content = self.llm_manager.generate(summary_prompt, max_tokens=400)
        
        # Store summary
        last_message_id = messages[-1].id if messages else None
        summary_message = self.memory_db.add_message(
            message_type='summary',
            conversation_id=conversation_id,
            content=summary_content,
            prev_id=last_message_id,
            confidence=0.8,  # Summaries are generally reliable
            source='system_summarization'
        )
        
        # Archive older messages (keep recent ones)
        if len(messages) > 6:
            messages_to_archive = [msg.id for msg in messages[:-6]]
            self.memory_db.archive_messages(messages_to_archive)
        
        # Update conversation metadata
        self.memory_db.update_conversation_summary_timestamp(conversation_id)
        
        logger.info(f"Created summary and archived {len(messages) - 6} messages")
    
    def _get_last_message_id(self, conversation_id: str) -> Optional[str]:
        """Get the ID of the last message in a conversation."""
        
        messages = self.memory_db.get_conversation_messages(
            conversation_id,
            limit=1
        )
        
        return messages[-1].id if messages else None
    
    def _log_turn(self, turn: ConversationTurn):
        """Log conversation turn for observability."""
        
        # Log to enhanced search history
        self.memory_db.log_search(
            conversation_id=turn.conversation_id,
            query=turn.user_message.content,
            query_type='conversational_rag',
            results_count=len(turn.doc_results) + len(turn.memory_results),
            doc_results_count=len(turn.doc_results),
            memory_results_count=len(turn.memory_results),
            latency_ms=turn.total_time * 1000
        )
        
        # Detailed logging
        logger.info(f"Turn completed: {turn.conversation_id}")
        logger.info(f"  - Doc results: {len(turn.doc_results)}")
        logger.info(f"  - Memory results: {len(turn.memory_results)}")
        logger.info(f"  - Confidence: {turn.confidence:.2f}")
        logger.info(f"  - Citations: {len(turn.citations or [])}")
        logger.info(f"  - Total time: {turn.total_time:.2f}s")
        logger.info(f"  - Tokens: {turn.prompt_metadata.get('total_tokens', 0)}")
    
    def get_conversation_history(self, 
                               conversation_id: str, 
                               limit: Optional[int] = None) -> List[ConversationMessage]:
        """Get conversation history."""
        
        return self.memory_db.get_conversation_messages(
            conversation_id,
            message_types=['user_msg', 'assistant_msg', 'summary'],
            limit=limit,
            include_archived=False
        )
    
    def get_conversations(self) -> List[Dict[str, Any]]:
        """Get list of all conversations."""
        
        return self.memory_db.get_conversations()
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation (mark as archived)."""
        
        cursor = self.memory_db.conn.cursor()
        cursor.execute("""
            UPDATE conversations 
            SET archived = TRUE 
            WHERE id = ?
        """, (conversation_id,))
        
        cursor.execute("""
            UPDATE conversation_messages 
            SET archived = TRUE 
            WHERE conversation_id = ?
        """, (conversation_id,))
        
        self.memory_db.conn.commit()
        
        # Remove from active sessions
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
        
        logger.info(f"Archived conversation: {conversation_id}")
    
    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """Get statistics for a conversation."""
        
        cursor = self.memory_db.conn.cursor()
        
        # Basic stats
        cursor.execute("""
            SELECT COUNT(*) as total_messages,
                   COUNT(CASE WHEN type = 'user_msg' THEN 1 END) as user_messages,
                   COUNT(CASE WHEN type = 'assistant_msg' THEN 1 END) as assistant_messages,
                   COUNT(CASE WHEN type = 'summary' THEN 1 END) as summaries,
                   AVG(CASE WHEN type = 'assistant_msg' THEN confidence END) as avg_confidence,
                   SUM(token_count) as total_tokens
            FROM conversation_messages 
            WHERE conversation_id = ? AND archived = FALSE
        """, (conversation_id,))
        
        stats = dict(cursor.fetchone())
        
        # Session info
        session = self.active_conversations.get(conversation_id, {})
        stats.update({
            'turn_count': session.get('turn_count', 0),
            'active': conversation_id in self.active_conversations,
            'last_activity': session.get('last_activity'),
            'topic': session.get('topic')
        })
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        
        self.memory_db.close()
        if hasattr(self.llm_manager, 'cleanup'):
            self.llm_manager.cleanup()
        
        logger.info("ConversationalRAG cleanup completed")