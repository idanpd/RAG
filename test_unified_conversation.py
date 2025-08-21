#!/usr/bin/env python3
"""
Test script for unified conversational RAG.
Tests document + conversation memory integration.
"""

import sys
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path

from utils import ConfigManager, setup_logger
from indexer import SemanticIndexer
from retriever import SemanticRetriever
from rag import LocalLLMManager


class TestConversationalRAG:
    """Test the unified conversational RAG system."""
    
    def __init__(self):
        self.config = ConfigManager()
        self.logger = setup_logger('INFO')
        
        # Initialize components
        self.indexer = SemanticIndexer(self.config)
        self.retriever = SemanticRetriever(self.config)
        self.llm_manager = LocalLLMManager(self.config)
        
        self.logger.info("Test system initialized")
    
    def add_conversation_message(self, conversation_id: str, message_type: str, 
                               content: str, prev_id: int = None,
                               confidence: float = 0.5, citations: list = None) -> int:
        """Add a message to conversation using the existing chunks table."""
        
        cursor = self.indexer.db_manager.conn.cursor()
        
        # Calculate token count
        token_count = len(content.split())
        citations_json = json.dumps(citations) if citations else None
        
        cursor.execute("""
            INSERT INTO chunks 
            (file_id, chunk_text, summary, chunk_type, token_count, 
             conversation_id, timestamp, source, confidence, citations, archived)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            None,  # No file for conversation messages
            content,
            content[:200],  # Summary
            message_type,
            token_count,
            conversation_id,
            datetime.now(timezone.utc).isoformat(),
            'test_script',
            confidence,
            citations_json,
            False
        ))
        
        message_id = cursor.lastrowid
        self.indexer.db_manager.conn.commit()
        
        return message_id
    
    def search_unified(self, query: str, conversation_id: str):
        """Search both documents and conversation memory."""
        
        # Search documents (existing functionality)
        doc_results = self.retriever.search(query, top_k=5)
        if doc_results and hasattr(doc_results[0], 'to_dict'):
            doc_results = [result.to_dict() for result in doc_results]
        
        # Search conversation memory
        cursor = self.indexer.db_manager.conn.cursor()
        cursor.execute("""
            SELECT * FROM chunks 
            WHERE conversation_id = ? 
            AND chunk_type IN ('assistant_msg', 'user_msg', 'summary')
            AND archived = FALSE
            ORDER BY timestamp DESC
            LIMIT 5
        """, (conversation_id,))
        
        memory_results = []
        for row in cursor.fetchall():
            message = dict(row)
            if message['citations']:
                try:
                    message['citations'] = json.loads(message['citations'])
                except:
                    message['citations'] = []
            else:
                message['citations'] = []
            memory_results.append(message)
        
        return doc_results, memory_results
    
    def assemble_prompt(self, query: str, doc_results: list, memory_results: list) -> str:
        """Assemble unified prompt with documents and conversation context."""
        
        system_prompt = """You are a helpful AI assistant with access to documents and conversation history.

Answer questions using the provided context from both documents and conversation history.
Cite sources using [doc:filename] for documents and [memory:turn] for conversation context.
Be concise but comprehensive.

"""
        
        # Document context
        docs_context = ""
        if doc_results:
            docs_context = "Relevant documents:\n"
            for result in doc_results:
                filename = result.get('path', 'Unknown').split('/')[-1]
                content = result.get('text', '')[:400]
                docs_context += f"[doc:{filename}] {content}\n"
            docs_context += "\n"
        
        # Memory context
        memory_context = ""
        if memory_results:
            memory_context = "Conversation history:\n"
            for msg in memory_results[-3:]:  # Last 3 messages
                if msg['chunk_type'] == 'user_msg':
                    memory_context += f"User: {msg['chunk_text']}\n"
                elif msg['chunk_type'] == 'assistant_msg':
                    memory_context += f"Assistant: {msg['chunk_text']}\n"
            memory_context += "\n"
        
        return system_prompt + docs_context + memory_context + f"User: {query}\nAssistant: "
    
    def test_conversation_flow(self):
        """Test the complete conversation flow."""
        print("🧪 Testing Unified Conversational RAG")
        print("=" * 50)
        
        # Check if index exists
        try:
            if not self.retriever.dense_retriever.index:
                print("❌ No search index found. Please build index first:")
                print("python main.py --build-index")
                return False
        except Exception as e:
            print(f"❌ Error checking index: {e}")
            return False
        
        # Check if models are available
        available_models = self.llm_manager.get_available_models()
        if not available_models:
            print("❌ No models available. Please download models:")
            print("python download_models.py --recommended")
            return False
        
        loaded_models = [m for m in available_models if m['is_loaded']]
        if not loaded_models:
            print("⚠️  No models loaded. Loading default model...")
            result = self.llm_manager.load_model(available_models[0]['name'])
            if not result['success']:
                print(f"❌ Failed to load model: {result}")
                return False
        
        print(f"✅ Using model: {self.llm_manager.active_model}")
        
        # Create test conversation
        conversation_id = str(uuid.uuid4())
        print(f"🆕 Created test conversation: {conversation_id[:8]}")
        
        # Test queries
        test_queries = [
            "What files do you have access to?",
            "Can you summarize what we've discussed?",
            "What is the main topic in the documents?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Turn {i} ---")
            print(f"User: {query}")
            
            # Add user message
            prev_id = self._get_last_message_id(conversation_id)
            user_msg_id = self.add_conversation_message(
                conversation_id, 'user_msg', query, prev_id, 1.0
            )
            
            # Search unified context
            doc_results, memory_results = self.search_unified(query, conversation_id)
            
            print(f"📚 Found {len(doc_results)} documents, {len(memory_results)} memory items")
            
            # Assemble prompt
            prompt = self.assemble_prompt(query, doc_results, memory_results)
            
            # Generate response
            response = self.llm_manager.generate(prompt)
            
            # Analyze and store response
            citations, confidence = self._analyze_response(response, doc_results)
            assistant_msg_id = self.add_conversation_message(
                conversation_id, 'assistant_msg', response, user_msg_id, confidence, citations
            )
            
            print(f"Assistant: {response}")
            print(f"📊 Confidence: {confidence:.2f}, Citations: {len(citations)}")
            
            if citations:
                print(f"🔗 Citations: {', '.join(citations)}")
        
        print("\n✅ Conversation flow test completed successfully!")
        return True
    
    def _get_last_message_id(self, conversation_id: str) -> int:
        """Get the last message ID in a conversation."""
        cursor = self.indexer.db_manager.conn.cursor()
        cursor.execute("""
            SELECT id FROM chunks 
            WHERE conversation_id = ? AND chunk_type IN ('user_msg', 'assistant_msg')
            ORDER BY timestamp DESC LIMIT 1
        """, (conversation_id,))
        
        result = cursor.fetchone()
        return result[0] if result else None
    
    def _analyze_response(self, response: str, doc_results: list) -> tuple:
        """Analyze response for citations and confidence."""
        
        citations = []
        confidence = 0.5
        
        # Extract citations
        import re
        doc_citations = re.findall(r'\[doc:([^\]]+)\]', response)
        citations.extend([f"doc:{cite}" for cite in doc_citations])
        
        # Calculate confidence
        if citations:
            confidence += 0.3
        
        if doc_results:
            response_words = set(response.lower().split())
            doc_words = set()
            for result in doc_results:
                doc_words.update(result.get('text', '').lower().split())
            
            overlap = len(response_words.intersection(doc_words))
            if overlap > 5:
                confidence += 0.2
        
        confidence = min(1.0, max(0.0, confidence))
        return citations, confidence
    
    def test_document_retrieval(self):
        """Test that document retrieval is working."""
        print("\n🔍 Testing document retrieval...")
        
        test_query = "machine learning"
        results = self.retriever.search(test_query, top_k=3)
        
        if results:
            if hasattr(results[0], 'to_dict'):
                results = [r.to_dict() for r in results]
            
            print(f"✅ Found {len(results)} documents for '{test_query}':")
            for i, result in enumerate(results, 1):
                filename = result.get('path', 'Unknown').split('/')[-1]
                score = result.get('score', 0)
                print(f"  {i}. {filename} (score: {score:.3f})")
        else:
            print("⚠️  No documents found. Please ensure you have indexed documents.")
        
        return len(results) > 0 if results else False


def main():
    """Main test function."""
    test_system = TestConversationalRAG()
    
    print("🚀 Testing Unified Conversational RAG System")
    
    # Test document retrieval first
    if not test_system.test_document_retrieval():
        print("\n❌ Document retrieval test failed")
        print("Please build the index first: python main.py --build-index")
        return 1
    
    # Test conversation flow
    if not test_system.test_conversation_flow():
        print("\n❌ Conversation flow test failed")
        return 1
    
    print("\n🎉 All tests passed! The unified conversational RAG system is working.")
    print("\nNext steps:")
    print("1. Run the Streamlit UI: streamlit run streamlit_unified_rag.py")
    print("2. Or use CLI mode: python main.py --conversation")
    print("3. Upload files in Streamlit to test automatic indexing")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())