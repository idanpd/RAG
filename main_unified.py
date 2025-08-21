#!/usr/bin/env python3
"""
Unified Main Script for Conversational RAG with Automatic Document Integration
Fixes the issue where uploaded data isn't included in conversations.
"""

import sys
import argparse
import logging
import time
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from utils import ConfigManager, setup_logger
from indexer import SemanticIndexer
from retriever import SemanticRetriever
from rag import LocalLLMManager
from model_api import LocalModelAPI
from caching_system import EmbeddingCache


class UnifiedConversationalRAG:
    """
    Unified conversational RAG that automatically includes both:
    1. Indexed documents from data path
    2. Conversation memory
    
    This fixes the core issue where conversations didn't access uploaded/indexed data.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = setup_logger(self.config.get('LOG_LEVEL', 'INFO'))
        
        # Initialize core components
        self.indexer = SemanticIndexer(self.config)
        self.retriever = SemanticRetriever(self.config)
        self.llm_manager = LocalLLMManager(self.config)
        self.model_api = LocalModelAPI(self.config)
        self.cache = EmbeddingCache(self.config)
        
        # Conversation settings
        self.doc_search_k = self.config.get('DOC_SEARCH_K', 8)
        self.memory_search_k = self.config.get('MEMORY_SEARCH_K', 6)
        self.confidence_threshold = self.config.get('CONFIDENCE_THRESHOLD', 0.7)
        
        self.logger.info("UnifiedConversationalRAG initialized")
    
    def add_conversation_message(self, conversation_id: str, message_type: str, 
                               content: str, prev_id: Optional[int] = None,
                               confidence: float = 0.5, citations: List[str] = None) -> int:
        """Add a message to conversation using the unified chunks table."""
        
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
            'unified_rag',
            confidence,
            citations_json,
            False
        ))
        
        message_id = cursor.lastrowid
        self.indexer.db_manager.conn.commit()
        
        return message_id
    
    def search_documents_and_memory(self, query: str, conversation_id: str) -> Tuple[List[Dict], List[Dict]]:
        """
        CORE FIX: Search both documents AND conversation memory in one unified query.
        This ensures conversations always have access to indexed documents.
        """
        
        # 1. Search documents using existing retriever (this includes ALL indexed documents)
        doc_results = self.retriever.search(query, top_k=self.doc_search_k)
        if doc_results and hasattr(doc_results[0], 'to_dict'):
            doc_results = [result.to_dict() for result in doc_results]
        
        # 2. Search conversation memory from the same chunks table
        memory_results = []
        if conversation_id:
            cursor = self.indexer.db_manager.conn.cursor()
            cursor.execute("""
                SELECT * FROM chunks 
                WHERE conversation_id = ? 
                AND chunk_type IN ('assistant_msg', 'user_msg', 'summary')
                AND archived = FALSE
                AND (
                    chunk_type != 'assistant_msg' 
                    OR confidence >= ? 
                    OR citations IS NOT NULL
                )
                ORDER BY timestamp DESC
                LIMIT ?
            """, (conversation_id, self.confidence_threshold, self.memory_search_k))
            
            for row in cursor.fetchall():
                message = dict(row)
                # Parse citations
                if message['citations']:
                    try:
                        message['citations'] = json.loads(message['citations'])
                    except:
                        message['citations'] = []
                else:
                    message['citations'] = []
                memory_results.append(message)
        
        self.logger.info(f"Retrieved {len(doc_results)} documents + {len(memory_results)} memory items")
        return doc_results, memory_results
    
    def assemble_unified_prompt(self, query: str, conversation_id: str,
                              doc_results: List[Dict], memory_results: List[Dict]) -> str:
        """Assemble prompt that includes BOTH documents and conversation context."""
        
        # System prompt emphasizing unified context
        system_prompt = """You are a helpful AI assistant with access to both documents and conversation history.

IMPORTANT: You have access to:
1. Indexed documents from the knowledge base (cite as [doc:filename])
2. Previous conversation turns (cite as [memory:turn])

Instructions:
- Always search and use BOTH document knowledge AND conversation history
- Cite sources clearly: [doc:filename] for documents, [memory:turn] for conversation
- If information isn't in either context, say so clearly
- Maintain conversation continuity
- Be comprehensive but concise

"""
        
        # Recent conversation context (sliding window)
        conversation_context = ""
        if memory_results:
            conversation_context = "Recent conversation:\n"
            recent_messages = [msg for msg in memory_results if msg['chunk_type'] in ['user_msg', 'assistant_msg']]
            for msg in recent_messages[-4:]:  # Last 4 messages
                role = "User" if msg['chunk_type'] == 'user_msg' else "Assistant"
                conversation_context += f"{role}: {msg['chunk_text']}\n"
            conversation_context += "\n"
        
        # Document context (knowledge base)
        docs_context = ""
        if doc_results:
            docs_context = "Relevant documents from knowledge base:\n"
            for result in doc_results[:5]:  # Top 5 documents
                filename = result.get('path', 'Unknown').split('/')[-1]
                content = result.get('text', '')[:400]  # Limit content for token budget
                docs_context += f"[doc:{filename}] {content}\n"
            docs_context += "\n"
        
        # Memory context (relevant past exchanges)
        memory_context = ""
        if memory_results:
            relevant_memory = [msg for msg in memory_results if msg['chunk_type'] == 'summary']
            if relevant_memory:
                memory_context = "Relevant conversation history:\n"
                for msg in relevant_memory[:2]:  # Top 2 summaries
                    memory_context += f"[memory:summary] {msg['chunk_text']}\n"
                memory_context += "\n"
        
        # Assemble final prompt
        prompt = (system_prompt + 
                 conversation_context + 
                 docs_context + 
                 memory_context + 
                 f"User: {query}\nAssistant: ")
        
        return prompt
    
    def analyze_response(self, response: str, doc_results: List[Dict]) -> Tuple[List[str], float]:
        """Analyze response for citations and confidence."""
        
        citations = []
        confidence = 0.5  # Base confidence
        
        # Extract citations
        import re
        doc_citations = re.findall(r'\[doc:([^\]]+)\]', response)
        memory_citations = re.findall(r'\[memory:([^\]]+)\]', response)
        
        citations.extend([f"doc:{cite}" for cite in doc_citations])
        citations.extend([f"memory:{cite}" for cite in memory_citations])
        
        # Calculate confidence based on citations and content overlap
        if citations:
            confidence += 0.3  # Boost for having citations
        
        # Check content overlap with documents
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
    
    def process_conversation_turn(self, conversation_id: str, user_input: str) -> Dict[str, Any]:
        """Process a complete conversation turn with unified document + memory retrieval."""
        
        start_time = time.time()
        
        # Step 1: Add user message
        prev_id = self._get_last_message_id(conversation_id)
        user_msg_id = self.add_conversation_message(
            conversation_id, 'user_msg', user_input, prev_id, 1.0
        )
        
        # Step 2: UNIFIED SEARCH - documents AND memory together
        doc_results, memory_results = self.search_documents_and_memory(user_input, conversation_id)
        
        # Step 3: Assemble unified prompt
        prompt = self.assemble_unified_prompt(user_input, conversation_id, doc_results, memory_results)
        
        # Step 4: Check cache first
        cached_response = self.cache.get_response(user_input, prompt, self.llm_manager.active_model)
        
        if cached_response:
            response = cached_response['response']
            confidence = cached_response['confidence']
            citations = cached_response['citations']
            generation_time = 0.0
            self.logger.info("Using cached response")
        else:
            # Step 5: Generate response
            generation_start = time.time()
            response = self.llm_manager.generate(prompt)
            generation_time = time.time() - generation_start
            
            # Step 6: Analyze response
            citations, confidence = self.analyze_response(response, doc_results)
            
            # Cache the response
            self.cache.store_response(user_input, prompt, self.llm_manager.active_model, 
                                    response, confidence, citations)
        
        # Step 7: Add assistant message
        assistant_msg_id = self.add_conversation_message(
            conversation_id, 'assistant_msg', response, user_msg_id, confidence, citations
        )
        
        total_time = time.time() - start_time
        
        return {
            'user_message_id': user_msg_id,
            'assistant_message_id': assistant_msg_id,
            'response': response,
            'confidence': confidence,
            'citations': citations,
            'doc_results': doc_results,
            'memory_results': memory_results,
            'generation_time': generation_time,
            'total_time': total_time,
            'cached': cached_response is not None
        }
    
    def _get_last_message_id(self, conversation_id: str) -> Optional[int]:
        """Get the last message ID in a conversation."""
        cursor = self.indexer.db_manager.conn.cursor()
        cursor.execute("""
            SELECT id FROM chunks 
            WHERE conversation_id = ? AND chunk_type IN ('user_msg', 'assistant_msg')
            ORDER BY timestamp DESC LIMIT 1
        """, (conversation_id,))
        
        result = cursor.fetchone()
        return result[0] if result else None
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history."""
        cursor = self.indexer.db_manager.conn.cursor()
        cursor.execute("""
            SELECT * FROM chunks 
            WHERE conversation_id = ? AND archived = FALSE
            AND chunk_type IN ('user_msg', 'assistant_msg', 'summary')
            ORDER BY timestamp ASC
        """, (conversation_id,))
        
        messages = []
        for row in cursor.fetchall():
            message = dict(row)
            if message['citations']:
                try:
                    message['citations'] = json.loads(message['citations'])
                except:
                    message['citations'] = []
            else:
                message['citations'] = []
            messages.append(message)
        
        return messages
    
    def get_conversations(self) -> List[Dict]:
        """Get list of conversations."""
        cursor = self.indexer.db_manager.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT conversation_id,
                   MIN(timestamp) as created_at,
                   MAX(timestamp) as updated_at,
                   COUNT(*) as message_count
            FROM chunks 
            WHERE conversation_id IS NOT NULL AND archived = FALSE
            GROUP BY conversation_id
            ORDER BY MAX(timestamp) DESC
            LIMIT 20
        """)
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                'id': row[0],
                'title': f"Chat {row[0][:8]}",
                'created_at': row[1],
                'updated_at': row[2],
                'message_count': row[3]
            })
        
        return conversations
    
    def conversation_mode(self):
        """Enhanced conversation mode with unified document + memory retrieval."""
        print("🤖 Unified Conversational RAG Mode")
        print("=" * 60)
        print("✨ Now includes BOTH indexed documents AND conversation memory!")
        
        # Check if index exists
        try:
            if not self.retriever.dense_retriever.index:
                print("❌ No search index found. Building index first...")
                if not self.indexer.build_index(rebuild=False):
                    print("❌ Failed to build index")
                    return
        except Exception as e:
            print(f"❌ Error with index: {e}")
            return
        
        # Show available models
        available_models = self.llm_manager.get_available_models()
        if available_models:
            print(f"\n🤖 Available models: {len(available_models)}")
            for model in available_models:
                status = "🔥 LOADED" if model['is_loaded'] else "💤 Available"
                print(f"  {status} {model['name']}: {model['description']}")
        
        # Show indexed documents
        stats = self.indexer.get_stats()
        print(f"\n📚 Knowledge Base: {stats['total_files']} files, {stats['total_chunks']} chunks")
        print(f"📊 File types: {stats['files_by_type']}")
        
        print("\n📖 Commands:")
        print("  - Type your message to chat (includes documents + memory)")
        print("  - 'new' - Start new conversation")
        print("  - 'list' - List conversations") 
        print("  - 'load <id>' - Load conversation")
        print("  - 'model <name>' - Switch model")
        print("  - 'rebuild' - Rebuild document index")
        print("  - 'debug' - Toggle debug mode")
        print("  - 'quit' - Exit")
        print()
        
        # Create initial conversation
        conversation_id = str(uuid.uuid4())
        print(f"🆕 Started conversation: {conversation_id[:8]}")
        print("💡 This conversation will access ALL your indexed documents + memory!")
        
        debug_mode = False
        turn_count = 0
        
        while True:
            try:
                user_input = input(f"\n[Turn {turn_count + 1}] 💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                elif user_input.lower() == 'new':
                    conversation_id = str(uuid.uuid4())
                    turn_count = 0
                    print(f"🆕 Started new conversation: {conversation_id[:8]}")
                    continue
                
                elif user_input.lower() == 'list':
                    conversations = self.get_conversations()
                    print(f"\n📋 Conversations ({len(conversations)}):")
                    for conv in conversations[:10]:
                        status = "🔥" if conv['id'] == conversation_id else "💬"
                        print(f"  {status} {conv['id'][:8]}: {conv['title']} ({conv['message_count']} msgs)")
                    continue
                
                elif user_input.lower().startswith('load '):
                    conv_id_prefix = user_input[5:].strip()
                    conversations = self.get_conversations()
                    matching_conv = None
                    for conv in conversations:
                        if conv['id'].startswith(conv_id_prefix):
                            matching_conv = conv
                            break
                    
                    if matching_conv:
                        conversation_id = matching_conv['id']
                        print(f"📂 Loaded conversation: {matching_conv['title']}")
                        
                        # Show recent history
                        history = self.get_conversation_history(conversation_id)
                        if history:
                            print("Recent history:")
                            for msg in history[-4:]:
                                role = "You" if msg['chunk_type'] == 'user_msg' else "Assistant"
                                content = msg['chunk_text'][:80] + "..." if len(msg['chunk_text']) > 80 else msg['chunk_text']
                                print(f"  {role}: {content}")
                    else:
                        print(f"❌ No conversation found starting with '{conv_id_prefix}'")
                    continue
                
                elif user_input.lower().startswith('model '):
                    model_name = user_input[6:].strip()
                    print(f"🔄 Loading model: {model_name}...")
                    if self.llm_manager.set_active_llm(model_name):
                        print(f"✅ Switched to model: {model_name}")
                    else:
                        print(f"❌ Model not available: {model_name}")
                    continue
                
                elif user_input.lower() == 'rebuild':
                    print("🔄 Rebuilding document index...")
                    if self.indexer.build_index(rebuild=True):
                        print("✅ Index rebuilt successfully")
                    else:
                        print("❌ Index rebuild failed")
                    continue
                
                elif user_input.lower() == 'debug':
                    debug_mode = not debug_mode
                    print(f"🔍 Debug mode: {'ON' if debug_mode else 'OFF'}")
                    continue
                
                # Process conversation turn with UNIFIED retrieval
                print("🔍 Searching documents + memory...")
                turn_result = self.process_conversation_turn(conversation_id, user_input)
                
                # Display response
                print(f"\n🤖 Assistant: {turn_result['response']}")
                
                # Show metadata
                cached_indicator = " (cached)" if turn_result['cached'] else ""
                print(f"\n📊 Confidence: {turn_result['confidence']:.2f} | "
                      f"Citations: {len(turn_result['citations'])} | "
                      f"Time: {turn_result['total_time']:.1f}s{cached_indicator}")
                
                # Debug information
                if debug_mode:
                    print(f"\n🔍 Debug Info:")
                    print(f"  📚 Documents: {len(turn_result['doc_results'])}")
                    print(f"  🧠 Memory items: {len(turn_result['memory_results'])}")
                    print(f"  ⚡ Generation: {turn_result['generation_time']:.2f}s")
                    print(f"  💾 Cached: {turn_result['cached']}")
                    
                    if turn_result['citations']:
                        print(f"  🔗 Citations: {', '.join(turn_result['citations'])}")
                    
                    # Show top document results
                    if turn_result['doc_results']:
                        print("  📄 Top documents:")
                        for i, doc in enumerate(turn_result['doc_results'][:3], 1):
                            filename = doc.get('path', 'Unknown').split('/')[-1]
                            score = doc.get('score', 0)
                            print(f"    {i}. {filename} (score: {score:.3f})")
                
                turn_count += 1
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                if debug_mode:
                    import traceback
                    traceback.print_exc()
        
        # Cleanup
        self.cleanup()
    
    def interactive_mode(self):
        """Enhanced interactive mode that also includes documents."""
        print("🔍 Enhanced Interactive Mode")
        print("=" * 50)
        print("✨ Now includes indexed documents in every response!")
        
        # Check system
        try:
            if not self.retriever.dense_retriever.index:
                print("❌ No search index found. Please build index first.")
                return
        except Exception as e:
            print(f"❌ Error: {e}")
            return
        
        # Show system info
        stats = self.indexer.get_stats()
        print(f"📚 Knowledge Base: {stats['total_files']} files, {stats['total_chunks']} chunks")
        
        print("\nCommands:")
        print("  - Type your question (searches documents)")
        print("  - 'rag: <question>' - Get AI answer with document context")
        print("  - 'model: <name>' - Switch model")
        print("  - 'quit' - Exit")
        print()
        
        while True:
            try:
                user_input = input("🔍 Query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                elif user_input.startswith('model:'):
                    model_name = user_input[6:].strip()
                    if self.llm_manager.set_active_llm(model_name):
                        print(f"✅ Switched to model: {model_name}")
                    else:
                        print(f"❌ Model not available: {model_name}")
                    continue
                
                elif user_input.startswith('rag:'):
                    query = user_input[4:].strip()
                    if query:
                        # Use conversation mode for single query
                        temp_conv_id = str(uuid.uuid4())
                        print("🤖 Generating answer with document context...")
                        
                        turn_result = self.process_conversation_turn(temp_conv_id, query)
                        
                        print(f"\n🤖 Answer: {turn_result['response']}")
                        print(f"📊 Confidence: {turn_result['confidence']:.2f} | Citations: {len(turn_result['citations'])}")
                        
                        if turn_result['citations']:
                            print(f"🔗 Sources: {', '.join(turn_result['citations'])}")
                    continue
                
                else:
                    # Regular search
                    print("🔍 Searching documents...")
                    results = self.retriever.search(user_input, top_k=5)
                    
                    if results:
                        if hasattr(results[0], 'to_dict'):
                            results = [r.to_dict() for r in results]
                        
                        print(f"\n📄 Found {len(results)} results:")
                        for i, result in enumerate(results, 1):
                            filename = result.get('path', 'Unknown').split('/')[-1]
                            score = result.get('score', 0)
                            print(f"  {i}. {filename} (score: {score:.3f})")
                    else:
                        print("No results found.")
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        self.indexer.close()
        self.retriever.close()
        self.llm_manager.cleanup()
        self.cache.close()


def main():
    """Main entry point with unified conversational RAG."""
    parser = argparse.ArgumentParser(
        description="Unified Conversational RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_unified.py --build-index              # Build search index
  python main_unified.py --conversation             # Conversational mode (RECOMMENDED)
  python main_unified.py --interactive              # Enhanced interactive mode
  python main_unified.py --models                   # List available models
        """
    )
    
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--build-index', action='store_true',
                       help='Build the search index')
    parser.add_argument('--rebuild-index', action='store_true',
                       help='Rebuild the search index from scratch')
    parser.add_argument('--conversation', action='store_true',
                       help='Conversational mode with memory (RECOMMENDED)')
    parser.add_argument('--interactive', action='store_true',
                       help='Enhanced interactive mode')
    parser.add_argument('--models', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    # Initialize system
    try:
        system = UnifiedConversationalRAG(args.config)
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return 1
    
    try:
        if args.build_index or args.rebuild_index:
            success = system.indexer.build_index(rebuild=args.rebuild_index)
            return 0 if success else 1
        
        elif args.models:
            available_models = system.model_api.list_available_models()
            print("🤖 Available Local Models:")
            print("=" * 50)
            for model in available_models:
                status = "🔥 LOADED" if model['is_loaded'] else "💤 Available"
                print(f"{status} {model['name']} ({model['size_gb']:.1f}GB)")
                print(f"    {model['description']}")
            return 0
        
        elif args.conversation:
            system.conversation_mode()
            return 0
        
        elif args.interactive:
            system.interactive_mode()
            return 0
        
        else:
            print("🚀 Unified Conversational RAG System")
            print("No mode specified. Choose an option:")
            print("  --conversation  (Recommended: Chat with memory + documents)")
            print("  --interactive   (Simple Q&A with documents)")
            print("  --build-index   (Build search index)")
            print("  --models        (List available models)")
            return 0
    
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        system.cleanup()


if __name__ == "__main__":
    sys.exit(main())