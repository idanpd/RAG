"""
Enhanced Streamlit UI for Unified Conversational RAG
Includes file upload, automatic indexing, and conversation with memory.
"""

import streamlit as st
import tempfile
import uuid
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import system components
from utils import ConfigManager, setup_logger
from indexer import SemanticIndexer
from retriever import SemanticRetriever
from rag import LocalLLMManager
from model_api import LocalModelAPI

# Page configuration
st.set_page_config(
    page_title="Conversational RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling with white background
st.markdown("""
<style>
/* Force white background and black text throughout */
.stApp {
    background-color: white !important;
    color: black !important;
}

.main .block-container {
    background-color: white !important;
    color: black !important;
}

.stSidebar .block-container {
    background-color: #f8f9fa !important;
    color: black !important;
}

/* Chat message styling */
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    border: 1px solid #ddd;
    background-color: white !important;
    color: black !important;
}

.user-message {
    background-color: #e3f2fd !important;
    margin-left: 10%;
    border-left: 4px solid #2196F3;
}

.assistant-message {
    background-color: #f8f9fa !important;
    margin-right: 10%;
    border-left: 4px solid #4CAF50;
}

.message-meta {
    font-size: 0.8rem;
    color: #666 !important;
    margin-top: 0.5rem;
    font-style: italic;
}

/* File upload area */
.upload-area {
    border: 2px dashed #007bff;
    border-radius: 0.5rem;
    padding: 2rem;
    text-align: center;
    background-color: #f8f9fa !important;
    color: black !important;
    margin: 1rem 0;
}

/* Status indicators */
.status-success {
    color: #28a745 !important;
    font-weight: bold;
}

.status-error {
    color: #dc3545 !important;
    font-weight: bold;
}

.status-warning {
    color: #ffc107 !important;
    font-weight: bold;
}

/* Button improvements */
.stButton > button {
    background-color: #007bff !important;
    color: white !important;
    border: none !important;
    border-radius: 0.25rem;
    font-weight: 500;
}

.stButton > button:hover {
    background-color: #0056b3 !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Input styling */
.stTextInput > div > div > input,
.stChatInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ddd !important;
    border-radius: 0.25rem;
}

/* Metrics */
.metric-container {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ddd;
    border-radius: 0.25rem;
    padding: 0.5rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


class UnifiedRAGSystem:
    """Unified RAG system that handles both documents and conversations."""
    
    def __init__(self):
        self.config = ConfigManager()
        self.logger = setup_logger(self.config.get('LOG_LEVEL', 'INFO'))
        
        # Initialize components
        self.indexer = SemanticIndexer(self.config)
        self.retriever = SemanticRetriever(self.config)
        self.llm_manager = LocalLLMManager(self.config)
        self.model_api = LocalModelAPI(self.config)
        
        # Conversation state
        self.conversations = {}
        
    def process_uploaded_file(self, uploaded_file, temp_dir: Path) -> bool:
        """Process an uploaded file and add to index."""
        try:
            # Save uploaded file temporarily
            temp_file_path = temp_dir / uploaded_file.name
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the file using existing indexer
            file_id, chunks = self.indexer.process_file(temp_file_path)
            
            if file_id > 0 and chunks:
                # Generate embeddings for new chunks
                chunk_texts = [chunk.chunk_text for chunk in chunks]
                embeddings = self.indexer.embedding_manager.create_embeddings(chunk_texts)
                
                # Update chunks with embedding IDs
                for i, chunk in enumerate(chunks):
                    chunk.emb_id = self.get_next_embedding_id() + i
                    chunk.chunk_type = 'doc'  # Mark as document chunk
                
                # Insert chunks into database
                chunk_ids = self.indexer.db_manager.insert_chunks(chunks)
                
                # Update FAISS index
                self.update_faiss_index(embeddings)
                
                self.logger.info(f"Successfully processed uploaded file: {uploaded_file.name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to process uploaded file: {e}")
            return False
    
    def get_next_embedding_id(self) -> int:
        """Get the next available embedding ID."""
        cursor = self.indexer.db_manager.conn.cursor()
        cursor.execute("SELECT MAX(emb_id) FROM chunks WHERE emb_id IS NOT NULL")
        result = cursor.fetchone()
        max_id = result[0] if result and result[0] is not None else -1
        return max_id + 1
    
    def update_faiss_index(self, new_embeddings):
        """Update FAISS index with new embeddings."""
        # For simplicity, we'll rebuild the index
        # In production, you'd use incremental updates
        if hasattr(self.retriever.dense_retriever, 'index') and self.retriever.dense_retriever.index:
            # Add new embeddings to existing index
            self.retriever.dense_retriever.index.add(new_embeddings.astype('float32'))
    
    def add_conversation_message(self, conversation_id: str, message_type: str, 
                               content: str, prev_id: Optional[int] = None,
                               confidence: float = 0.5, citations: List[str] = None) -> int:
        """Add a message to conversation using chunks table."""
        
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
            'streamlit_ui',
            confidence,
            citations_json,
            False
        ))
        
        message_id = cursor.lastrowid
        self.indexer.db_manager.conn.commit()
        
        return message_id
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict]:
        """Get messages from a conversation."""
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
            # Parse citations
            if message['citations']:
                try:
                    message['citations'] = json.loads(message['citations'])
                except:
                    message['citations'] = []
            else:
                message['citations'] = []
            messages.append(message)
        
        return messages
    
    def search_with_memory(self, query: str, conversation_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Search both documents and conversation memory."""
        
        # Search documents (existing functionality)
        doc_results = self.retriever.search(query, top_k=8)
        if doc_results and hasattr(doc_results[0], 'to_dict'):
            doc_results = [result.to_dict() for result in doc_results]
        
        # Search conversation memory
        memory_results = []
        if conversation_id:
            cursor = self.indexer.db_manager.conn.cursor()
            cursor.execute("""
                SELECT * FROM chunks 
                WHERE conversation_id = ? 
                AND chunk_type IN ('assistant_msg', 'user_msg', 'summary')
                AND archived = FALSE
                AND (chunk_type != 'assistant_msg' OR confidence >= 0.7 OR citations IS NOT NULL)
                ORDER BY timestamp DESC
                LIMIT 6
            """, (conversation_id,))
            
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
    
    def assemble_conversation_prompt(self, query: str, conversation_id: str,
                                   doc_results: List[Dict], memory_results: List[Dict]) -> str:
        """Assemble prompt with both documents and conversation context."""
        
        # System prompt
        system_prompt = """You are a helpful AI assistant with access to documents and conversation history.

Instructions:
- Answer questions using the provided context from documents and conversation history
- Cite sources using [doc:filename] for documents and [memory:turn] for conversation context
- If information is not in the provided context, say so clearly
- Be concise but comprehensive
- Maintain conversation continuity using the chat history

"""
        
        # Recent conversation context
        conversation_context = ""
        if memory_results:
            conversation_context = "Conversation history:\n"
            for msg in memory_results[-4:]:  # Last 4 messages
                if msg['chunk_type'] == 'user_msg':
                    conversation_context += f"User: {msg['chunk_text']}\n"
                elif msg['chunk_type'] == 'assistant_msg':
                    conversation_context += f"Assistant: {msg['chunk_text']}\n"
                elif msg['chunk_type'] == 'summary':
                    conversation_context += f"Summary: {msg['chunk_text']}\n"
            conversation_context += "\n"
        
        # Document context
        docs_context = ""
        if doc_results:
            docs_context = "Relevant documents:\n"
            for result in doc_results[:5]:  # Top 5 documents
                filename = result.get('path', 'Unknown').split('/')[-1]
                content = result.get('text', '')[:500]  # Limit content
                docs_context += f"[doc:{filename}] {content}\n"
            docs_context += "\n"
        
        # Assemble final prompt
        prompt = system_prompt + conversation_context + docs_context + f"User: {query}\nAssistant: "
        
        return prompt
    
    def analyze_response(self, response: str, doc_results: List[Dict]) -> Tuple[List[str], float]:
        """Analyze response for citations and confidence."""
        
        citations = []
        confidence = 0.5
        
        # Extract citations
        import re
        doc_citations = re.findall(r'\[doc:([^\]]+)\]', response)
        memory_citations = re.findall(r'\[memory:([^\]]+)\]', response)
        
        citations.extend([f"doc:{cite}" for cite in doc_citations])
        citations.extend([f"memory:{cite}" for cite in memory_citations])
        
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


@st.cache_resource
def get_unified_system():
    """Get the unified RAG system."""
    try:
        return UnifiedRAGSystem()
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None


def file_upload_section():
    """File upload section in sidebar."""
    st.subheader("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload files to add to knowledge base",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'md', 'csv', 'xlsx', 'png', 'jpg', 'jpeg'],
        help="Upload documents to automatically add them to the conversation context"
    )
    
    if uploaded_files:
        system = get_unified_system()
        if system:
            if st.button("🚀 Process Uploaded Files", use_container_width=True):
                with st.spinner("Processing uploaded files..."):
                    success_count = 0
                    
                    # Create temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        
                        for uploaded_file in uploaded_files:
                            if system.process_uploaded_file(uploaded_file, temp_path):
                                success_count += 1
                                st.session_state.uploaded_files.append(uploaded_file.name)
                    
                    if success_count > 0:
                        st.success(f"✅ Successfully processed {success_count}/{len(uploaded_files)} files")
                        st.session_state.index_needs_rebuild = False
                        st.rerun()
                    else:
                        st.error("❌ Failed to process files")
    
    # Show uploaded files
    if st.session_state.uploaded_files:
        st.write("**Recently uploaded:**")
        for filename in st.session_state.uploaded_files[-5:]:
            st.write(f"📄 {filename}")


def sidebar():
    """Enhanced sidebar with file upload and model management."""
    with st.sidebar:
        st.title("🤖 Conversational RAG")
        
        # System status
        system = get_unified_system()
        if system:
            st.markdown('<p class="status-success">✅ System Ready</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ System Error</p>', unsafe_allow_html=True)
            return
        
        # File upload section
        file_upload_section()
        
        st.divider()
        
        # Model management
        st.subheader("🤖 Model Management")
        
        available_models = system.model_api.list_available_models()
        if available_models:
            model_options = [f"{m['name']} ({m['size_gb']:.1f}GB)" for m in available_models]
            selected_idx = st.selectbox(
                "Select Model",
                range(len(model_options)),
                format_func=lambda x: model_options[x],
                help="Choose the LLM for responses"
            )
            
            selected_model = available_models[selected_idx]
            
            # Model status and load button
            if selected_model['is_loaded']:
                st.markdown(f'<p class="status-success">🔥 {selected_model["description"]}</p>', 
                           unsafe_allow_html=True)
                
                if st.button("🔄 Unload Model", use_container_width=True):
                    result = system.model_api.unload_model(selected_model['name'])
                    if result['success']:
                        st.success("Model unloaded")
                        st.rerun()
            else:
                st.markdown(f'<p class="status-warning">💤 {selected_model["description"]}</p>', 
                           unsafe_allow_html=True)
                
                if st.button(f"⚡ Load {selected_model['name']}", use_container_width=True):
                    with st.spinner(f"Loading {selected_model['name']}..."):
                        result = system.model_api.load_model(selected_model['name'])
                        if result['success']:
                            st.success(f"Loaded in {result['load_time']:.1f}s")
                            st.rerun()
                        else:
                            st.error("Failed to load model")
        
        st.divider()
        
        # Conversation management
        st.subheader("💬 Conversations")
        
        # New conversation button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            new_conv_id = str(uuid.uuid4())
            st.session_state.current_conversation_id = new_conv_id
            st.session_state.chat_history = []
            st.session_state.conversations[new_conv_id] = {
                'title': f"Chat {datetime.now().strftime('%H:%M')}",
                'created_at': datetime.now().isoformat(),
                'message_count': 0
            }
            st.rerun()
        
        # List conversations
        if st.session_state.conversations:
            st.write("**Recent conversations:**")
            for conv_id, conv_info in list(st.session_state.conversations.items())[-5:]:
                is_active = conv_id == st.session_state.current_conversation_id
                button_type = "primary" if is_active else "secondary"
                
                if st.button(
                    f"💬 {conv_info['title']} ({conv_info['message_count']})",
                    key=f"conv_{conv_id}",
                    use_container_width=True,
                    type=button_type
                ):
                    st.session_state.current_conversation_id = conv_id
                    # Load conversation history
                    messages = system.get_conversation_messages(conv_id)
                    st.session_state.chat_history = [
                        {
                            'type': 'user' if msg['chunk_type'] == 'user_msg' else 'assistant',
                            'content': msg['chunk_text'],
                            'timestamp': msg['timestamp'],
                            'confidence': msg.get('confidence'),
                            'citations': msg.get('citations', [])
                        }
                        for msg in messages
                    ]
                    st.rerun()
        
        st.divider()
        
        # Debug and settings
        st.session_state.show_debug = st.checkbox("🔍 Show Debug Info")
        
        # System stats
        if st.button("📊 System Stats"):
            stats = system.model_api.get_model_stats()
            st.json(stats)


def main_chat_interface():
    """Main chat interface."""
    system = get_unified_system()
    
    if not system:
        st.error("❌ System not initialized. Please check the sidebar.")
        return
    
    # Header
    if st.session_state.current_conversation_id:
        conv_info = st.session_state.conversations.get(
            st.session_state.current_conversation_id, 
            {'title': 'Current Chat', 'message_count': 0}
        )
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.title(f"💬 {conv_info['title']}")
        with col2:
            st.metric("Messages", conv_info['message_count'])
        with col3:
            if st.session_state.uploaded_files:
                st.metric("Uploaded", len(st.session_state.uploaded_files))
    else:
        st.title("💬 Conversational RAG")
        st.info("👈 Create a new conversation in the sidebar to start chatting")
        return
    
    # Chat history display
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            message_class = "user-message" if message['type'] == 'user' else "assistant-message"
            
            # Format timestamp
            timestamp = message.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M:%S')
                except:
                    time_str = timestamp[:19]
            else:
                time_str = ''
            
            # Build metadata string
            meta_parts = [time_str]
            if message.get('confidence') is not None:
                meta_parts.append(f"Confidence: {message['confidence']:.2f}")
            if message.get('citations'):
                meta_parts.append(f"Citations: {len(message['citations'])}")
            
            meta_str = " | ".join(filter(None, meta_parts))
            
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <div><strong>{'You' if message['type'] == 'user' else '🤖 Assistant'}:</strong></div>
                <div style="margin: 0.5rem 0;">{message['content']}</div>
                <div class="message-meta">{meta_str}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input and st.session_state.current_conversation_id:
        # Add user message to display
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'confidence': None,
            'citations': []
        })
        
        # Update conversation info
        conv_id = st.session_state.current_conversation_id
        if conv_id in st.session_state.conversations:
            st.session_state.conversations[conv_id]['message_count'] += 1
        
        # Process with unified RAG
        with st.spinner("🤔 Thinking..."):
            try:
                start_time = time.time()
                
                # Get last message ID for threading
                messages = system.get_conversation_messages(conv_id)
                prev_id = messages[-1]['id'] if messages else None
                
                # Add user message to database
                user_msg_id = system.add_conversation_message(
                    conv_id, 'user_msg', user_input, prev_id, 1.0
                )
                
                # Search both documents and memory
                doc_results, memory_results = system.search_with_memory(user_input, conv_id)
                
                # Assemble prompt
                prompt = system.assemble_conversation_prompt(
                    user_input, conv_id, doc_results, memory_results
                )
                
                # Generate response
                response = system.llm_manager.generate(prompt)
                
                # Analyze response
                citations, confidence = system.analyze_response(response, doc_results)
                
                # Add assistant message to database
                assistant_msg_id = system.add_conversation_message(
                    conv_id, 'assistant_msg', response, user_msg_id, confidence, citations
                )
                
                total_time = time.time() - start_time
                
                # Add assistant response to display
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': response,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'confidence': confidence,
                    'citations': citations
                })
                
                # Update conversation info
                st.session_state.conversations[conv_id]['message_count'] += 1
                
                # Show debug info if enabled
                if st.session_state.show_debug:
                    with st.expander("🔍 Debug Information", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**📚 Document Results:**")
                            for i, doc in enumerate(doc_results[:3]):
                                filename = doc.get('path', 'Unknown').split('/')[-1]
                                score = doc.get('score', 0)
                                st.write(f"{i+1}. {filename} (score: {score:.3f})")
                            
                            st.write("**🧠 Memory Results:**")
                            for i, mem in enumerate(memory_results[:3]):
                                content = mem['chunk_text'][:50] + "..."
                                msg_type = mem['chunk_type']
                                st.write(f"{i+1}. {msg_type}: {content}")
                        
                        with col2:
                            st.write("**⚙️ Generation Info:**")
                            st.json({
                                'total_time': f"{total_time:.2f}s",
                                'confidence': confidence,
                                'citations_count': len(citations),
                                'doc_results': len(doc_results),
                                'memory_results': len(memory_results),
                                'prompt_length': len(prompt),
                                'response_length': len(response)
                            })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing message: {e}")
                if st.session_state.show_debug:
                    st.exception(e)


def main():
    """Main Streamlit application."""
    # Initialize session state
    if "system" not in st.session_state:
        st.session_state.system = get_unified_system()
    
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    # Layout
    sidebar()
    main_chat_interface()


if __name__ == "__main__":
    main()