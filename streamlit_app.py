"""
Enhanced Streamlit UI for Conversational RAG with Document + Memory Integration
Fixed to ensure indexed documents are ALWAYS included in conversation answers.
"""

import streamlit as st
import tempfile
import uuid
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Conversational RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with white background and black text
st.markdown("""
<style>
/* Force white background and black text */
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

/* Button styling */
.stButton > button {
    background-color: #007bff !important;
    color: white !important;
    border: none !important;
    border-radius: 0.25rem;
}

.stButton > button:hover {
    background-color: #0056b3 !important;
}

/* Input styling */
.stTextInput > div > div > input,
.stChatInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ddd !important;
}

/* File uploader */
.stFileUploader {
    background-color: white !important;
    color: black !important;
    border: 2px dashed #007bff !important;
    border-radius: 0.5rem;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the enhanced RAG system."""
    try:
        from utils import ConfigManager
        from indexer import SemanticIndexer
        from retriever import SemanticRetriever
        from rag import RAGSystem, LocalLLMManager
        from model_api import LocalModelAPI
        
        config = ConfigManager()
        
        # Initialize components
        indexer = SemanticIndexer(config)
        retriever = SemanticRetriever(config)
        rag_system = RAGSystem(retriever, config)
        model_api = LocalModelAPI(config)
        
        # Ensure index is built from DATA_PATH if it doesn't exist
        try:
            stats = indexer.get_stats()
            if stats['total_chunks'] == 0:
                # No data indexed yet, build from DATA_PATH
                data_path = config.get('DATA_PATH', './data')
                if Path(data_path).exists():
                    indexer.index_directory(data_path)
        except Exception as e:
            # Index might not exist, try to build it
            data_path = config.get('DATA_PATH', './data')
            if Path(data_path).exists():
                try:
                    indexer.index_directory(data_path)
                except Exception:
                    pass  # Continue even if indexing fails
        
        return {
            'indexer': indexer,
            'retriever': retriever,
            'rag_system': rag_system,
            'model_api': model_api,
            'config': config
        }, None
    except Exception as e:
        return None, str(e)


def process_uploaded_file(uploaded_file, system):
    """Process uploaded file and add to index."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded file
            file_path = temp_path / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process file using existing indexer
            file_id, chunks = system['indexer'].process_file(file_path)
            
            if file_id > 0 and chunks:
                # Generate embeddings
                chunk_texts = [chunk.chunk_text for chunk in chunks]
                embeddings = system['indexer'].embedding_manager.create_embeddings(chunk_texts)
                
                # Get next embedding ID
                cursor = system['indexer'].db_manager.conn.cursor()
                cursor.execute("SELECT MAX(emb_id) FROM chunks WHERE emb_id IS NOT NULL")
                result = cursor.fetchone()
                max_id = result[0] if result and result[0] is not None else -1
                
                # Update chunks with embedding IDs
                for i, chunk in enumerate(chunks):
                    chunk.emb_id = max_id + 1 + i
                    chunk.chunk_type = 'doc'  # Ensure it's marked as document
                
                # Insert chunks
                system['indexer'].db_manager.insert_chunks(chunks)
                
                # Update FAISS index with consistency check
                if hasattr(system['retriever'].dense_retriever, 'index') and system['retriever'].dense_retriever.index:
                    # Check FAISS index size matches database
                    cursor.execute("SELECT COUNT(*) FROM chunks WHERE emb_id IS NOT NULL")
                    db_count = cursor.fetchone()[0]
                    faiss_count = system['retriever'].dense_retriever.index.ntotal
                    
                    # If mismatch, rebuild FAISS index
                    if abs(db_count - faiss_count) > len(chunks):
                        # Significant mismatch, rebuild FAISS
                        cursor.execute("SELECT emb_id FROM chunks WHERE emb_id IS NOT NULL ORDER BY emb_id")
                        emb_ids = [row[0] for row in cursor.fetchall()]
                        if emb_ids:
                            # Get all embeddings and rebuild
                            all_embeddings = []
                            for emb_id in emb_ids:
                                cursor.execute("SELECT chunk_text FROM chunks WHERE emb_id = ?", (emb_id,))
                                text_row = cursor.fetchone()
                                if text_row:
                                    emb = system['indexer'].embedding_manager.create_embeddings([text_row[0]])
                                    all_embeddings.append(emb[0])
                            
                            if all_embeddings:
                                import numpy as np
                                all_embeddings = np.array(all_embeddings).astype('float32')
                                system['retriever'].dense_retriever._build_index(all_embeddings)
                    
                    # Add new embeddings
                    system['retriever'].dense_retriever.index.add(embeddings.astype('float32'))
                
                return True
        
        return False
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return False


def sidebar():
    """Enhanced sidebar with file upload and model management."""
    with st.sidebar:
        st.title("🤖 Conversational RAG")
        
        # System status
        system, error = st.session_state.get('system'), st.session_state.get('init_error')
        if system:
            st.markdown('<span style="color: #28a745; font-weight: bold;">✅ System Ready</span>', 
                       unsafe_allow_html=True)
            
            # Show knowledge base stats
            try:
                stats = system['indexer'].get_stats()
                st.info(f"📚 Knowledge Base: {stats['total_files']} files, {stats['total_chunks']} chunks")
            except:
                pass
        else:
            st.markdown('<span style="color: #dc3545; font-weight: bold;">❌ System Error</span>', 
                       unsafe_allow_html=True)
            if error:
                st.error(error)
            return
        
        # Conversation management - moved above file upload
        st.subheader("💬 New Chat")
        
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
        
        st.divider()
        
        # File upload section
        st.subheader("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Add files to knowledge base",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'md', 'csv', 'xlsx', 'png', 'jpg', 'jpeg'],
            help="Files will be automatically indexed and available in conversations"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Files", use_container_width=True):
                with st.spinner("Processing files..."):
                    success_count = 0
                    for uploaded_file in uploaded_files:
                        if process_uploaded_file(uploaded_file, system):
                            success_count += 1
                            if 'uploaded_files' not in st.session_state:
                                st.session_state.uploaded_files = []
                            st.session_state.uploaded_files.append(uploaded_file.name)
                    
                    if success_count > 0:
                        st.success(f"✅ Processed {success_count}/{len(uploaded_files)} files")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process files")
        
        # Show recently uploaded files
        if st.session_state.get('uploaded_files'):
            st.write("**Recently uploaded:**")
            for filename in st.session_state.uploaded_files[-5:]:
                st.write(f"📄 {filename}")
        
        st.divider()
        
        # Model management
        st.subheader("🤖 Model Management")
        
        try:
            available_models = system['model_api'].list_available_models()
            if available_models:
                model_names = [model['name'] for model in available_models]
                
                # Model selection
                selected_model_name = st.selectbox(
                    "Select Model",
                    model_names,
                    help="Choose the LLM for responses"
                )
                
                selected_model = next(m for m in available_models if m['name'] == selected_model_name)
                
                # Model status
                if selected_model['is_loaded']:
                    st.success(f"🔥 {selected_model['description']}")
                    
                    if st.button("🔄 Unload Model"):
                        result = system['model_api'].unload_model(selected_model_name)
                        if result['success']:
                            st.success("Model unloaded")
                            st.rerun()
                else:
                    st.info(f"💤 {selected_model['description']}")
                    
                    if st.button(f"⚡ Load {selected_model_name}"):
                        with st.spinner(f"Loading {selected_model_name}..."):
                            result = system['model_api'].load_model(selected_model_name)
                            if result['success']:
                                st.success(f"Loaded in {result['load_time']:.1f}s")
                                st.rerun()
                            else:
                                st.error("Failed to load model")
        except Exception as e:
            st.error(f"Model management error: {e}")
        
        st.divider()
        
        # Debug toggle
        st.session_state.show_debug = st.checkbox("🔍 Show Debug Info")


def main_chat_interface():
    """Main chat interface with dual retrieval (documents + memory)."""
    system = st.session_state.get('system')
    rag_system = st.session_state.get('rag_system')
    
    if not system or not rag_system:
        st.error("❌ System not initialized. Please check the sidebar.")
        return
    
    # Header
    if st.session_state.current_conversation_id:
        conv_info = st.session_state.conversations.get(
            st.session_state.current_conversation_id,
            {'title': 'Current Chat', 'message_count': 0}
        )
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.title(f"💬 {conv_info['title']}")
        with col2:
            st.metric("Messages", conv_info['message_count'])
        with col3:
            uploaded_count = len(st.session_state.get('uploaded_files', []))
            st.metric("Uploaded", uploaded_count)
        with col4:
            try:
                stats = system['indexer'].get_stats()
                st.metric("Total Docs", stats['total_files'])
            except:
                st.metric("Total Docs", "?")
    else:
        st.title("💬 Conversational RAG")
        st.info("👈 Click 'New Chat' in the sidebar to start a conversation")
        st.info("💡 Conversations will search BOTH your indexed documents AND chat history!")
        return
    
    # Chat history display
    for message in st.session_state.chat_history:
        message_class = "user-message" if message['type'] == 'user' else "assistant-message"
        
        # Format metadata
        meta_parts = []
        if message.get('timestamp'):
            try:
                dt = datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00'))
                meta_parts.append(dt.strftime('%H:%M:%S'))
            except:
                meta_parts.append(message['timestamp'][:19])
        
        if message.get('confidence') is not None:
            meta_parts.append(f"Confidence: {message['confidence']:.2f}")
        
        if message.get('citations'):
            meta_parts.append(f"Citations: {len(message['citations'])}")
        
        meta_str = " | ".join(meta_parts)
        
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <div><strong>{'You' if message['type'] == 'user' else '🤖 Assistant'}:</strong></div>
            <div style="margin: 0.5rem 0;">{message['content']}</div>
            <div class="message-meta">{meta_str}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your message... (searches documents + memory)")
    
    if user_input and st.session_state.current_conversation_id:
        # Add user message to display
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'confidence': None,
            'citations': []
        })
        
        # Update conversation count
        conv_id = st.session_state.current_conversation_id
        if conv_id in st.session_state.conversations:
            st.session_state.conversations[conv_id]['message_count'] += 1
        
        # Process with enhanced RAG system (documents + memory)
        with st.spinner("🔍 Searching documents + memory..."):
            try:
                start_time = time.time()
                
                # Use conversational query method (searches BOTH docs and memory)
                result = rag_system.answer_conversational_query(
                    query=user_input,
                    conversation_id=conv_id
                )
                
                processing_time = time.time() - start_time
                
                # Add assistant response to display
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': result['answer'],
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'confidence': result['confidence'],
                    'citations': result['citations']
                })
                
                # Update conversation count
                st.session_state.conversations[conv_id]['message_count'] += 1
                
                # Show debug info if enabled
                if st.session_state.show_debug:
                    with st.expander("🔍 Debug Information", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**📚 Document Results:**")
                            for i, doc in enumerate(result['doc_sources'][:5]):
                                filename = doc['path'].split('/')[-1] if doc['path'] else 'Unknown'
                                st.write(f"{i+1}. {filename} (score: {doc['score']:.3f})")
                            
                            st.write("**🧠 Memory Results:**")
                            for i, mem in enumerate(result['memory_sources'][:3]):
                                content = mem['content'][:50] + "..." if len(mem['content']) > 50 else mem['content']
                                st.write(f"{i+1}. {mem['type']}: {content}")
                        
                        with col2:
                            st.write("**⚙️ Generation Info:**")
                            debug_info = {
                                'processing_time': f"{processing_time:.2f}s",
                                'confidence': result['confidence'],
                                'citations_count': len(result['citations']),
                                'doc_sources': len(result['doc_sources']),
                                'memory_sources': len(result['memory_sources']),
                                'sliding_window': len(result['sliding_window']),
                                'model_used': result['llm_used']
                            }
                            st.json(debug_info)
                            
                            if result['citations']:
                                st.write("**🔗 Citations:**")
                                for citation in result['citations']:
                                    st.write(f"- {citation}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing message: {e}")
                if st.session_state.show_debug:
                    st.exception(e)


def main():
    """Main Streamlit application."""
    # Initialize session state
    if "system" not in st.session_state:
        system, error = initialize_system()
        st.session_state.system = system
        st.session_state.init_error = error
        
        # Initialize RAG system
        if system:
            try:
                st.session_state.rag_system = system['rag_system']
            except Exception as e:
                st.session_state.rag_system = None
    
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
