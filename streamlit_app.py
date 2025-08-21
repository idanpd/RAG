"""
Enhanced Streamlit UI for Conversational RAG with Memory
Industry-grade chat interface with conversation management.
"""

import streamlit as st
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# Import our conversational RAG system
from conversational_rag import ConversationalRAG
from conversation_memory import ConversationMessage
from utils import ConfigManager
from model_api import LocalModelAPI

# Page configuration
st.set_page_config(
    page_title="Conversational RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat UI with white background and black text
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

/* Chat message styling */
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    background-color: white !important;
    color: black !important;
    border: 1px solid #ddd;
}

.user-message {
    background-color: #e3f2fd !important;
    margin-left: 15%;
    color: black !important;
}

.assistant-message {
    background-color: #f8f9fa !important;
    margin-right: 15%;
    color: black !important;
}

.message-meta {
    font-size: 0.8rem;
    color: #666 !important;
    margin-top: 0.5rem;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #f8f9fa !important;
}

/* Input styling */
.stTextInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ddd !important;
}

.stChatInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ddd !important;
}

/* Button styling */
.stButton > button {
    background-color: #007bff !important;
    color: white !important;
    border: none !important;
}

.stButton > button:hover {
    background-color: #0056b3 !important;
}

/* File uploader styling */
.stFileUploader {
    background-color: white !important;
    color: black !important;
    border: 2px dashed #007bff !important;
    border-radius: 0.5rem;
    padding: 1rem;
}

/* Metrics styling */
.metric-container {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ddd;
    border-radius: 0.25rem;
    padding: 0.5rem;
}

/* Debug info styling */
.stExpander {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the conversational RAG system."""
    try:
        from utils import ConfigManager
        from indexer import SemanticIndexer
        from retriever import SemanticRetriever
        from rag import LocalLLMManager
        from model_api import LocalModelAPI
        
        config = ConfigManager()
        
        # Initialize core components
        indexer = SemanticIndexer(config)
        retriever = SemanticRetriever(config)
        llm_manager = LocalLLMManager(config)
        model_api = LocalModelAPI(config)
        
        return {
            'indexer': indexer,
            'retriever': retriever, 
            'llm_manager': llm_manager,
            'model_api': model_api,
            'config': config
        }, None
    except Exception as e:
        return None, str(e)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "system" not in st.session_state:
        system, error = initialize_system()
        st.session_state.system = system
        st.session_state.init_error = error
    
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
    
    if "index_needs_rebuild" not in st.session_state:
        st.session_state.index_needs_rebuild = False


def load_conversations():
    """Load conversation list."""
    if st.session_state.rag_system:
        try:
            conversations = st.session_state.rag_system.get_conversations()
            st.session_state.conversations = conversations
        except Exception as e:
            st.error(f"Failed to load conversations: {e}")


def load_conversation_history(conversation_id: str):
    """Load history for a specific conversation."""
    if st.session_state.rag_system:
        try:
            messages = st.session_state.rag_system.get_conversation_history(conversation_id)
            
            # Convert to chat format
            chat_history = []
            for msg in messages:
                if msg.type in ['user_msg', 'assistant_msg']:
                    chat_history.append({
                        'type': 'user' if msg.type == 'user_msg' else 'assistant',
                        'content': msg.content,
                        'timestamp': msg.timestamp,
                        'confidence': msg.confidence if msg.type == 'assistant_msg' else None,
                        'citations': msg.citations if msg.type == 'assistant_msg' else None,
                        'id': msg.id
                    })
            
            st.session_state.chat_history = chat_history
            
        except Exception as e:
            st.error(f"Failed to load conversation history: {e}")


def sidebar():
    """Render sidebar with conversation management."""
    with st.sidebar:
        st.title("🤖 Conversational RAG")
        
        # Model selection
        if st.session_state.model_api:
            available_models = st.session_state.model_api.list_available_models()
            if available_models:
                model_names = [model['name'] for model in available_models]
                current_model = st.selectbox(
                    "Select Model",
                    model_names,
                    help="Choose the LLM for responses"
                )
                
                # Show model info
                selected_model = next(m for m in available_models if m['name'] == current_model)
                if selected_model['is_loaded']:
                    st.success(f"✅ {selected_model['description']}")
                else:
                    if st.button(f"Load {current_model}"):
                        with st.spinner(f"Loading {current_model}..."):
                            result = st.session_state.model_api.load_model(current_model)
                            if result['success']:
                                st.success(f"Loaded in {result['load_time']:.1f}s")
                                st.rerun()
                            else:
                                st.error("Failed to load model")
        
        st.divider()
        
        # Conversation management
        st.subheader("Conversations")
        
        # New conversation button
        if st.button("➕ New Conversation", use_container_width=True):
            if st.session_state.rag_system:
                new_conv_id = st.session_state.rag_system.create_conversation()
                st.session_state.current_conversation_id = new_conv_id
                st.session_state.chat_history = []
                load_conversations()
                st.rerun()
        
        # Load conversations
        load_conversations()
        
        # Display conversations
        for conv in st.session_state.conversations:
            is_active = conv['id'] == st.session_state.current_conversation_id
            
            # Create conversation item
            conv_container = st.container()
            with conv_container:
                if st.button(
                    f"💬 {conv['title'][:30]}...",
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.current_conversation_id = conv['id']
                    load_conversation_history(conv['id'])
                    st.rerun()
                
                # Show conversation metadata
                st.caption(f"Messages: {conv['message_count']} | {conv['created_at'][:10]}")
        
        st.divider()
        
        # Debug toggle
        st.session_state.show_debug = st.checkbox("Show Debug Info")
        
        # System status
        st.subheader("System Status")
        if st.session_state.rag_system:
            st.success("✅ RAG System Ready")
        else:
            st.error("❌ RAG System Error")
            if st.session_state.init_error:
                st.error(st.session_state.init_error)


def chat_interface():
    """Main chat interface."""
    
    # Check if system is ready
    if not st.session_state.rag_system:
        st.error("System not initialized. Please check the sidebar for errors.")
        return
    
    # Chat header
    if st.session_state.current_conversation_id:
        # Get conversation stats
        try:
            stats = st.session_state.rag_system.get_conversation_stats(
                st.session_state.current_conversation_id
            )
            
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.subheader("💬 Chat")
            with col2:
                st.metric("Turns", stats.get('turn_count', 0))
            with col3:
                st.metric("Confidence", f"{stats.get('avg_confidence', 0):.2f}")
            with col4:
                st.metric("Tokens", stats.get('total_tokens', 0))
        
        except Exception as e:
            st.subheader("💬 Chat")
            if st.session_state.show_debug:
                st.error(f"Stats error: {e}")
    else:
        st.subheader("💬 Select or create a conversation to start chatting")
        return
    
    # Chat history display
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            message_class = "user-message" if message['type'] == 'user' else "assistant-message"
            
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <div><strong>{'You' if message['type'] == 'user' else 'Assistant'}:</strong></div>
                <div>{message['content']}</div>
                <div class="message-meta">
                    {message['timestamp'][:19]}
                    {f" | Confidence: {message['confidence']:.2f}" if message.get('confidence') else ""}
                    {f" | Citations: {len(message['citations'])}" if message.get('citations') else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    st.divider()
    
    # User input
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        # Add user message to display immediately
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat(),
            'confidence': None,
            'citations': None,
            'id': None
        })
        
        # Process with RAG system
        with st.spinner("Thinking..."):
            try:
                start_time = time.time()
                
                turn = st.session_state.rag_system.process_user_message(
                    st.session_state.current_conversation_id,
                    user_input
                )
                
                processing_time = time.time() - start_time
                
                # Add assistant response to display
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': turn.assistant_message.content,
                    'timestamp': turn.assistant_message.timestamp,
                    'confidence': turn.confidence,
                    'citations': turn.citations,
                    'id': turn.assistant_message.id
                })
                
                # Show debug info if enabled
                if st.session_state.show_debug:
                    with st.expander("🔍 Debug Information"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Retrieval Results")
                            st.write(f"**Documents:** {len(turn.doc_results)}")
                            for i, doc in enumerate(turn.doc_results[:3]):
                                st.write(f"{i+1}. {doc.get('path', 'Unknown')[:50]}...")
                            
                            st.write(f"**Memory Items:** {len(turn.memory_results)}")
                            for i, (msg, score) in enumerate(turn.memory_results[:3]):
                                st.write(f"{i+1}. {msg.content[:50]}... (score: {score:.3f})")
                        
                        with col2:
                            st.subheader("Generation Metadata")
                            st.json({
                                'total_time': f"{turn.total_time:.2f}s",
                                'generation_time': f"{turn.generation_time:.2f}s",
                                'confidence': turn.confidence,
                                'citations_count': len(turn.citations or []),
                                'prompt_tokens': turn.prompt_metadata.get('total_tokens', 0),
                                'trimming_applied': turn.prompt_metadata.get('trimming_applied', False)
                            })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing message: {e}")
                if st.session_state.show_debug:
                    st.exception(e)


def main():
    """Main application."""
    initialize_session_state()
    
    # Layout
    sidebar()
    chat_interface()


if __name__ == "__main__":
    main()
