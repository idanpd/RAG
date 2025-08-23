"""
RAG (Retrieval-Augmented Generation) system with support for multiple local small LLMs.
Optimized for fast CPU inference with local models.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

from utils import ConfigManager
from model_api import LocalModelAPI

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Base class for LLM implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get('LLM_MODEL_PATH', '')
        self.max_tokens = config.get('LLM_MAX_TOKENS', 512)
        self.temperature = config.get('LLM_TEMP', 0.3)
        self.context_length = config.get('LLM_CTX', 2048)
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is available."""
        pass


class LlamaCppLLM(BaseLLM):
    """LLM implementation using llama.cpp Python bindings."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = None
        self.family = config.get('LLM_FAMILY', 'llama').lower()
        self.threads = config.get('LLM_THREADS', 4)
        self.n_batch = config.get('LLM_N_BATCH', 256)
        
        # Chat format mapping for different model families
        self.chat_format = None
        if self.family == 'gemma':
            self.chat_format = 'gemma'
        elif self.family == 'phi':
            self.chat_format = 'phi-3'
        elif self.family == 'qwen':
            self.chat_format = 'qwen'
        elif self.family == 'mistral':
            self.chat_format = 'mistral-instruct'
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the llama.cpp model."""
        try:
            from llama_cpp import Llama
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.context_length,
                n_threads=self.threads,
                n_batch=self.n_batch,
                chat_format=self.chat_format,
                verbose=False
            )
            
            logger.info(f"Initialized {self.family} model: {Path(self.model_path).name}")
            
        except ImportError:
            logger.error("llama-cpp-python not available. Install with: pip install llama-cpp-python")
        except Exception as e:
            logger.error(f"Failed to initialize llama.cpp model: {e}")
    
    def is_available(self) -> bool:
        """Check if the LLM is available."""
        return self.llm is not None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using llama.cpp."""
        if not self.is_available():
            return "Error: LLM not available"
        
        try:
            # Override config with kwargs
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            
            # Try chat completion first (for instruct models)
            if hasattr(self.llm, 'create_chat_completion') and self.chat_format:
                try:
                    response = self.llm.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False
                    )
                    return response["choices"][0]["message"]["content"].strip()
                except Exception:
                    # Fallback to completion
                    pass
            
            # Use completion API
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=kwargs.get('stop', []),
                echo=False
            )
            
            return response["choices"][0]["text"].strip()
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return f"Error: {e}"


class TransformersLLM(BaseLLM):
    """LLM implementation using Transformers library."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = config.get('DEVICE', 'cpu')
        self._initialize()
    
    def _initialize(self):
        """Initialize the Transformers model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_name = self.config.get('HF_MODEL_NAME', 'microsoft/DialoGPT-small')
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map=self.device if self.device == 'cuda' else None
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Initialized Transformers model: {model_name}")
            
        except ImportError:
            logger.error("transformers library not available. Install with: pip install transformers torch")
        except Exception as e:
            logger.error(f"Failed to initialize Transformers model: {e}")
    
    def is_available(self) -> bool:
        """Check if the LLM is available."""
        return self.model is not None and self.tokenizer is not None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Transformers."""
        if not self.is_available():
            return "Error: LLM not available"
        
        try:
            import torch
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated = outputs[0][inputs.shape[1]:]  # Remove input tokens
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return f"Error: {e}"


class OllamaLLM(BaseLLM):
    """LLM implementation using Ollama API."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.model_name = config.get('OLLAMA_MODEL', 'llama2')
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            self.available = response.status_code == 200
            if self.available:
                logger.info(f"Connected to Ollama server: {self.base_url}")
        except Exception as e:
            logger.warning(f"Ollama server not available: {e}")
            self.available = False
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return getattr(self, 'available', False)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama API."""
        if not self.is_available():
            return "Error: Ollama not available"
        
        try:
            import requests
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": kwargs.get('temperature', self.temperature),
                    "num_predict": kwargs.get('max_tokens', self.max_tokens),
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=False,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error: {e}"


class LocalLLMManager:
    """Manages multiple local LLM models with CPU optimization."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model_api = LocalModelAPI(config)
        self.active_model = self.config.get('DEFAULT_MODEL', 'tinyllama-1.1b-chat')
        
        # Get available models
        self.available_models = self.model_api.list_available_models()
        
        if self.available_models:
            logger.info(f"Found {len(self.available_models)} local models")
            for model in self.available_models:
                status = "loaded" if model['is_loaded'] else "available"
                logger.info(f"  - {model['name']}: {model['description']} ({status})")
        else:
            logger.error("No local models found! Please download models to ./models/ directory")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with details."""
        return self.available_models
    
    def get_available_llms(self) -> List[str]:
        """Get list of available model names (for compatibility)."""
        return [model['name'] for model in self.available_models]
    
    def set_active_llm(self, model_name: str) -> bool:
        """Set the active model."""
        available_names = [model['name'] for model in self.available_models]
        
        if model_name in available_names:
            self.active_model = model_name
            # Ensure model is loaded
            result = self.model_api.load_model(model_name)
            if result['success']:
                logger.info(f"Switched to model: {model_name}")
                return True
            else:
                logger.error(f"Failed to load model: {model_name}")
                return False
        else:
            logger.error(f"Model not available: {model_name}")
            return False
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a specific model."""
        return self.model_api.load_model(model_name)
    
    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a specific model."""
        return self.model_api.unload_model(model_name)
    
    def generate(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> str:
        """Generate text using specified or active model."""
        model_to_use = model_name or self.active_model
        
        if not model_to_use:
            return "Error: No model specified"
        
        result = self.model_api.generate_text(prompt, model_to_use, **kwargs)
        
        if result['success']:
            return result['text']
        else:
            return f"Error: {result.get('error', 'Generation failed')}"
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return self.model_api.get_model_stats()
    
    def cleanup(self):
        """Clean up resources."""
        self.model_api.cleanup()


# Keep the old LLMManager name for compatibility
LLMManager = LocalLLMManager


class PromptTemplate:
    """Template for RAG prompts with different styles."""
    
    TEMPLATES = {
        'default': """You are a helpful expert assistant. Answer the question using only the provided contexts.

CONTEXTS:
{context}

QUESTION:
{query}

Instructions:
- Provide a clear and comprehensive summary of the answer
- Always include citations in the format [file:filename or file:path]

FINAL ANSWER:"""
}
    
    @classmethod
    def build_prompt(cls, query: str, results: List[Dict[str, Any]], 
                    template: str = 'default', max_context_length: int = 2000) -> str:
        """Build a prompt from query and search results."""
        if template not in cls.TEMPLATES:
            template = 'default'
        
        # Build context from results
        contexts = []
        current_length = 0
        
        for result in results:
            file_path = result.get('path', 'unknown')
            chunk_text = result.get('text', '')
            chunk_id = result.get('chunk_id', '')
            
            context_entry = f"FILE: {file_path}\nCHUNK_ID: {chunk_id}\n---\n{chunk_text}\n"
            
            # Check if adding this context would exceed the limit
            if current_length + len(context_entry) > max_context_length and contexts:
                break
            
            contexts.append(context_entry)
            current_length += len(context_entry)
        
        context_text = "\n\n".join(contexts)
        
        return cls.TEMPLATES[template].format(
            context=context_text,
            query=query
        )


class ConversationManager:
    """Manages conversation state and memory using existing database."""
    
    def __init__(self, db_retriever, config: ConfigManager):
        self.db_retriever = db_retriever
        self.config = config
        self.summarization_threshold = config.get('SUMMARIZATION_THRESHOLD', 10)
        
    def add_message(self, conversation_id: str, message_type: str, content: str, 
                   prev_id: Optional[int] = None, confidence: float = 0.5, 
                   citations: Optional[List[str]] = None) -> int:
        """Add a conversation message to the chunks table."""
        import uuid
        from datetime import datetime, timezone
        
        cursor = self.db_retriever.conn.cursor()
        
        # Calculate token count
        token_count = len(content.split())
        citations_json = json.dumps(citations) if citations else None
        
        cursor.execute("""
            INSERT INTO chunks 
            (file_id, chunk_text, summary, chunk_type, token_count, prev_id,
             conversation_id, timestamp, source, confidence, citations, archived)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            None,  # No file for conversation messages
            content,
            content[:200],  # Summary
            message_type,
            token_count,
            prev_id,
            conversation_id,
            datetime.now(timezone.utc).isoformat(),
            'rag_system',
            confidence,
            citations_json,
            False
        ))
        
        message_id = cursor.lastrowid
        self.db_retriever.conn.commit()
        
        return message_id
    
    def get_sliding_window(self, conversation_id: str, window_size: int = 6) -> List[Dict[str, Any]]:
        """Get recent conversation messages."""
        cursor = self.db_retriever.conn.cursor()
        cursor.execute("""
            SELECT * FROM chunks 
            WHERE conversation_id = ? AND chunk_type IN ('user_msg', 'assistant_msg')
            AND archived = FALSE
            ORDER BY timestamp DESC
            LIMIT ?
        """, (conversation_id, window_size))
        
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
        
        return list(reversed(messages))  # Return in chronological order


class RAGSystem:
    """Complete RAG system with retrieval and generation, now with conversation support."""
    
    def __init__(self, retriever, config: Optional[ConfigManager] = None):
        self.retriever = retriever
        self.config = config or ConfigManager()
        self.llm_manager = LLMManager(self.config)
        
        # Add conversation manager
        self.conversation_manager = ConversationManager(retriever.db_retriever, self.config)
        
        # Configuration
        self.default_template = self.config.get('RAG_TEMPLATE', 'default')
        self.max_context_length = self.config.get('MAX_CONTEXT_LENGTH', 2000)
        
        logger.info("RAG system initialized with conversation support")
    
    def answer_query(self, query: str, template: Optional[str] = None, 
                    llm_name: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Answer a query using RAG."""
        # Retrieve relevant documents
        logger.info(f"Retrieving documents for query: {query}")
        results = self.retriever.search(query, top_k)
        
        if not results:
            return {
                'query': query,
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'llm_used': None
            }
        
        # Convert results to dict format if needed
        if hasattr(results[0], 'to_dict'):
            results = [r.to_dict() for r in results]
        
        # Build prompt
        template_name = template or self.default_template
        prompt = PromptTemplate.build_prompt(
            query, results, template_name, self.max_context_length
        )
        
        # Generate answer
        logger.info(f"Generating answer using template: {template_name}")
        answer = self.llm_manager.generate(prompt, llm_name)
        
        # Prepare sources
        sources = [
            {
                'path': r.get('path', ''),
                'chunk_id': r.get('chunk_id', ''),
                'score': r.get('final_score', r.get('score', 0)),
                'summary': r.get('summary', '')
            }
            for r in results
        ]
        
        return {
            'query': query,
            'answer': answer,
            'sources': sources,
            'llm_used': self.llm_manager.active_llm,
            'template_used': template_name,
            'context_length': len(prompt)
        }
    
    def get_available_templates(self) -> List[str]:
        """Get available prompt templates."""
        return list(PromptTemplate.TEMPLATES.keys())
    
    def get_available_llms(self) -> List[str]:
        """Get available LLMs."""
        return self.llm_manager.get_available_llms()
    
    def set_active_llm(self, llm_name: str) -> bool:
        """Set the active LLM."""
        return self.llm_manager.set_active_llm(llm_name)
    
    def answer_conversational_query(self, query: str, conversation_id: str, 
                                  template: Optional[str] = None, 
                                  llm_name: Optional[str] = None) -> Dict[str, Any]:
        """Answer a query with both document and conversation context."""
        
        # Dual retrieval: documents + memory
        doc_results, memory_results = self.retriever.search_dual(query, conversation_id)
        
        # Convert results to dict format if needed
        if doc_results and hasattr(doc_results[0], 'to_dict'):
            doc_results = [r.to_dict() for r in doc_results]
        if memory_results and hasattr(memory_results[0], 'to_dict'):
            memory_results = [r.to_dict() for r in memory_results]
        
        # Get sliding window
        sliding_window = self.conversation_manager.get_sliding_window(conversation_id)
        
        # Build conversational prompt
        template_name = template or self.default_template
        prompt = self._build_conversational_prompt(
            query, doc_results, memory_results, sliding_window, template_name
        )
        
        # Generate answer
        response = self.llm_manager.generate(prompt, llm_name)
        
        # Analyze response
        citations, confidence = self._analyze_response(response, doc_results, memory_results)
        
        # Get last message ID for threading
        last_msg_id = self._get_last_message_id(conversation_id)
        
        # Add user message
        user_msg_id = self.conversation_manager.add_message(
            conversation_id, 'user_msg', query, last_msg_id, 1.0
        )
        
        # Add assistant message
        assistant_msg_id = self.conversation_manager.add_message(
            conversation_id, 'assistant_msg', response, user_msg_id, confidence, citations
        )
        
        return {
            'query': query,
            'answer': response,
            'doc_sources': [{'path': r.get('path', ''), 'chunk_id': r.get('chunk_id', ''), 
                           'score': r.get('final_score', r.get('score', 0))} for r in doc_results],
            'memory_sources': [{'content': r.get('text', ''), 'type': r.get('chunk_type', ''),
                              'timestamp': r.get('timestamp', '')} for r in memory_results],
            'sliding_window': sliding_window,
            'llm_used': self.llm_manager.active_model,
            'template_used': template_name,
            'confidence': confidence,
            'citations': citations,
            'conversation_id': conversation_id,
            'user_msg_id': user_msg_id,
            'assistant_msg_id': assistant_msg_id
        }
    
    def _build_conversational_prompt(self, query: str, doc_results: List[Dict], 
                                   memory_results: List[Dict], sliding_window: List[Dict],
                                   template: str) -> str:
        """Build prompt with both document and conversation context."""
        
        # Simplified system prompt for conversational RAG
        system_prompt = """You are a helpful AI assistant with access to documents and conversation history.

Instructions:
- Answer questions using the provided context from documents and conversation history
- Cite sources using [doc:filename] for documents and [memory:turn] for conversation context
- If information is not in the provided context, say so clearly
"""
        
        # Recent conversation (sliding window)
        conversation_context = ""
        if sliding_window:
            conversation_context = "Recent conversation:\n"
            for msg in sliding_window[-4:]:  # Last 4 messages
                # Handle both dict and object representations
                chunk_type = msg.get('chunk_type') if isinstance(msg, dict) else getattr(msg, 'chunk_type', '')
                text = msg.get('chunk_text') or msg.get('text', '') if isinstance(msg, dict) else getattr(msg, 'text', '')
                
                role = "User" if chunk_type == 'user_msg' else "Assistant"
                conversation_context += f"{role}: {text}\n"
            conversation_context += "\n"
        
        # Document context
        docs_context = ""
        if doc_results:
            docs_context = "Relevant documents:\n"
            for result in doc_results[:5]:  # Top 5 documents
                filename = result.get('path', 'Unknown').split('/')[-1]
                content = result.get('text', '')[:400]  # Limit for token budget
                docs_context += f"[doc:{filename}] {content}\n"
            docs_context += "\n"
        
        # Memory context (relevant past exchanges)
        memory_context = ""
        if memory_results:
            memory_context = "Relevant conversation history:\n"
            for msg in memory_results[:3]:  # Top 3 memory items
                # Handle both SearchResult objects and dict representations
                chunk_type = msg.get('chunk_type') if isinstance(msg, dict) else getattr(msg, 'chunk_type', '')
                text = msg.get('text') or msg.get('chunk_text', '') if isinstance(msg, dict) else getattr(msg, 'text', '')
                
                if chunk_type == 'summary':
                    memory_context += f"[memory:summary] {text}\n"
                elif chunk_type == 'assistant_msg':
                    memory_context += f"[memory:response] {text}\n"
                elif chunk_type == 'user_msg':
                    memory_context += f"[memory:question] {text}\n"
            memory_context += "\n"
        
        # Assemble final prompt
        prompt = (system_prompt + 
                 conversation_context + 
                 docs_context + 
                 memory_context + 
                 f"User: {query}\nAssistant: ")
        
        return prompt
    
    def _analyze_response(self, response: str, doc_results: List[Dict], 
                         memory_results: List[Dict]) -> Tuple[List[str], float]:
        """Analyze response for citations and confidence."""
        
        citations = []
        confidence = 0.5  # Base confidence
        
        # Extract citations
        import re
        doc_citations = re.findall(r'\[doc:([^\]]+)\]', response)
        memory_citations = re.findall(r'\[memory:([^\]]+)\]', response)
        
        citations.extend([f"doc:{cite}" for cite in doc_citations])
        citations.extend([f"memory:{cite}" for cite in memory_citations])
        
        # Calculate confidence
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
    
    def _get_last_message_id(self, conversation_id: str) -> Optional[int]:
        """Get the last message ID in a conversation."""
        cursor = self.conversation_manager.db_retriever.conn.cursor()
        cursor.execute("""
            SELECT id FROM chunks 
            WHERE conversation_id = ? AND chunk_type IN ('user_msg', 'assistant_msg')
            ORDER BY timestamp DESC LIMIT 1
        """, (conversation_id,))
        
        result = cursor.fetchone()
        return result[0] if result else None


# Legacy compatibility functions
def build_prompt(query: str, results: List[Dict[str, Any]]) -> str:
    """Legacy function for building prompts."""
    return PromptTemplate.build_prompt(query, results, 'default')


def answer_with_llm(prompt: str) -> str:
    """Legacy function for answering with LLM."""
    config = ConfigManager()
    llm_manager = LLMManager(config)
    return llm_manager.generate(prompt)