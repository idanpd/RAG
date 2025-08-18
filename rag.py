"""
RAG (Retrieval-Augmented Generation) system with support for multiple small LLMs.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

from utils import ConfigManager

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


class LLMManager:
    """Manages multiple LLM implementations and provides unified interface."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.llms = {}
        self.active_llm = None
        self._initialize_llms()
    
    def _initialize_llms(self):
        """Initialize available LLMs."""
        llm_config = self.config.config
        
        # Try llama.cpp first (most common for local models)
        if llm_config.get('LLM_MODEL_PATH'):
            try:
                llamacpp_llm = LlamaCppLLM(llm_config)
                if llamacpp_llm.is_available():
                    self.llms['llamacpp'] = llamacpp_llm
                    if not self.active_llm:
                        self.active_llm = 'llamacpp'
                        logger.info("Using llama.cpp as primary LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize llama.cpp: {e}")
        
        # Try Transformers
        if llm_config.get('HF_MODEL_NAME'):
            try:
                transformers_llm = TransformersLLM(llm_config)
                if transformers_llm.is_available():
                    self.llms['transformers'] = transformers_llm
                    if not self.active_llm:
                        self.active_llm = 'transformers'
                        logger.info("Using Transformers as primary LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize Transformers: {e}")
        
        # Try Ollama
        if llm_config.get('OLLAMA_MODEL'):
            try:
                ollama_llm = OllamaLLM(llm_config)
                if ollama_llm.is_available():
                    self.llms['ollama'] = ollama_llm
                    if not self.active_llm:
                        self.active_llm = 'ollama'
                        logger.info("Using Ollama as primary LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama: {e}")
        
        if not self.llms:
            logger.error("No LLMs available!")
        else:
            logger.info(f"Available LLMs: {list(self.llms.keys())}")
    
    def get_available_llms(self) -> List[str]:
        """Get list of available LLM names."""
        return list(self.llms.keys())
    
    def set_active_llm(self, llm_name: str) -> bool:
        """Set the active LLM."""
        if llm_name in self.llms:
            self.active_llm = llm_name
            logger.info(f"Switched to LLM: {llm_name}")
            return True
        else:
            logger.error(f"LLM not available: {llm_name}")
            return False
    
    def generate(self, prompt: str, llm_name: Optional[str] = None, **kwargs) -> str:
        """Generate text using specified or active LLM."""
        llm_to_use = llm_name or self.active_llm
        
        if not llm_to_use or llm_to_use not in self.llms:
            return "Error: No LLM available"
        
        return self.llms[llm_to_use].generate(prompt, **kwargs)


class PromptTemplate:
    """Template for RAG prompts with different styles."""
    
    TEMPLATES = {
        'default': """You are a helpful assistant. Answer the question using only the provided contexts and cite file paths.

CONTEXTS:
{context}

QUESTION:
{query}

Answer concisely and include citations like [file:path].""",
        
        'detailed': """You are an expert assistant. Based on the provided contexts, give a comprehensive answer to the question.

CONTEXTS:
{context}

QUESTION:
{query}

Instructions:
- Use only the information from the provided contexts
- Be thorough and detailed in your response
- Cite sources using [file:filename] format
- If the contexts don't contain relevant information, say so clearly

ANSWER:""",
        
        'concise': """Answer briefly using only the provided contexts.

CONTEXTS:
{context}

QUESTION: {query}

BRIEF ANSWER:""",
        
        'analytical': """You are an analytical assistant. Analyze the provided contexts to answer the question.

CONTEXTS:
{context}

QUESTION:
{query}

Please provide:
1. A direct answer to the question
2. Supporting evidence from the contexts
3. Any limitations or gaps in the available information

Use [file:filename] to cite sources.

ANALYSIS:"""
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


class RAGSystem:
    """Complete RAG system with retrieval and generation."""
    
    def __init__(self, retriever, config: Optional[ConfigManager] = None):
        self.retriever = retriever
        self.config = config or ConfigManager()
        self.llm_manager = LLMManager(self.config)
        
        # Configuration
        self.default_template = self.config.get('RAG_TEMPLATE', 'default')
        self.max_context_length = self.config.get('MAX_CONTEXT_LENGTH', 2000)
        
        logger.info("RAG system initialized")
    
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


# Legacy compatibility functions
def build_prompt(query: str, results: List[Dict[str, Any]]) -> str:
    """Legacy function for building prompts."""
    return PromptTemplate.build_prompt(query, results, 'default')


def answer_with_llm(prompt: str) -> str:
    """Legacy function for answering with LLM."""
    config = ConfigManager()
    llm_manager = LLMManager(config)
    return llm_manager.generate(prompt)