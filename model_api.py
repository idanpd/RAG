"""
Local Model API for fast CPU inference with multiple small LLMs.
Supports TinyLlama, Gemma, Phi, Qwen, and Mistral models locally.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import time

from utils import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a local model."""
    name: str
    family: str  # llama, gemma, phi, qwen, mistral
    path: str
    size_gb: float
    context_length: int
    description: str
    chat_template: Optional[str] = None
    is_loaded: bool = False
    load_time: float = 0.0
    inference_speed: float = 0.0  # tokens/second


class BaseLocalModel(ABC):
    """Base class for local model implementations."""
    
    def __init__(self, model_info: ModelInfo, config: Dict[str, Any]):
        self.model_info = model_info
        self.config = config
        self.model = None
        self.is_loaded = False
        self.load_time = 0.0
        self._lock = threading.Lock()
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def unload_model(self):
        """Unload the model from memory."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if model is available and loaded."""
        pass
    
    def get_info(self) -> ModelInfo:
        """Get model information."""
        self.model_info.is_loaded = self.is_loaded
        self.model_info.load_time = self.load_time
        return self.model_info


class LlamaCppLocalModel(BaseLocalModel):
    """Local model implementation using llama.cpp for CPU optimization."""
    
    def __init__(self, model_info: ModelInfo, config: Dict[str, Any]):
        super().__init__(model_info, config)
        self.threads = config.get('LLM_THREADS', os.cpu_count())
        self.n_batch = config.get('LLM_N_BATCH', 512)  # Larger batch for CPU
        self.context_length = min(model_info.context_length, config.get('LLM_CTX', 2048))
        
        # CPU-specific optimizations
        self.use_mmap = config.get('USE_MMAP', True)
        self.use_mlock = config.get('USE_MLOCK', False)
        self.numa = config.get('NUMA', False)
    
    def load_model(self) -> bool:
        """Load model with CPU optimizations."""
        if self.is_loaded:
            return True
        
        try:
            from llama_cpp import Llama
            
            if not os.path.exists(self.model_info.path):
                logger.error(f"Model file not found: {self.model_info.path}")
                return False
            
            start_time = time.time()
            logger.info(f"Loading {self.model_info.name} model for CPU inference...")
            
            # CPU-optimized parameters
            self.model = Llama(
                model_path=self.model_info.path,
                n_ctx=self.context_length,
                n_threads=self.threads,
                n_batch=self.n_batch,
                use_mmap=self.use_mmap,
                use_mlock=self.use_mlock,
                numa=self.numa,
                chat_format=self.model_info.chat_template,
                verbose=False,
                # CPU-specific optimizations
                n_gpu_layers=0,  # Force CPU usage
                f16_kv=True,     # Use float16 for key-value cache
            )
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            self.model_info.is_loaded = True
            self.model_info.load_time = self.load_time
            
            logger.info(f"Model {self.model_info.name} loaded in {self.load_time:.2f}s")
            
            # Test inference speed
            self._benchmark_speed()
            
            return True
            
        except ImportError:
            logger.error("llama-cpp-python not available. Install with: pip install llama-cpp-python")
            return False
        except Exception as e:
            logger.error(f"Failed to load model {self.model_info.name}: {e}")
            return False
    
    def _benchmark_speed(self):
        """Benchmark inference speed."""
        try:
            test_prompt = "The weather today is"
            start_time = time.time()
            
            response = self.model(
                test_prompt,
                max_tokens=20,
                temperature=0.1,
                echo=False
            )
            
            inference_time = time.time() - start_time
            tokens_generated = len(response["choices"][0]["text"].split())
            
            if inference_time > 0:
                self.model_info.inference_speed = tokens_generated / inference_time
                logger.info(f"Inference speed: {self.model_info.inference_speed:.2f} tokens/sec")
            
        except Exception as e:
            logger.warning(f"Speed benchmark failed: {e}")
    
    def unload_model(self):
        """Unload model from memory."""
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
                self.is_loaded = False
                self.model_info.is_loaded = False
                logger.info(f"Model {self.model_info.name} unloaded")
    
    def is_available(self) -> bool:
        """Check if model is loaded and available."""
        return self.is_loaded and self.model is not None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text with CPU-optimized settings."""
        if not self.is_available():
            return "Error: Model not loaded"
        
        with self._lock:
            try:
                # Default parameters optimized for CPU
                max_tokens = kwargs.get('max_tokens', self.config.get('LLM_MAX_TOKENS', 512))
                temperature = kwargs.get('temperature', self.config.get('LLM_TEMP', 0.3))
                top_p = kwargs.get('top_p', 0.9)
                repeat_penalty = kwargs.get('repeat_penalty', 1.1)
                
                # Use chat completion if available and model supports it
                if self.model_info.chat_template and hasattr(self.model, 'create_chat_completion'):
                    try:
                        response = self.model.create_chat_completion(
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            repeat_penalty=repeat_penalty,
                            stream=False
                        )
                        return response["choices"][0]["message"]["content"].strip()
                    except Exception:
                        # Fallback to completion
                        pass
                
                # Standard completion
                response = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=kwargs.get('stop', []),
                    echo=False
                )
                
                return response["choices"][0]["text"].strip()
                
            except Exception as e:
                logger.error(f"Generation failed for {self.model_info.name}: {e}")
                return f"Error: {e}"


class LocalModelRegistry:
    """Registry for managing local models based on your config."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, BaseLocalModel] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models from your original config."""
        models_dir = Path(self.config.get('MODELS_DIR', './models'))
        
        # Define your local models based on the original config
        model_configs = [
            {
                'name': 'tinyllama-1.1b-chat',
                'family': 'llama',
                'filename': 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
                'size_gb': 0.6,
                'context_length': 2048,
                'description': 'TinyLlama 1.1B - Fastest, good for simple tasks',
                'chat_template': None  # Uses default llama format
            },
            {
                'name': 'gemma-2b-instruct',
                'family': 'gemma',
                'filename': 'gemma-2b-it-q4_k_m.gguf',
                'size_gb': 1.4,
                'context_length': 8192,
                'description': 'Gemma 2B Instruct - Balanced performance and quality',
                'chat_template': 'gemma'
            },
            {
                'name': 'phi3-mini-instruct',
                'family': 'phi',
                'filename': 'phi-3-mini-4k-instruct-q4.gguf',
                'size_gb': 2.2,
                'context_length': 4096,
                'description': 'Phi-3 Mini 3.8B - High quality reasoning',
                'chat_template': 'phi-3'
            },
            {
                'name': 'qwen2-0.5b-instruct',
                'family': 'qwen',
                'filename': 'qwen2-0_5b-instruct-q4_0.gguf',
                'size_gb': 0.4,
                'context_length': 32768,
                'description': 'Qwen2 0.5B - Ultra-fast with long context',
                'chat_template': 'qwen'
            },
            {
                'name': 'mistral-7b-instruct',
                'family': 'mistral',
                'filename': 'mistral-7b-instruct-v0.3.Q4_K_M.gguf',
                'size_gb': 4.1,
                'context_length': 32768,
                'description': 'Mistral 7B - Highest quality, slower on CPU',
                'chat_template': 'mistral-instruct'
            }
        ]
        
        # Register available models
        for model_config in model_configs:
            model_path = models_dir / model_config['filename']
            
            model_info = ModelInfo(
                name=model_config['name'],
                family=model_config['family'],
                path=str(model_path),
                size_gb=model_config['size_gb'],
                context_length=model_config['context_length'],
                description=model_config['description'],
                chat_template=model_config['chat_template']
            )
            
            # Only register if file exists
            if model_path.exists():
                self.models[model_info.name] = model_info
                logger.info(f"Registered model: {model_info.name} ({model_info.size_gb}GB)")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        if not self.models:
            logger.warning("No local models found. Please download models to the models directory.")
        else:
            logger.info(f"Found {len(self.models)} local models")
    
    def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        return list(self.models.values())
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.models.get(model_name)
    
    def load_model(self, model_name: str) -> bool:
        """Load a model into memory."""
        if model_name not in self.models:
            logger.error(f"Model not found: {model_name}")
            return False
        
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        model_info = self.models[model_name]
        
        # Create model instance
        local_model = LlamaCppLocalModel(model_info, self.config.config)
        
        # Load model
        if local_model.load_model():
            self.loaded_models[model_name] = local_model
            return True
        
        return False
    
    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.loaded_models:
            self.loaded_models[model_name].unload_model()
            del self.loaded_models[model_name]
            logger.info(f"Model {model_name} unloaded")
    
    def unload_all_models(self):
        """Unload all models from memory."""
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
    
    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate text using a specific model."""
        if model_name not in self.loaded_models:
            # Try to load the model automatically
            if not self.load_model(model_name):
                return f"Error: Could not load model {model_name}"
        
        return self.loaded_models[model_name].generate(prompt, **kwargs)
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model names."""
        return list(self.loaded_models.keys())
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about models."""
        stats = {
            'total_models': len(self.models),
            'loaded_models': len(self.loaded_models),
            'available_models': [],
            'loaded_model_info': []
        }
        
        for model_info in self.models.values():
            stats['available_models'].append({
                'name': model_info.name,
                'family': model_info.family,
                'size_gb': model_info.size_gb,
                'description': model_info.description,
                'is_loaded': model_info.name in self.loaded_models
            })
        
        for model_name, model in self.loaded_models.items():
            info = model.get_info()
            stats['loaded_model_info'].append({
                'name': info.name,
                'load_time': info.load_time,
                'inference_speed': info.inference_speed
            })
        
        return stats


class LocalModelAPI:
    """High-level API for local model management."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.registry = LocalModelRegistry(self.config)
        self.default_model = self.config.get('DEFAULT_MODEL', 'tinyllama-1.1b-chat')
        
        # Auto-load default model if specified
        if self.default_model and self.default_model in self.registry.models:
            logger.info(f"Auto-loading default model: {self.default_model}")
            self.registry.load_model(self.default_model)
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with their info."""
        models = []
        for model_info in self.registry.list_models():
            models.append({
                'name': model_info.name,
                'family': model_info.family,
                'size_gb': model_info.size_gb,
                'context_length': model_info.context_length,
                'description': model_info.description,
                'is_loaded': model_info.name in self.registry.loaded_models,
                'path': model_info.path
            })
        return models
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a model and return status."""
        start_time = time.time()
        success = self.registry.load_model(model_name)
        load_time = time.time() - start_time
        
        return {
            'success': success,
            'model_name': model_name,
            'load_time': load_time,
            'message': f"Model {model_name} {'loaded successfully' if success else 'failed to load'}"
        }
    
    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a model and return status."""
        was_loaded = model_name in self.registry.loaded_models
        self.registry.unload_model(model_name)
        
        return {
            'success': True,
            'model_name': model_name,
            'message': f"Model {model_name} {'unloaded successfully' if was_loaded else 'was not loaded'}"
        }
    
    def generate_text(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate text using specified or default model."""
        model_to_use = model_name or self.default_model
        
        if not model_to_use:
            return {
                'success': False,
                'error': 'No model specified and no default model set',
                'text': ''
            }
        
        start_time = time.time()
        try:
            generated_text = self.registry.generate(model_to_use, prompt, **kwargs)
            generation_time = time.time() - start_time
            
            return {
                'success': True,
                'text': generated_text,
                'model_used': model_to_use,
                'generation_time': generation_time,
                'prompt_length': len(prompt),
                'response_length': len(generated_text)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'model_used': model_to_use
            }
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        return self.registry.get_model_stats()
    
    def set_default_model(self, model_name: str) -> Dict[str, Any]:
        """Set the default model for generation."""
        if model_name not in self.registry.models:
            return {
                'success': False,
                'message': f"Model {model_name} not found"
            }
        
        self.default_model = model_name
        return {
            'success': True,
            'message': f"Default model set to {model_name}"
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.registry.unload_all_models()
        self.registry.executor.shutdown(wait=True)