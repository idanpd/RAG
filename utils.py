import os
import yaml
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with fallback to environment variables."""
    config = {}
    
    # Try to load from YAML file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
    
    # Override with environment variables
    env_mappings = {
        'DATA_PATH': 'DATA_PATH',
        'INDEX_DIR': 'INDEX_DIR',
        'SQLITE_DB': 'SQLITE_DB',
        'EMBED_MODEL': 'EMBED_MODEL',
        'LLM_MODEL_PATH': 'LLM_MODEL_PATH',
        'LOG_LEVEL': 'LOG_LEVEL',
    }
    
    for config_key, env_key in env_mappings.items():
        if env_key in os.environ:
            config[config_key] = os.environ[env_key]
    
    return config


def setup_logger(level: str = "INFO") -> logging.Logger:
    """Setup logger with specified level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_sha256(file_path: Union[str, Path]) -> str:
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception:
        return ""


def save_json(data: Any, file_path: Union[str, Path]) -> bool:
    """Save data to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def load_json(file_path: Union[str, Path]) -> Optional[Any]:
    """Load data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def normalize_path(path: Union[str, Path]) -> str:
    """Normalize file path for cross-platform compatibility."""
    return str(Path(path).resolve())


class ConfigManager:
    """Configuration manager with validation and defaults."""
    
    DEFAULT_CONFIG = {
        # Paths
        'DATA_PATH': './data',
        'INDEX_DIR': './indices',
        'SQLITE_DB': 'index.db',
        
        # Embeddings
        'EMBED_MODEL': 'all-MiniLM-L6-v2',
        'CHUNK_SIZE': 500,
        'CHUNK_OVERLAP': 100,
        'FAISS_NLIST': 100,
        'FAISS_PQ_M': 8,
        
        # Retriever
        'BM25_TOPK': 200,
        'DENSE_TOPK': 50,
        'RERANK_TOPK': 3,
        'CROSS_ENCODER': 'cross-encoder/ms-marco-TinyBERT-L-2-v2',
        
        # LLM
        'LLM_MODEL_PATH': 'models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
        'LLM_FAMILY': 'llama',
        'LLM_CTX': 2048,
        'LLM_THREADS': 4,
        'LLM_N_BATCH': 256,
        'LLM_TEMP': 0.3,
        'LLM_MAX_TOKENS': 512,
        
        # Image & Video extraction
        'USE_OCR': True,
        'USE_IMAGE_CAPTION': True,
        'IMAGE_CAPTION_MODEL': 'Salesforce/blip-image-captioning-base',
        'USE_VIDEO_TRANSCRIPT': False,
        'WHISPER_CPP_BIN': '',
        'WHISPER_MODEL': '',
        'USE_VIDEO_KEYFRAME_OCR': True,
        'KEYFRAME_EVERY_SEC': 3,
        'KEYFRAME_MAX': 10,
        'CAPTION_MAX_TOKENS': 32,
        
        # Logging
        'LOG_LEVEL': 'INFO',
        
        # System
        'REBUILD_INDEX': True,
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self):
        """Load and merge configuration from file and environment."""
        # Load from file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f) or {}
                self.config.update(file_config)
            except Exception as e:
                logging.warning(f"Failed to load config from {self.config_path}: {e}")
        
        # Override with environment variables
        for key in self.config:
            env_value = os.environ.get(key)
            if env_value is not None:
                # Try to convert to appropriate type
                if isinstance(self.config[key], bool):
                    self.config[key] = env_value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(self.config[key], int):
                    try:
                        self.config[key] = int(env_value)
                    except ValueError:
                        pass
                elif isinstance(self.config[key], float):
                    try:
                        self.config[key] = float(env_value)
                    except ValueError:
                        pass
                else:
                    self.config[key] = env_value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save config to {self.config_path}: {e}")
            return False