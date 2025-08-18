# Multi-Modal Semantic Search System

A powerful semantic search engine that processes **text, images, and videos**, indexes them efficiently, and provides intelligent retrieval with AI-powered answers using multiple small LLMs.

## 🎯 System Overview

This system builds a **semantic search engine** that:
- Processes multi-modal content (text documents, images, videos)
- Splits content into **fixed-size semantic chunks** with stable token budgets
- Provides **hybrid search** (BM25 + dense retrieval + reranking)
- Supports **multiple small LLMs** for RAG (Retrieval-Augmented Generation)
- Offers both **CLI and interactive interfaces**

## ✨ Key Features

### 🔍 **Advanced Search**
- **Hybrid retrieval**: BM25 sparse + dense vector search + cross-encoder reranking
- **Semantic chunking**: Token-aware splitting with sentence boundary detection
- **Multi-modal support**: Text, images (OCR + captioning), videos (transcripts + keyframe OCR)

### 🤖 **Multiple LLM Support**
- **llama.cpp**: TinyLlama, Gemma 2B, Phi-3 Mini, Qwen2 0.5B, Mistral 7B
- **Transformers**: HuggingFace models with GPU/CPU support
- **Ollama**: Local API server integration
- **Automatic fallbacks** and model switching

### 📊 **Robust Architecture**
- **Class-based design** with proper inheritance
- **Database management** with SQLite + FAISS indexing
- **Configurable components** with YAML configuration
- **Error handling** and recovery mechanisms

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo>
cd semantic-search-system

# Option A: Automated CPU-optimized setup (Recommended)
python setup_cpu.py

# Option B: Manual installation
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr ffmpeg

# macOS
brew install tesseract ffmpeg
```

### 2. Download Models

```bash
# Download recommended models (1GB total)
python download_models.py --recommended

# Or list all available models
python download_models.py --list

# Or download specific model
python download_models.py --model tinyllama-1.1b-chat
```

### 3. Build Index

```bash
# Build search index from your documents
python main.py --build-index
```

### 4. Start Searching

```bash
# Interactive mode (recommended)
python main.py --interactive

# Or one-time search
python main.py --search "your query here"

# Or ask a question with AI
python main.py --ask "What is this document about?"
```

## 📖 Usage Guide

### Interactive Mode

The interactive mode provides the best experience:

```bash
python main.py --interactive
```

**Available commands:**
- Type any question to search
- `rag: <question>` - Get AI-powered answers
- `model: <model_name>` - Switch/load models
- `unload: <model_name>` - Unload models from memory
- `models` - List available models and status
- `template: <template_name>` - Change prompt templates
- `stats` - View index statistics
- `quit` - Exit

### Command Line Interface

```bash
# Build/rebuild index
python main.py --build-index
python main.py --rebuild-index

# Model management
python main.py --models                              # List available models
python main.py --load-model tinyllama-1.1b-chat     # Load specific model
python main.py --unload-model tinyllama-1.1b-chat   # Unload model

# Search operations
python main.py --search "machine learning" --top-k 10
python main.py --ask "Explain machine learning" --template detailed --llm phi3-mini-instruct

# View statistics
python main.py --stats
```

### Supported File Types

**Documents:**
- PDF, DOCX, TXT, MD, CSV, JSON
- Excel (XLSX, XLS), PowerPoint (PPTX)

**Images:**
- PNG, JPG, JPEG, BMP, TIFF, WEBP
- Automatic OCR text extraction
- AI-powered image captioning (BLIP)

**Videos:**
- MP4, AVI, MKV, MOV, WEBM
- Keyframe OCR extraction
- Transcript extraction (with whisper.cpp)

## 🔧 Configuration

### LLM Models

The system supports multiple small LLM families:

**llama.cpp Models:**
```yaml
LLM_FAMILY: llama
LLM_MODEL_PATH: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf  # 1.1B params
```

**Gemma Models:**
```yaml
LLM_FAMILY: gemma
LLM_MODEL_PATH: models/gemma-2b-it-q4_k_m.gguf  # 2B params
```

**Phi-3 Models:**
```yaml
LLM_FAMILY: phi
LLM_MODEL_PATH: models/phi-3-mini-4k-instruct-q4.gguf  # 3.8B params
```

**Qwen Models:**
```yaml
LLM_FAMILY: qwen
LLM_MODEL_PATH: models/qwen2-0_5b-instruct-q4_0.gguf  # 0.5B params
```

### Prompt Templates

Choose from different answer styles:

- `default`: Balanced answers with citations
- `detailed`: Comprehensive analysis
- `concise`: Brief, to-the-point responses
- `analytical`: Structured analysis with evidence

### Search Configuration

```yaml
# Chunking
CHUNK_SIZE: 500        # Target tokens per chunk
CHUNK_OVERLAP: 100     # Overlap between chunks

# Retrieval
BM25_TOPK: 200        # BM25 prefilter results
DENSE_TOPK: 50        # Dense retrieval results  
RERANK_TOPK: 5        # Final reranked results

# Models
EMBED_MODEL: all-MiniLM-L6-v2
CROSS_ENCODER: cross-encoder/ms-marco-TinyBERT-L-2-v2
```

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Extractors    │    │    Indexer      │    │   Retriever     │
│                 │    │                 │    │                 │
│ • TextExtractor │───▶│ • SemanticIndexer│───▶│ • SemanticRetriever│
│ • ImageExtractor│    │ • DatabaseManager│    │ • BM25Retriever │
│ • VideoExtractor│    │ • EmbeddingManager│   │ • DenseRetriever│
└─────────────────┘    └─────────────────┘    │ • CrossEncoder  │
                                              └─────────────────┘
                                                       │
                                              ┌─────────────────┐
                                              │   RAG System    │
                                              │                 │
                                              │ • LLMManager    │
                                              │ • PromptTemplate│
                                              │ • Multiple LLMs │
                                              └─────────────────┘
```

### Data Flow

1. **Extraction**: Multi-modal content → structured text
2. **Chunking**: Text → semantic chunks with token awareness
3. **Indexing**: Chunks → embeddings → FAISS index + SQLite metadata
4. **Retrieval**: Query → BM25 + dense search → reranking
5. **Generation**: Retrieved chunks → LLM → final answer

## 📁 Project Structure

```
semantic-search-system/
├── config.yaml              # System configuration
├── main.py                  # Main CLI interface
├── utils.py                 # Utilities and config management
├── indexer.py               # Indexing system
├── retriever.py             # Search and retrieval
├── rag.py                   # RAG system with multiple LLMs
├── chunker.py               # Semantic chunking
├── extractors/              # Content extraction
│   ├── __init__.py
│   ├── base.py             # Base extractor classes
│   ├── text_extractor.py   # Document processing
│   ├── image_extractor.py  # Image OCR + captioning
│   └── video_extractor.py  # Video analysis
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔄 Migration from Legacy Code

The system maintains **backward compatibility** with your original code:

```python
# Legacy interface still works
from utils import load_config, setup_logger
from indexer import Indexer
from retriever import Retriever  
from rag import build_prompt, answer_with_llm

# New class-based interface (recommended)
from indexer import SemanticIndexer
from retriever import SemanticRetriever
from rag import RAGSystem
```

## 🛠️ Development

### Adding New LLMs

```python
# In rag.py
class CustomLLM(BaseLLM):
    def generate(self, prompt: str, **kwargs) -> str:
        # Your implementation
        pass
    
    def is_available(self) -> bool:
        # Check availability
        pass
```

### Adding New Extractors

```python
# In extractors/
class CustomExtractor(BaseExtractor):
    SUPPORTED_EXTENSIONS = {'.custom'}
    
    def extract(self, file_path: Path) -> ExtractionResult:
        # Your extraction logic
        pass
```

## 📊 Performance Tips

1. **Use appropriate chunk sizes**: 300-800 tokens work well for most LLMs
2. **Enable BM25 prefiltering**: Reduces dense search time
3. **Use cross-encoder reranking**: Improves result quality
4. **Choose right FAISS index**: IVF+PQ for large datasets, Flat for small
5. **GPU acceleration**: Use `faiss-gpu` and CUDA-enabled models

## 🐛 Troubleshooting

**Index building fails:**
- Check data path in `config.yaml`
- Ensure write permissions for index directory
- Verify dependencies are installed

**No LLMs available:**
- Download model files to the specified path
- Install `llama-cpp-python` with proper compilation flags
- Check model compatibility with your system

**Poor search results:**
- Rebuild index with `--rebuild-index`
- Adjust chunk size and overlap settings
- Enable cross-encoder reranking

## 📄 License

[Your License Here]

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md] for guidelines.

---

**Built for efficient semantic search with multiple small LLMs** 🚀