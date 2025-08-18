# Local Models API Guide

## 🚀 Fast CPU Inference with Local Models

This system now includes a comprehensive local model API optimized for fast CPU inference with the models from your original configuration. All components have been updated to support efficient local model management and inference.

## 📁 New Files Added

### Core API Files
- **`model_api.py`** - Complete local model management API
- **`setup_cpu.py`** - Automated CPU-optimized installation script
- **`download_models.py`** - Model download and management script
- **`LOCAL_MODELS_GUIDE.md`** - This guide

### Updated Files
- **`config.yaml`** - Added local model configurations and CPU optimizations
- **`rag.py`** - Updated with `LocalLLMManager` using the model API
- **`main.py`** - Added model management commands and CLI arguments
- **`requirements.txt`** - Updated with CPU-optimized dependencies
- **`README.md`** - Updated documentation for local models

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Local Model API                          │
├─────────────────────────────────────────────────────────────┤
│  LocalModelRegistry  │  LocalModelAPI  │  LlamaCppLocalModel │
│  - Model discovery   │  - High-level   │  - CPU optimized    │
│  - Loading/unloading │    interface    │  - Memory efficient │
│  - Performance stats │  - Error handling│  - Fast inference   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                 RAG System Integration                      │
├─────────────────────────────────────────────────────────────┤
│         LocalLLMManager (replaces old LLMManager)          │
│  - Multiple model support  │  - Dynamic model switching    │
│  - Memory management       │  - Performance monitoring     │
└─────────────────────────────────────────────────────────────┘
```

## 🤖 Supported Local Models

Based on your original config, the system supports these models:

| Model | Size | Speed | Quality | Context | Description |
|-------|------|-------|---------|---------|-------------|
| **Qwen2 0.5B** | 400MB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | 32k | Ultra-fast, long context |
| **TinyLlama 1.1B** | 600MB | ⚡⚡⚡⚡ | ⭐⭐⭐ | 2k | Fastest, good for simple tasks |
| **Gemma 2B** | 1.4GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | 8k | Balanced performance |
| **Phi-3 Mini 3.8B** | 2.2GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | 4k | High quality reasoning |
| **Mistral 7B** | 4.1GB | ⚡ | ⭐⭐⭐⭐⭐ | 32k | Highest quality |

## 🚀 Quick Setup

### 1. Automated Setup (Recommended)
```bash
# Install with CPU optimizations
python setup_cpu.py

# Download recommended models (1GB total)
python download_models.py --recommended

# Build index and start
python main.py --build-index
python main.py --interactive
```

### 2. Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download specific model
python download_models.py --model qwen2-0.5b-instruct

# Use the system
python main.py --models  # List available models
python main.py --interactive
```

## 💻 CPU Optimizations

The system includes several CPU optimizations:

### Model Loading Optimizations
- **Memory mapping** for faster loading
- **Multi-threading** with auto-detection of CPU cores
- **Larger batches** for CPU efficiency
- **Float16 key-value cache** for memory efficiency

### Configuration Options
```yaml
# CPU Optimization Settings
LLM_THREADS: 0           # 0 = auto-detect CPU cores
LLM_N_BATCH: 512         # Larger batch for CPU
USE_MMAP: true           # Memory mapping
USE_MLOCK: false         # Lock memory (high RAM usage)
NUMA: false              # NUMA optimization (multi-socket systems)
```

### Compile-time Optimizations
The setup script automatically detects and enables:
- **AVX/AVX2** support for faster math operations
- **FMA** (Fused Multiply-Add) instructions
- **OpenBLAS** integration where available
- **AVX512** on supported CPUs

## 🎮 Interactive Commands

### Model Management
```bash
models                    # List all available models with status
model: tinyllama-1.1b-chat # Load/switch to specific model
unload: gemma-2b-instruct # Unload model from memory
```

### Search & RAG
```bash
search query here         # Regular semantic search
rag: your question here   # AI-powered answer with current model
template: detailed        # Switch prompt template
```

### System Info
```bash
stats                     # Index statistics
quit                      # Exit system
```

## 🖥️ CLI Interface

### Model Management
```bash
# List available models
python main.py --models

# Load specific model
python main.py --load-model qwen2-0.5b-instruct

# Unload model
python main.py --unload-model tinyllama-1.1b-chat

# Use specific model for Q&A
python main.py --ask "your question" --llm phi3-mini-instruct
```

### Performance Testing
```bash
# Test different models
python main.py --load-model qwen2-0.5b-instruct
python main.py --ask "What is AI?" --llm qwen2-0.5b-instruct

python main.py --load-model tinyllama-1.1b-chat  
python main.py --ask "What is AI?" --llm tinyllama-1.1b-chat
```

## 📊 Performance Monitoring

The system automatically tracks:
- **Model load times**
- **Inference speed** (tokens/second)
- **Memory usage** per model
- **Active/loaded models** status

View performance stats:
```bash
# In interactive mode
models

# Or via CLI
python main.py --models
```

## 🔧 Advanced Configuration

### Model Selection Strategy
```python
# In code, you can access the model API directly
from model_api import LocalModelAPI

api = LocalModelAPI()
models = api.list_available_models()

# Load fastest model
fastest = min(models, key=lambda m: m['size_gb'])
api.load_model(fastest['name'])
```

### Custom Model Addition
Add new models to `model_api.py`:
```python
{
    'name': 'your-custom-model',
    'family': 'llama',  # or gemma, phi, qwen, mistral
    'filename': 'your-model.gguf',
    'size_gb': 1.0,
    'context_length': 2048,
    'description': 'Your custom model description',
    'chat_template': None  # or 'gemma', 'phi-3', etc.
}
```

## 🚨 Troubleshooting

### Model Not Loading
```bash
# Check if model file exists
ls -la models/

# Download missing models
python download_models.py --list
python download_models.py --model model-name
```

### Slow Performance
```bash
# Check CPU features
python setup_cpu.py  # Re-run to see CPU capabilities

# Try smaller model
python main.py --load-model qwen2-0.5b-instruct

# Check model stats
python main.py --models
```

### Memory Issues
```bash
# Unload unused models
python main.py --unload-model large-model-name

# Use smaller models
python download_models.py --model qwen2-0.5b-instruct
```

## 🎯 Performance Expectations

On a modern CPU (8 cores, 16GB RAM):

| Model | Load Time | Inference Speed | Memory Usage |
|-------|-----------|----------------|--------------|
| Qwen2 0.5B | ~2s | 15-25 tok/s | ~1GB |
| TinyLlama 1.1B | ~3s | 10-20 tok/s | ~1.5GB |
| Gemma 2B | ~5s | 8-15 tok/s | ~2.5GB |
| Phi-3 Mini 3.8B | ~8s | 5-12 tok/s | ~4GB |
| Mistral 7B | ~15s | 3-8 tok/s | ~6GB |

## 🔄 Migration from Original Code

Your original code still works unchanged:
```python
# Legacy interface (still supported)
from rag import build_prompt, answer_with_llm

# New interface (recommended)
from model_api import LocalModelAPI
from rag import LocalLLMManager
```

## 🎉 Benefits of the New System

1. **🚀 Faster CPU Inference** - Optimized compilation and runtime settings
2. **🔧 Easy Model Management** - Load/unload models dynamically
3. **📊 Performance Monitoring** - Real-time stats and benchmarking
4. **💾 Memory Efficient** - Smart memory management and cleanup
5. **🎯 Model Selection** - Choose the right model for your task
6. **🔄 Hot Swapping** - Switch models without restarting
7. **📱 User Friendly** - Simple commands and clear status messages

The system is now fully optimized for local CPU inference with your specific models while maintaining all the original functionality!