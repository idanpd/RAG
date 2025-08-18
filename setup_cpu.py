#!/usr/bin/env python3
"""
Setup script for CPU-optimized semantic search system.
Installs dependencies with CPU optimizations for fast local inference.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    if description:
        print(f"📦 {description}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False, e.stderr


def detect_cpu_features():
    """Detect CPU features for optimization."""
    features = {
        'avx': False,
        'avx2': False,
        'avx512': False,
        'fma': False
    }
    
    try:
        if platform.system() == "Linux":
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'avx' in cpuinfo:
                    features['avx'] = True
                if 'avx2' in cpuinfo:
                    features['avx2'] = True
                if 'avx512' in cpuinfo:
                    features['avx512'] = True
                if 'fma' in cpuinfo:
                    features['fma'] = True
        elif platform.system() == "Darwin":  # macOS
            # On macOS, assume modern CPU features
            features['avx'] = True
            features['avx2'] = True
            features['fma'] = True
        elif platform.system() == "Windows":
            # On Windows, assume modern CPU features
            features['avx'] = True
            features['avx2'] = True
            features['fma'] = True
    except:
        pass
    
    return features


def install_llama_cpp_optimized():
    """Install llama-cpp-python with CPU optimizations."""
    print("🚀 Installing llama-cpp-python with CPU optimizations...")
    
    # Detect CPU features
    cpu_features = detect_cpu_features()
    
    # Set environment variables for CPU optimization
    env_vars = {
        'CMAKE_ARGS': '-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS',
        'FORCE_CMAKE': '1'
    }
    
    if cpu_features['avx2']:
        env_vars['CMAKE_ARGS'] += ' -DLLAMA_AVX2=ON'
        print("✅ Enabling AVX2 support")
    
    if cpu_features['fma']:
        env_vars['CMAKE_ARGS'] += ' -DLLAMA_FMA=ON'
        print("✅ Enabling FMA support")
    
    if cpu_features['avx512']:
        env_vars['CMAKE_ARGS'] += ' -DLLAMA_AVX512=ON'
        print("✅ Enabling AVX512 support")
    
    # Try to use OpenBLAS for better performance
    if platform.system() == "Linux":
        # Check if OpenBLAS is available
        success, _ = run_command("pkg-config --exists openblas")
        if success:
            env_vars['CMAKE_ARGS'] += ' -DLLAMA_OPENBLAS=ON'
            print("✅ Enabling OpenBLAS support")
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Setting {key}={value}")
    
    # Install llama-cpp-python
    cmd = f"{sys.executable} -m pip install llama-cpp-python --force-reinstall --no-cache-dir"
    success, output = run_command(cmd, "Installing optimized llama-cpp-python")
    
    if success:
        print("✅ llama-cpp-python installed with CPU optimizations")
    else:
        print("⚠️  Failed to install optimized version, trying standard installation...")
        cmd = f"{sys.executable} -m pip install llama-cpp-python"
        success, _ = run_command(cmd)
        if success:
            print("✅ Standard llama-cpp-python installed")
        else:
            print("❌ Failed to install llama-cpp-python")
            return False
    
    return True


def install_requirements():
    """Install other requirements."""
    print("📦 Installing other requirements...")
    
    # Install CPU-only PyTorch first
    torch_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    success, _ = run_command(torch_cmd, "Installing CPU-only PyTorch")
    
    if not success:
        print("⚠️  Failed to install CPU-only PyTorch, trying standard version...")
        torch_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio"
        success, _ = run_command(torch_cmd)
    
    # Install other requirements (excluding torch and llama-cpp-python)
    requirements_cmd = f"{sys.executable} -m pip install numpy pandas PyYAML sentence-transformers faiss-cpu transformers rank-bm25 PyMuPDF python-docx python-pptx openpyxl Pillow pytesseract opencv-python requests tiktoken"
    success, _ = run_command(requirements_cmd, "Installing other requirements")
    
    return success


def check_system_dependencies():
    """Check for system dependencies."""
    print("🔍 Checking system dependencies...")
    
    # Check for tesseract
    success, _ = run_command("tesseract --version")
    if success:
        print("✅ Tesseract OCR found")
    else:
        print("⚠️  Tesseract OCR not found. Install with:")
        if platform.system() == "Linux":
            print("  sudo apt-get install tesseract-ocr")
        elif platform.system() == "Darwin":
            print("  brew install tesseract")
        elif platform.system() == "Windows":
            print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    # Check for ffmpeg (optional)
    success, _ = run_command("ffmpeg -version")
    if success:
        print("✅ FFmpeg found")
    else:
        print("ℹ️  FFmpeg not found (optional for video processing)")
    
    return True


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = ['./data', './indices', './models']
    
    for dir_path in directories:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_path}")


def main():
    """Main setup function."""
    print("🚀 Setting up CPU-optimized semantic search system...")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return 1
    
    print(f"✅ Python {sys.version}")
    print(f"✅ Platform: {platform.system()} {platform.machine()}")
    
    # Detect CPU features
    cpu_features = detect_cpu_features()
    print(f"🔧 CPU Features: {', '.join(k for k, v in cpu_features.items() if v)}")
    
    # Create directories
    create_directories()
    
    # Check system dependencies
    check_system_dependencies()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return 1
    
    # Install optimized llama-cpp-python
    if not install_llama_cpp_optimized():
        print("❌ Failed to install llama-cpp-python")
        return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Download models:")
    print("   python download_models.py --recommended")
    print("\n2. Build search index:")
    print("   python main.py --build-index")
    print("\n3. Start using the system:")
    print("   python main.py --interactive")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())