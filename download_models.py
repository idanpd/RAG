#!/usr/bin/env python3
"""
Model download script for the semantic search system.
Downloads the recommended local models for CPU inference.
"""

import os
import sys
import requests
from pathlib import Path
from urllib.parse import urlparse
import hashlib
from typing import Dict, List
import argparse


# Model configurations with download URLs and checksums
MODELS = {
    'tinyllama-1.1b-chat': {
        'filename': 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
        'url': 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
        'size_mb': 669,
        'description': 'TinyLlama 1.1B - Fastest model, good for simple tasks',
        'sha256': None  # Add if available
    },
    'gemma-2b-instruct': {
        'filename': 'gemma-2b-it-q4_k_m.gguf',
        'url': 'https://huggingface.co/bartowski/gemma-2b-it-GGUF/resolve/main/gemma-2b-it-Q4_K_M.gguf',
        'size_mb': 1400,
        'description': 'Gemma 2B Instruct - Balanced performance and quality',
        'sha256': None
    },
    'phi3-mini-instruct': {
        'filename': 'phi-3-mini-4k-instruct-q4.gguf',
        'url': 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
        'size_mb': 2200,
        'description': 'Phi-3 Mini 3.8B - High quality reasoning',
        'sha256': None
    },
    'qwen2-0.5b-instruct': {
        'filename': 'qwen2-0_5b-instruct-q4_0.gguf',
        'url': 'https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_0.gguf',
        'size_mb': 400,
        'description': 'Qwen2 0.5B - Ultra-fast with long context',
        'sha256': None
    },
    'mistral-7b-instruct': {
        'filename': 'mistral-7b-instruct-v0.3.Q4_K_M.gguf',
        'url': 'https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf',
        'size_mb': 4100,
        'description': 'Mistral 7B - Highest quality, slower on CPU',
        'sha256': None
    }
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress indication."""
    try:
        print(f"Downloading from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded // (1024*1024):.1f}MB / {total_size // (1024*1024):.1f}MB)", end='', flush=True)
        
        print()  # New line after progress
        return True
        
    except Exception as e:
        print(f"\nError downloading file: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial file
        return False


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verify file checksum."""
    if not expected_sha256:
        return True  # Skip verification if no checksum provided
    
    print("Verifying checksum...")
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    actual_sha256 = sha256_hash.hexdigest()
    
    if actual_sha256 == expected_sha256:
        print("✅ Checksum verified")
        return True
    else:
        print(f"❌ Checksum mismatch!")
        print(f"Expected: {expected_sha256}")
        print(f"Actual:   {actual_sha256}")
        return False


def list_models():
    """List available models for download."""
    print("Available models for download:")
    print("=" * 60)
    
    for model_id, info in MODELS.items():
        size_gb = info['size_mb'] / 1024
        print(f"🤖 {model_id}")
        print(f"   File: {info['filename']}")
        print(f"   Size: {size_gb:.1f}GB")
        print(f"   Description: {info['description']}")
        print()


def download_model(model_id: str, models_dir: Path, force: bool = False) -> bool:
    """Download a specific model."""
    if model_id not in MODELS:
        print(f"❌ Unknown model: {model_id}")
        print("Available models:", list(MODELS.keys()))
        return False
    
    model_info = MODELS[model_id]
    dest_path = models_dir / model_info['filename']
    
    # Check if model already exists
    if dest_path.exists() and not force:
        print(f"✅ Model {model_id} already exists at {dest_path}")
        print("Use --force to re-download")
        return True
    
    # Create models directory
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading {model_id}...")
    print(f"Size: {model_info['size_mb']/1024:.1f}GB")
    print(f"Destination: {dest_path}")
    
    # Download the model
    success = download_file(model_info['url'], dest_path)
    
    if success:
        # Verify checksum if available
        if model_info.get('sha256'):
            if not verify_checksum(dest_path, model_info['sha256']):
                print("❌ Checksum verification failed")
                dest_path.unlink()
                return False
        
        print(f"✅ Successfully downloaded {model_id}")
        print(f"File: {dest_path}")
        return True
    else:
        print(f"❌ Failed to download {model_id}")
        return False


def download_recommended_models(models_dir: Path, force: bool = False) -> bool:
    """Download recommended models for getting started."""
    recommended = ['qwen2-0.5b-instruct', 'tinyllama-1.1b-chat']
    
    print("📦 Downloading recommended models for getting started...")
    print("This will download:")
    for model_id in recommended:
        info = MODELS[model_id]
        print(f"  - {model_id}: {info['description']} ({info['size_mb']/1024:.1f}GB)")
    
    total_size = sum(MODELS[model_id]['size_mb'] for model_id in recommended) / 1024
    print(f"\nTotal download size: {total_size:.1f}GB")
    
    if not force:
        response = input("\nProceed with download? (y/N): ").strip().lower()
        if response != 'y':
            print("Download cancelled.")
            return False
    
    success_count = 0
    for model_id in recommended:
        if download_model(model_id, models_dir, force):
            success_count += 1
        print()  # Add spacing between downloads
    
    if success_count == len(recommended):
        print(f"🎉 Successfully downloaded all {len(recommended)} recommended models!")
        return True
    else:
        print(f"⚠️  Downloaded {success_count}/{len(recommended)} models")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download local models for semantic search system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py --list                    # List available models
  python download_models.py --recommended             # Download recommended models
  python download_models.py --model tinyllama-1.1b-chat  # Download specific model
  python download_models.py --all                     # Download all models
        """
    )
    
    parser.add_argument('--models-dir', default='./models',
                       help='Directory to download models to (default: ./models)')
    parser.add_argument('--list', action='store_true',
                       help='List available models')
    parser.add_argument('--model', type=str,
                       help='Download specific model')
    parser.add_argument('--recommended', action='store_true',
                       help='Download recommended models for getting started')
    parser.add_argument('--all', action='store_true',
                       help='Download all available models')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if file exists')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    
    if args.list:
        list_models()
        return 0
    
    elif args.model:
        success = download_model(args.model, models_dir, args.force)
        return 0 if success else 1
    
    elif args.recommended:
        success = download_recommended_models(models_dir, args.force)
        return 0 if success else 1
    
    elif args.all:
        print("📦 Downloading all available models...")
        total_size = sum(info['size_mb'] for info in MODELS.values()) / 1024
        print(f"Total download size: {total_size:.1f}GB")
        
        if not args.force:
            response = input("\nThis will download a lot of data. Proceed? (y/N): ").strip().lower()
            if response != 'y':
                print("Download cancelled.")
                return 0
        
        success_count = 0
        for model_id in MODELS.keys():
            if download_model(model_id, models_dir, args.force):
                success_count += 1
            print()
        
        print(f"Downloaded {success_count}/{len(MODELS)} models")
        return 0 if success_count == len(MODELS) else 1
    
    else:
        print("No action specified. Use --help for usage information.")
        print("\nQuick start:")
        print("  python download_models.py --recommended")
        return 0


if __name__ == "__main__":
    sys.exit(main())