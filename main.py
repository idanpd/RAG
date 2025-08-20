#!/usr/bin/env python3
"""
Multi-modal semantic search system with RAG capabilities.

This is the main entry point for the semantic search system that can process
text, images, and videos, index them efficiently, and provide semantic search
with LLM-powered answers.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from utils import ConfigManager, setup_logger
from indexer import SemanticIndexer
from retriever import SemanticRetriever
from rag import RAGSystem


class SemanticSearchSystem:
    """Main system class that orchestrates indexing, retrieval, and RAG."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the semantic search system."""
        self.config = ConfigManager(config_path)
        self.logger = setup_logger(self.config.get('LOG_LEVEL', 'INFO'))
        
        # Initialize components
        self.indexer: Optional[SemanticIndexer] = None
        self.retriever: Optional[SemanticRetriever] = None
        self.rag_system: Optional[RAGSystem] = None
        
        self.logger.info("Semantic Search System initialized")
    
    def initialize_indexer(self) -> SemanticIndexer:
        """Initialize the indexer component."""
        if self.indexer is None:
            self.indexer = SemanticIndexer(self.config)
        return self.indexer
    
    def initialize_retriever(self) -> SemanticRetriever:
        """Initialize the retriever component."""
        if self.retriever is None:
            self.retriever = SemanticRetriever(self.config)
        return self.retriever
    
    def initialize_rag(self) -> RAGSystem:
        """Initialize the RAG system."""
        if self.rag_system is None:
            retriever = self.initialize_retriever()
            self.rag_system = RAGSystem(retriever, self.config)
        return self.rag_system
    
    def build_index(self, rebuild: bool = False) -> bool:
        """Build or rebuild the search index."""
        indexer = self.initialize_indexer()
        
        if rebuild:
            self.logger.info("Rebuilding index from scratch...")
        else:
            self.logger.info("Building/updating index...")
        
        success = indexer.build_index(rebuild=rebuild)
        
        if success:
            stats = indexer.get_stats()
            self.logger.info(f"Index built successfully:")
            self.logger.info(f"  - Files: {stats['total_files']}")
            self.logger.info(f"  - Chunks: {stats['total_chunks']}")
            self.logger.info(f"  - Avg tokens per chunk: {stats['average_tokens_per_chunk']}")
        else:
            self.logger.error("Index building failed")
        
        return success
    
    def search(self, query: str, top_k: int = 5) -> list:
        """Perform semantic search."""
        retriever = self.initialize_retriever()
        results = retriever.search(query, top_k)
        
        self.logger.info(f"Found {len(results)} results for query: '{query}'")
        return results
    
    def ask_question(self, query: str, template: str = 'default', 
                    llm_name: Optional[str] = None, top_k: int = 5) -> dict:
        """Ask a question and get an AI-powered answer."""
        rag_system = self.initialize_rag()
        result = rag_system.answer_query(
            query=query,
            template=template,
            llm_name=llm_name,
            top_k=top_k
        )
        
        self.logger.info(f"Generated answer using {result['llm_used']} with template '{result['template_used']}'")
        return result
    
    def interactive_mode(self):
        """Run in interactive mode."""
        print("\n🔍 Semantic Search System - Interactive Mode")
        print("=" * 50)
        
        # Check if index exists
        try:
            retriever = self.initialize_retriever()
            if retriever.dense_retriever.index is None:
                print("\n⚠️  No search index found. Please build the index first.")
                return
        except Exception as e:
            print(f"\n❌ Error initializing retriever: {e}")
            return
        
        # Initialize RAG system
        try:
            rag_system = self.initialize_rag()
            available_llms = rag_system.get_available_llms()
            available_templates = rag_system.get_available_templates()
            
            if not available_llms:
                print("\n⚠️  No local models available. Search-only mode.")
                print("💡 Download models to ./models/ directory to enable AI answers")
                use_rag = False
            else:
                print(f"\n✅ Available models: {', '.join(available_llms)}")
                print(f"✅ Available templates: {', '.join(available_templates)}")
                
                # Show model stats
                model_stats = rag_system.llm_manager.get_model_stats()
                if model_stats['loaded_models'] > 0:
                    print(f"🔥 Loaded models: {model_stats['loaded_models']}/{model_stats['total_models']}")
                use_rag = True
        except Exception as e:
            print(f"\n⚠️  RAG system unavailable: {e}")
            use_rag = False
        
        print("\nCommands:")
        print("  - Type your question to search")
        if use_rag:
            print("  - Use 'rag: <question>' for AI-powered answers")
            print("  - Use 'model: <model_name>' to switch/load model")
            print("  - Use 'unload: <model_name>' to unload model")
            print("  - Use 'models' to list available models")
            print("  - Use 'template: <template_name>' to switch template")
        print("  - Type 'stats' to see index statistics")
        print("  - Type 'quit' or 'exit' to quit")
        print()
        
        current_template = 'default'
        
        while True:
            try:
                user_input = input("🔍 Query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                elif user_input.lower() == 'stats':
                    self._show_stats()
                
                elif user_input.lower() == 'models' and use_rag:
                    self._show_models(rag_system)
                
                elif user_input.startswith('model:') and use_rag:
                    model_name = user_input[6:].strip()
                    print(f"🔄 Loading model: {model_name}...")
                    if rag_system.set_active_llm(model_name):
                        print(f"✅ Switched to model: {model_name}")
                    else:
                        print(f"❌ Model not available or failed to load: {model_name}")
                
                elif user_input.startswith('unload:') and use_rag:
                    model_name = user_input[7:].strip()
                    result = rag_system.llm_manager.unload_model(model_name)
                    if result['success']:
                        print(f"✅ {result['message']}")
                    else:
                        print(f"❌ Failed to unload model: {model_name}")
                
                elif user_input.startswith('llm:') and use_rag:
                    # Keep for backward compatibility
                    model_name = user_input[4:].strip()
                    print(f"🔄 Loading model: {model_name}...")
                    if rag_system.set_active_llm(model_name):
                        print(f"✅ Switched to model: {model_name}")
                    else:
                        print(f"❌ Model not available or failed to load: {model_name}")
                
                elif user_input.startswith('template:') and use_rag:
                    template_name = user_input[9:].strip()
                    if template_name in available_templates:
                        current_template = template_name
                        print(f"✅ Switched to template: {template_name}")
                    else:
                        print(f"❌ Template not available: {template_name}")
                
                elif user_input.startswith('rag:') and use_rag:
                    query = user_input[4:].strip()
                    if query:
                        print(f"\n🤖 Generating answer...")
                        result = self.ask_question(query, current_template)
                        self._display_rag_result(result)
                    else:
                        print("❌ Please provide a question after 'rag:'")
                
                else:
                    # Regular search
                    print(f"\n🔍 Searching...")
                    results = self.search(user_input)
                    self._display_search_results(results)
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_stats(self):
        """Show index statistics."""
        try:
            indexer = self.initialize_indexer()
            stats = indexer.get_stats()
            
            print("\n📊 Index Statistics:")
            print(f"  Total files: {stats['total_files']}")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Average tokens per chunk: {stats['average_tokens_per_chunk']}")
            print(f"  Embedding dimension: {stats['embedding_dimension']}")
            
            if stats['files_by_type']:
                print("\n  Files by type:")
                for file_type, count in stats['files_by_type'].items():
                    print(f"    {file_type}: {count}")
            
            if stats['chunks_by_type']:
                print("\n  Chunks by type:")
                for chunk_type, count in stats['chunks_by_type'].items():
                    print(f"    {chunk_type}: {count}")
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def _show_models(self, rag_system):
        """Show available models and their status."""
        try:
            model_stats = rag_system.llm_manager.get_model_stats()
            available_models = rag_system.llm_manager.get_available_models()
            
            print("\n🤖 Available Local Models:")
            print("=" * 60)
            
            for model in available_models:
                status_icon = "🔥" if model['is_loaded'] else "💤"
                size_info = f"({model['size_gb']:.1f}GB)"
                context_info = f"ctx:{model['context_length']//1024}k" if model['context_length'] >= 1024 else f"ctx:{model['context_length']}"
                
                print(f"{status_icon} {model['name']} {size_info}")
                print(f"    Family: {model['family']} | {context_info}")
                print(f"    {model['description']}")
                print()
            
            # Show loaded model performance
            if model_stats['loaded_model_info']:
                print("📈 Loaded Model Performance:")
                for info in model_stats['loaded_model_info']:
                    load_time = info['load_time']
                    speed = info['inference_speed']
                    print(f"  {info['name']}: Load {load_time:.1f}s | Speed {speed:.1f} tok/s")
            
            print(f"Total: {model_stats['total_models']} available, {model_stats['loaded_models']} loaded")
            
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
    
    def _display_search_results(self, results):
        """Display search results."""
        if not results:
            print("No results found.")
            return
        
        print(f"\n📄 Found {len(results)} results:")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            if hasattr(result, 'to_dict'):
                result = result.to_dict()
            
            # Handle scoring - cross_score takes precedence, then distance
            if 'cross_score' in result and result['cross_score'] is not None:
                score = result['cross_score']
                score_type = "Cross-Encoder"
            else:
                score = result.get('score', 0)
                score_type = "Distance"
            
            path = result.get('path', 'Unknown')
            summary = result.get('summary', '')
            chunk_type = result.get('chunk_type', 'content')
            
            print(f"{i}. [{chunk_type.upper()}] {Path(path).name}")
            print(f"   {score_type}: {score:.4f}")
            print(f"   Path: {path}")
            if summary:
                print(f"   Summary: {summary}")
            print()
    
    def _display_rag_result(self, result):
        """Display RAG result."""
        print("\n" + "=" * 50)
        print("🤖 AI ANSWER")
        print("=" * 50)
        print(result['answer'])
        
        if result['sources']:
            print(f"\n📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                score = source.get('score', 0)
                path = source.get('path', 'Unknown')
                print(f"  {i}. {Path(path).name} (score: {score:.4f})")
        
        print(f"\n🔧 Generated using: {result['llm_used']} | Template: {result['template_used']}")
        print("=" * 50)
    
    def cleanup(self):
        """Clean up resources."""
        if self.indexer:
            self.indexer.close()
        if self.retriever:
            self.retriever.close()
        if self.rag_system:
            self.rag_system.llm_manager.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-modal Semantic Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --build-index                    # Build search index
  python main.py --rebuild-index                  # Rebuild from scratch
  python main.py --interactive                    # Interactive mode
  python main.py --search "your query"            # One-time search
  python main.py --ask "your question"            # One-time Q&A
  python main.py --models                         # List available models
  python main.py --load-model tinyllama-1.1b-chat # Load specific model
  python main.py --ask "question" --llm phi3-mini-instruct # Use specific model
        """
    )
    
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--build-index', action='store_true',
                       help='Build the search index')
    parser.add_argument('--rebuild-index', action='store_true',
                       help='Rebuild the search index from scratch')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--search', '-s', type=str,
                       help='Perform a one-time search')
    parser.add_argument('--ask', '-a', type=str,
                       help='Ask a question and get AI answer')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                       help='Number of results to return (default: 5)')
    parser.add_argument('--template', '-t', default='default',
                       help='RAG template to use (default: default)')
    parser.add_argument('--llm', '-l', type=str,
                       help='LLM to use for answers')
    parser.add_argument('--stats', action='store_true',
                       help='Show index statistics')
    parser.add_argument('--models', action='store_true',
                       help='List available local models')
    parser.add_argument('--load-model', type=str,
                       help='Load a specific model')
    parser.add_argument('--unload-model', type=str,
                       help='Unload a specific model')
    
    args = parser.parse_args()
    
    # Initialize system
    try:
        system = SemanticSearchSystem(args.config)
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return 1
    
    try:
        # Handle different modes
        if args.build_index or args.rebuild_index:
            success = system.build_index(rebuild=args.rebuild_index)
            return 0 if success else 1
        
        elif args.stats:
            system._show_stats()
            return 0
        
        elif args.models:
            try:
                rag_system = system.initialize_rag()
                system._show_models(rag_system)
            except Exception as e:
                print(f"❌ Error listing models: {e}")
                return 1
            return 0
        
        elif args.load_model:
            try:
                rag_system = system.initialize_rag()
                print(f"🔄 Loading model: {args.load_model}...")
                result = rag_system.llm_manager.load_model(args.load_model)
                if result['success']:
                    print(f"✅ {result['message']} (Load time: {result['load_time']:.1f}s)")
                else:
                    print(f"❌ Failed to load model: {args.load_model}")
                    return 1
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return 1
            return 0
        
        elif args.unload_model:
            try:
                rag_system = system.initialize_rag()
                result = rag_system.llm_manager.unload_model(args.unload_model)
                if result['success']:
                    print(f"✅ {result['message']}")
                else:
                    print(f"❌ Failed to unload model: {args.unload_model}")
                    return 1
            except Exception as e:
                print(f"❌ Error unloading model: {e}")
                return 1
            return 0
        
        elif args.search:
            results = system.search(args.search, args.top_k)
            system._display_search_results(results)
            return 0
        
        elif args.ask:
            result = system.ask_question(
                args.ask, 
                template=args.template,
                llm_name=args.llm,
                top_k=args.top_k
            )
            system._display_rag_result(result)
            return 0
        
        elif args.interactive:
            system.interactive_mode()
            return 0
        
        else:
            # Default to interactive if no specific action
            print("No action specified. Starting interactive mode...")
            system.interactive_mode()
            return 0
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        system.cleanup()


# Legacy compatibility - maintain the original main function interface
if __name__ == "__main__":
    # Check if being called in legacy mode
    if len(sys.argv) == 1:
        # Original legacy behavior
        from utils import load_config, setup_logger
        from indexer import Indexer
        from retriever import Retriever
        from rag import build_prompt, answer_with_llm
        
        cfg = load_config()
        logger = setup_logger(cfg.get("LOG_LEVEL", "INFO"))
        
        def legacy_main():
            data_path = cfg.get("DATA_PATH", "./data")
            
            # Step 1: Build or update index
            indexer = Indexer(data_path)    
            if cfg.get("REBUILD_INDEX", True):
                logger.info("Rebuilding index from scratch...")
                indexer.index_all()
            else:
                logger.info("Loading existing index...")
            
            # Step 2: Search
            retriever = Retriever()
            query = input("Enter your search query: ")
            results = retriever.search(query)
            
            # Step 3: Build RAG prompt
            prompt = build_prompt(query, results)
            
            # Step 4: Answer using LLM
            answer = answer_with_llm(prompt)
            
            # Step 5: Show answer
            print("\n===== ANSWER =====\n")
            print(answer)
        
        legacy_main()
    else:
        # New CLI interface
        sys.exit(main())