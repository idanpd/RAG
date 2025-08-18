# main.py
from utils import load_config, setup_logger
from indexer import Indexer
from retriever import Retriever
from rag import build_prompt, answer_with_llm

cfg = load_config()
logger = setup_logger(cfg.get("LOG_LEVEL", "INFO"))

def main():
    data_path = cfg.get("DATA_PATH", "./data")
    index_path = cfg.get("INDEX_PATH", "./index/faiss.index")
    metadata_path = cfg.get("METADATA_PATH", "./index/metadata.json")

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

if __name__ == "__main__":
    main()
