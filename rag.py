# rag.py
from retriever import Retriever
from utils import setup_logger, load_config
from llama_cpp import Llama

cfg = load_config()
logger = setup_logger(cfg.get("LOG_LEVEL", "INFO"))

# --- LLM config (supports TinyLLaMA or Gemma 3 270M through llama.cpp) ---
LLM_MODEL_PATH = cfg.get("LLM_MODEL_PATH", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
LLM_CTX = int(cfg.get("LLM_CTX", 2048))
LLM_THREADS = int(cfg.get("LLM_THREADS", 4))
LLM_N_BATCH = int(cfg.get("LLM_N_BATCH", 256))
# LLM_FAMILY: "llama" (default) or "gemma"
LLM_FAMILY = cfg.get("LLM_FAMILY", "llama").lower()

# llama.cpp supports chat templates via chat_format
CHAT_FORMAT = None
if LLM_FAMILY == "gemma":
    CHAT_FORMAT = "gemma"

# Single global instance
_llm = None
def _get_llm():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=LLM_CTX,
            n_threads=LLM_THREADS,
            n_batch=LLM_N_BATCH,
            chat_format=CHAT_FORMAT  # None for llama-family; "gemma" for Gemma 3 270M
        )
    return _llm

def build_prompt(query, results):
    # results: list of dicts {path, text, chunk_id, score}
    contexts = []
    for r in results:
        contexts.append(f"FILE: {r['path']}\nCHUNK_ID: {r['chunk_id']}\n---\n{r['text']}\n")
    ctx = "\n\n".join(contexts)
    prompt = f"""You are a helpful assistant. Answer the question ONLY using the provided contexts and cite file paths.

CONTEXTS:
{ctx}

QUESTION:
{query}

Answer concisely and include citations like [file:path]."""
    return prompt

def answer_with_llm(prompt):
    llm = _get_llm()

    # Prefer chat-completions if available; otherwise, plain generate
    try:
        # llama_cpp >= 0.2.0
        res = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=float(cfg.get("LLM_TEMP", 0.3)),
            max_tokens=int(cfg.get("LLM_MAX_TOKENS", 512))
        )
        return res["choices"][0]["message"]["content"].strip()
    except Exception:
        # Fallback to raw completion call
        out = llm(
            prompt,
            temperature=float(cfg.get("LLM_TEMP", 0.3)),
            max_tokens=int(cfg.get("LLM_MAX_TOKENS", 512)),
        )
        # llama_cpp raw uses "choices"[0]["text"]
        return out["choices"][0].get("text", "").strip()
