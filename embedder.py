# embedder.py
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts):
        # returns np.float32 [N, D]
        embs = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return np.array(embs, dtype="float32")
