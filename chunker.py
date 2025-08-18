# chunker.py
class Chunker:
    def __init__(self, chunk_size=800, overlap=120):
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)

    def chunk_text(self, text: str):
        if not text:
            return []
        words = text.split()
        chunks, i = [], 0
        step = max(1, self.chunk_size - self.overlap)
        while i < len(words):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += step
        return chunks
