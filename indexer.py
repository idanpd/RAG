# indexer.py
import os
import faiss
import sqlite3
import numpy as np
from pathlib import Path
from utils import setup_logger, load_config, ensure_dir, file_sha256
from sentence_transformers import SentenceTransformer
from chunker import Chunker
from extractors.basic import extract_text  # existing
from extractors.images import extract_text_from_image, IMAGE_EXTS, caption_image_description  # new caption option
from extractors.video_extractor import (
    VIDEO_EXTS,
    extract_video_transcript_optional,
    extract_video_keyframe_texts_optional
)

cfg = load_config()
logger = setup_logger(cfg.get("LOG_LEVEL", "INFO"))

DB_PATH = cfg.get("SQLITE_DB", "index.db")
INDEX_DIR = Path(cfg.get("INDEX_DIR", "./indices"))
ensure_dir(INDEX_DIR)

EMBED_MODEL = cfg.get("EMBED_MODEL", "all-MiniLM-L6-v2")
FAISS_NLIST = int(cfg.get("FAISS_NLIST", 100))
FAISS_PQ_M = int(cfg.get("FAISS_PQ_M", 8))

# Audio handled as "other" (no ASR unless you add it separately)
AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'}
DOC_EXTS = {'.pdf', '.docx', '.doc', '.txt', '.rtf', '.odt'}
SPREADSHEET_EXTS = {'.xlsx', '.xls', '.csv', '.ods'}
PRESENTATION_EXTS = {'.pptx', '.ppt', '.odp'}

def get_file_type(path: Path):
    ext = path.suffix.lower()
    if ext in VIDEO_EXTS:
        return 'video'
    elif ext in AUDIO_EXTS:
        return 'audio'
    elif ext in IMAGE_EXTS:
        return 'image'
    elif ext in DOC_EXTS:
        return 'document'
    elif ext in SPREADSHEET_EXTS:
        return 'spreadsheet'
    elif ext in PRESENTATION_EXTS:
        return 'presentation'
    else:
        return 'other'

class Indexer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.chunker = Chunker(
            chunk_size=int(cfg.get("CHUNK_SIZE", 800)),
            overlap=int(cfg.get("CHUNK_OVERLAP", 120))
        )
        self.conn = sqlite3.connect(DB_PATH)
        self._ensure_tables()

    def _ensure_tables(self):
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            filename TEXT,
            file_extension TEXT,
            file_type TEXT,
            sha256 TEXT,
            size INTEGER
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            file_id INTEGER,
            chunk_text TEXT,
            summary TEXT,
            chunk_type TEXT,
            prev_id INTEGER,
            next_id INTEGER,
            emb_id INTEGER
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY,
            query TEXT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        self.conn.commit()

    def _create_file_metadata_chunk(self, file_path: str, file_type: str):
        p = Path(file_path)
        meta = [
            f"File: {p.name}",
            f"Type: {file_type}",
            f"Extension: {p.suffix}",
            f"Directory: {p.parent.name}",
            f"Full path: {file_path}",
            f"Summary: this is a {file_type} named {p.name} located at {p.parent}"
        ]
        # lightweight keywords to help discovery
        if file_type == 'video':
            meta.append("Keywords: video file, movie, clip, recording, media, visual content, footage")
        elif file_type == 'image':
            meta.append("Keywords: image file, picture, photo, graphic, visual, screenshot")
        elif file_type == 'audio':
            meta.append("Keywords: audio file, music, sound, recording, voice, song")
        elif file_type == 'document':
            meta.append("Keywords: document, text file, pdf, word document, report, paper")
        return "\n".join(meta)

    def delete_index(self):
        # remove faiss index files in INDEX_DIR and clear sqlite tables
        try:
            for f in INDEX_DIR.glob("*.index"):
                f.unlink(missing_ok=True)
            for f in INDEX_DIR.glob("*.faiss"):
                f.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Failed deleting FAISS files: {e}")

        try:
            c = self.conn.cursor()
            c.execute("DELETE FROM chunks")
            c.execute("DELETE FROM files")
            self.conn.commit()
            logger.info("Cleared SQLite metadata.")
        except Exception as e:
            logger.error(f"Failed clearing SQLite: {e}")

    def index_all(self):
        # Find files
        files = [p for p in self.data_path.rglob("*") if p.is_file()]
        if not files:
            logger.info("No files found.")
            return

        chunk_texts = []
        chunk_meta = []   # tuples: (file_id, summary, prev_id, chunk_type)
        file_id_map = {}

        ENABLE_OCR = bool(cfg.get("USE_OCR", True))
        ENABLE_CAPTION = bool(cfg.get("USE_IMAGE_CAPTION", False))
        ENABLE_VID_TRANSCRIPT = bool(cfg.get("USE_VIDEO_TRANSCRIPT", False))
        ENABLE_VID_KEYFRAME = bool(cfg.get("USE_VIDEO_KEYFRAME_OCR", False))

        for p in files:
            try:
                ftype = get_file_type(p)
                suf = p.suffix.lower()

                # Extract text content based on type
                content_text = ""
                if ftype == 'image':
                    # OCR
                    if ENABLE_OCR:
                        content_text = extract_text_from_image(p)
                    # Caption (optional tiny model on CPU)
                    if ENABLE_CAPTION:
                        cap = caption_image_description(p)
                        if cap:
                            content_text = (content_text + "\n" + cap).strip()

                elif ftype == 'video':
                    # transcript (optional)
                    if ENABLE_VID_TRANSCRIPT:
                        t = extract_video_transcript_optional(p)
                        if t:
                            content_text += ("\n" + t if content_text else t)
                    # keyframe OCR (optional)
                    if ENABLE_VID_KEYFRAME:
                        ktxt = extract_video_keyframe_texts_optional(p)
                        if ktxt:
                            content_text += ("\n" + ktxt if content_text else ktxt)

                else:
                    # documents/spreadsheets/presentations/others → text extractor
                    content_text = extract_text(p)

                # Store files entry
                sha = file_sha256(p)
                size = p.stat().st_size
                c = self.conn.cursor()
                c.execute("""INSERT OR REPLACE INTO files
                             (path, filename, file_extension, file_type, sha256, size)
                             VALUES (?,?,?,?,?,?)""",
                          (str(p), p.name, suf, ftype, sha, size))
                self.conn.commit()
                c.execute("SELECT id FROM files WHERE path=?", (str(p),))
                file_id = c.fetchone()[0]
                file_id_map[str(p)] = file_id

                # Always add metadata chunk for discovery
                metadata_chunk = self._create_file_metadata_chunk(str(p), ftype)
                chunk_texts.append(metadata_chunk)
                chunk_meta.append((file_id, f"Metadata for {p.name}", None, 'metadata'))

                # Add content chunks if any
                if content_text and content_text.strip():
                    chunks = self.chunker.chunk_text(content_text)
                    prev_sql_id = None
                    for ch in chunks:
                        if not ch.strip():
                            continue
                        summary = (ch.split(".")[0] + ".")[:300] if "." in ch else ch[:300]
                        chunk_texts.append(ch)
                        # we don't know SQL row id before insert; set prev later after insert
                        chunk_meta.append((file_id, summary, prev_sql_id, 'content'))
                        # SQL id linking for prev/next is set after we insert below

            except Exception as e:
                logger.error(f"Failed processing {p}: {e}")

        if not chunk_texts:
            logger.info("No chunk texts collected.")
            return

        # Build embeddings
        logger.info(f"Embedding {len(chunk_texts)} chunks with {EMBED_MODEL}")
        embeddings = self.embedder.encode(chunk_texts)  # returns float32 np.ndarray

        # Build FAISS index with safe small-dataset fallback
        index_path = INDEX_DIR / "chunks_ivfpq.index"
        if index_path.exists():
            try:
                index_path.unlink()
            except Exception as e:
                logger.error(f"Failed removing existing index: {e}")

        num_vecs = embeddings.shape[0]
        if num_vecs < 100:  # too small for IVF training → use flat
            logger.info(f"Small dataset ({num_vecs}) → using IndexFlatL2")
            index = faiss.IndexFlatL2(self.dim)
            index.add(embeddings)
        else:
            # pick nlist & m safely
            nlist = max(8, min(int(np.sqrt(num_vecs)), FAISS_NLIST, num_vecs // 4))
            # choose m that divides the dim or fallback to nearest divisor
            m = min(FAISS_PQ_M, max(2, self.dim // 8))
            while m > 1 and self.dim % m != 0:
                m -= 1
            if m <= 1 or self.dim % m != 0:
                m = 8 if self.dim >= 64 else max(2, self.dim // 4)

            logger.info(f"Using IVF+PQ nlist={nlist}, m={m}")
            quant = faiss.IndexFlatL2(self.dim)
            index = faiss.IndexIVFPQ(quant, self.dim, nlist, m, 8)
            logger.info("Training IVF+PQ…")
            index.train(embeddings)
            index.add(embeddings)

        faiss.write_index(index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")

        # Insert chunk rows (and link prev/next)
        c = self.conn.cursor()
        row_ids = []
        for i, meta in enumerate(chunk_meta):
            file_id, summary, prev_id, chunk_type = meta
            c.execute("""INSERT INTO chunks (file_id, chunk_text, summary, chunk_type, prev_id, emb_id)
                         VALUES (?,?,?,?,?,?)""",
                      (file_id, chunk_texts[i], summary, chunk_type, None, int(i)))
            row_ids.append(c.lastrowid)

        # now set prev/next using row_ids order per file
        # We approximate prev/next by consecutive rows per same file_id
        # (good enough for contextual navigation)
        last_for_file = {}
        for sql_id, meta in zip(row_ids, chunk_meta):
            file_id = meta[0]
            if file_id in last_for_file:
                # set prev of current to last, and next of last to current
                c.execute("UPDATE chunks SET prev_id=? WHERE id=?", (last_for_file[file_id], sql_id))
                c.execute("UPDATE chunks SET next_id=? WHERE id=?", (sql_id, last_for_file[file_id]))
            last_for_file[file_id] = sql_id

        self.conn.commit()
        logger.info(f"Indexed {len(files)} files with {len(chunk_texts)} chunks.")
