# utils.py
import os, json, hashlib, logging
from pathlib import Path
import yaml

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def file_sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def setup_logger(level="INFO"):
    logger = logging.getLogger("rag")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch = logging.StreamHandler()
    fmt = logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger
