# extractors/video.py
from pathlib import Path
import subprocess
import tempfile
from utils import load_config, setup_logger
from extractors.images import extract_text_from_image, caption_image_description
import os

cfg = load_config()
logger = setup_logger(cfg.get("LOG_LEVEL", "INFO"))

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}

def _ffmpeg_exists():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def extract_video_transcript_optional(path: Path) -> str:
    """Use whisper.cpp CLI if available (optional).
       Configure cfg['WHISPER_CPP_BIN'] and cfg['WHISPER_MODEL'] if you have them.
       Otherwise returns '' (non-fatal)."""
    if not bool(cfg.get("USE_VIDEO_TRANSCRIPT", False)):
        return ""
    whisper_bin = cfg.get("WHISPER_CPP_BIN")  # e.g., "./whisper.cpp/main"
    whisper_model = cfg.get("WHISPER_MODEL")  # e.g., "ggml-base.en.bin"
    if not whisper_bin or not whisper_model or not os.path.exists(whisper_bin) or not os.path.exists(whisper_model):
        logger.info("Whisper disabled or missing; skipping transcript.")
        return ""
    # Extract audio then run whisper
    if not _ffmpeg_exists():
        logger.info("ffmpeg not found; skipping transcript.")
        return ""
    with tempfile.TemporaryDirectory() as td:
        wav_path = str(Path(td) / "audio.wav")
        # extract audio
        try:
            subprocess.run(["ffmpeg", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", wav_path],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.error(f"ffmpeg audio extract failed: {e}")
            return ""
        # whisper.cpp
        try:
            res = subprocess.run([whisper_bin, "-m", whisper_model, "-f", wav_path, "-otxt"],
                                 check=True, capture_output=True, text=True)
            # stdout has logs; whisper writes transcript to wav_path + ".txt"
            txt_path = wav_path + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read().strip()
        except Exception as e:
            logger.error(f"whisper.cpp failed: {e}")
            return ""
    return ""

def extract_video_keyframe_texts_optional(path: Path) -> str:
    """Grab sparse keyframes (1 every N seconds), OCR them, optionally caption."""
    if not bool(cfg.get("USE_VIDEO_KEYFRAME_OCR", False)):
        return ""
    if not _ffmpeg_exists():
        logger.info("ffmpeg not found; skipping keyframe OCR.")
        return ""
    every_sec = int(cfg.get("KEYFRAME_EVERY_SEC", 3))
    texts = []
    with tempfile.TemporaryDirectory() as td:
        pattern = str(Path(td) / "kf_%04d.jpg")
        try:
            subprocess.run(
                ["ffmpeg", "-i", str(path), "-vf", f"fps=1/{every_sec}", "-frames:v", cfg.get("KEYFRAME_MAX", "10"), pattern],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logger.error(f"ffmpeg keyframe extract failed: {e}")
            return ""
        # OCR + optional caption
        for img_path in sorted(Path(td).glob("kf_*.jpg")):
            ocr = extract_text_from_image(img_path)
            cap = ""
            if bool(cfg.get("USE_IMAGE_CAPTION", False)):
                cap = caption_image_description(img_path)
            snippet = "\n".join([s for s in [ocr, cap] if s]).strip()
            if snippet:
                texts.append(f"[Keyframe {img_path.name}] {snippet}")
    return "\n".join(texts)
