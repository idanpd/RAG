# extractors/images.py
from pathlib import Path
from PIL import Image
import pytesseract
from utils import load_config, setup_logger

# Optional captioning (tiny CPU model)
# Requires: pip install transformers pillow torch --index-url https://download.pytorch.org/whl/cpu
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    _TRANS_AVAILABLE = True
except Exception:
    _TRANS_AVAILABLE = False

cfg = load_config()
logger = setup_logger(cfg.get("LOG_LEVEL", "INFO"))

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

_CAPTION_MODEL = None
_CAPTION_PROCESSOR = None

def _get_caption_model():
    global _CAPTION_MODEL, _CAPTION_PROCESSOR
    if _CAPTION_MODEL is not None:
        return _CAPTION_MODEL, _CAPTION_PROCESSOR
    if not _TRANS_AVAILABLE:
        return None, None
    if not bool(cfg.get("USE_IMAGE_CAPTION", False)):
        return None, None
    model_name = cfg.get("IMAGE_CAPTION_MODEL", "Salesforce/blip-image-captioning-base")  # tiny-ish CPU-capable
    try:
        _CAPTION_PROCESSOR = BlipProcessor.from_pretrained(model_name)
        _CAPTION_MODEL = BlipForConditionalGeneration.from_pretrained(model_name)
        return _CAPTION_MODEL, _CAPTION_PROCESSOR
    except Exception as e:
        logger.error(f"Failed loading BLIP caption model: {e}")
        return None, None

def extract_text_from_image(path: Path) -> str:
    try:
        img = Image.open(path)
        t = pytesseract.image_to_string(img)
        return t.strip()
    except Exception as e:
        logger.error(f"OCR failed for {path}: {e}")
        return ""

def caption_image_description(path: Path) -> str:
    model, proc = _get_caption_model()
    if model is None or proc is None:
        return ""
    try:
        image = Image.open(path).convert("RGB")
        inputs = proc(images=image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=int(cfg.get("CAPTION_MAX_TOKENS", 32)))
        caption = proc.decode(out[0], skip_special_tokens=True)
        return f"Image caption: {caption}"
    except Exception as e:
        logger.error(f"Captioning failed for {path}: {e}")
        return ""
