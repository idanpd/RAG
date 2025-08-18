"""
Image content extractor with OCR and captioning capabilities.
"""

from pathlib import Path
from typing import Union, Dict, Any
import logging

from .base import BaseExtractor, ExtractionResult

# Import libraries with fallbacks
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageExtractor(BaseExtractor):
    """Extractor for image files with OCR and captioning."""
    
    SUPPORTED_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif'
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.use_ocr = self.config.get('USE_OCR', True)
        self.use_caption = self.config.get('USE_IMAGE_CAPTION', True)
        self.caption_model_name = self.config.get('IMAGE_CAPTION_MODEL', 
                                                 'Salesforce/blip-image-captioning-base')
        self.max_caption_tokens = self.config.get('CAPTION_MAX_TOKENS', 32)
        self.max_file_size = self.config.get('max_image_file_size', 50 * 1024 * 1024)  # 50MB
        
        # Lazy loading for caption model
        self._caption_model = None
        self._caption_processor = None
    
    def can_extract(self, file_path: Union[str, Path]) -> bool:
        """Check if this extractor can handle the file."""
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def extract(self, file_path: Union[str, Path]) -> ExtractionResult:
        """Extract content from image files using OCR and/or captioning."""
        path = Path(file_path)
        
        # Check file size
        try:
            if path.stat().st_size > self.max_file_size:
                return ExtractionResult(
                    content="",
                    success=False,
                    error_message=f"Image file too large: {path.stat().st_size} bytes"
                )
        except Exception as e:
            return ExtractionResult(
                content="",
                success=False,
                error_message=f"Cannot access image file: {e}"
            )
        
        content_parts = []
        metadata = {'filename': path.name, 'path': str(path)}
        extraction_methods = []
        
        # Try OCR extraction
        if self.use_ocr:
            ocr_result = self._extract_ocr_text(path)
            if ocr_result.success and ocr_result.content.strip():
                content_parts.append(f"OCR Text:\n{ocr_result.content}")
                extraction_methods.append('ocr')
                metadata.update(ocr_result.metadata)
        
        # Try caption generation
        if self.use_caption:
            caption_result = self._generate_caption(path)
            if caption_result.success and caption_result.content.strip():
                content_parts.append(f"Image Description:\n{caption_result.content}")
                extraction_methods.append('caption')
                metadata.update(caption_result.metadata)
        
        # Combine results
        if content_parts:
            content = "\n\n".join(content_parts)
            return ExtractionResult(
                content=content,
                metadata=metadata,
                success=True,
                extraction_method='+'.join(extraction_methods)
            )
        else:
            # Fallback to basic metadata
            return self._create_fallback_result(path)
    
    def _extract_ocr_text(self, path: Path) -> ExtractionResult:
        """Extract text from image using OCR."""
        if not PIL_AVAILABLE:
            return ExtractionResult(
                content="",
                success=False,
                error_message="PIL not available for image processing"
            )
        
        if not PYTESSERACT_AVAILABLE:
            return ExtractionResult(
                content="",
                success=False,
                error_message="pytesseract not available for OCR"
            )
        
        try:
            # Open and process image
            with Image.open(path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get image info
                metadata = {
                    'image_size': img.size,
                    'image_mode': img.mode,
                    'image_format': img.format
                }
                
                # Perform OCR
                text = pytesseract.image_to_string(img, config='--psm 3')
                
                return ExtractionResult(
                    content=text.strip(),
                    metadata=metadata,
                    success=True,
                    extraction_method='pytesseract'
                )
                
        except Exception as e:
            self.logger.error(f"OCR extraction failed for {path}: {e}")
            return ExtractionResult(
                content="",
                success=False,
                error_message=f"OCR failed: {e}",
                extraction_method='pytesseract'
            )
    
    def _get_caption_model(self):
        """Lazy load caption model and processor."""
        if self._caption_model is not None:
            return self._caption_model, self._caption_processor
        
        if not TRANSFORMERS_AVAILABLE:
            return None, None
        
        try:
            self._caption_processor = BlipProcessor.from_pretrained(self.caption_model_name)
            self._caption_model = BlipForConditionalGeneration.from_pretrained(self.caption_model_name)
            self.logger.info(f"Loaded caption model: {self.caption_model_name}")
            return self._caption_model, self._caption_processor
        except Exception as e:
            self.logger.error(f"Failed to load caption model {self.caption_model_name}: {e}")
            return None, None
    
    def _generate_caption(self, path: Path) -> ExtractionResult:
        """Generate caption for image using BLIP model."""
        model, processor = self._get_caption_model()
        if model is None or processor is None:
            return ExtractionResult(
                content="",
                success=False,
                error_message="Caption model not available"
            )
        
        if not PIL_AVAILABLE:
            return ExtractionResult(
                content="",
                success=False,
                error_message="PIL not available for image processing"
            )
        
        try:
            with Image.open(path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Generate caption
                inputs = processor(images=img, return_tensors="pt")
                out = model.generate(**inputs, max_new_tokens=self.max_caption_tokens)
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                return ExtractionResult(
                    content=caption,
                    metadata={'caption_model': self.caption_model_name},
                    success=True,
                    extraction_method='blip_caption'
                )
                
        except Exception as e:
            self.logger.error(f"Caption generation failed for {path}: {e}")
            return ExtractionResult(
                content="",
                success=False,
                error_message=f"Caption generation failed: {e}",
                extraction_method='blip_caption'
            )