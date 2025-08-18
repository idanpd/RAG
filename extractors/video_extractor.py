"""
Video content extractor with transcript and keyframe analysis.
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Union, Dict, Any, List
import logging

from .base import BaseExtractor, ExtractionResult
from .image_extractor import ImageExtractor

# Import libraries with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class VideoExtractor(BaseExtractor):
    """Extractor for video files with transcript and keyframe analysis."""
    
    SUPPORTED_EXTENSIONS = {
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp'
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.use_transcript = self.config.get('USE_VIDEO_TRANSCRIPT', False)
        self.use_keyframe_ocr = self.config.get('USE_VIDEO_KEYFRAME_OCR', True)
        self.whisper_bin = self.config.get('WHISPER_CPP_BIN', '')
        self.whisper_model = self.config.get('WHISPER_MODEL', '')
        self.keyframe_interval = self.config.get('KEYFRAME_EVERY_SEC', 3)
        self.max_keyframes = self.config.get('KEYFRAME_MAX', 10)
        self.max_file_size = self.config.get('max_video_file_size', 500 * 1024 * 1024)  # 500MB
        
        # Initialize image extractor for keyframe processing
        self.image_extractor = ImageExtractor(config)
    
    def can_extract(self, file_path: Union[str, Path]) -> bool:
        """Check if this extractor can handle the file."""
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def extract(self, file_path: Union[str, Path]) -> ExtractionResult:
        """Extract content from video files."""
        path = Path(file_path)
        
        # Check file size
        try:
            if path.stat().st_size > self.max_file_size:
                return ExtractionResult(
                    content="",
                    success=False,
                    error_message=f"Video file too large: {path.stat().st_size} bytes"
                )
        except Exception as e:
            return ExtractionResult(
                content="",
                success=False,
                error_message=f"Cannot access video file: {e}"
            )
        
        content_parts = []
        metadata = {'filename': path.name, 'path': str(path)}
        extraction_methods = []
        
        # Try transcript extraction
        if self.use_transcript:
            transcript_result = self._extract_transcript(path)
            if transcript_result.success and transcript_result.content.strip():
                content_parts.append(f"Video Transcript:\n{transcript_result.content}")
                extraction_methods.append('transcript')
                metadata.update(transcript_result.metadata)
        
        # Try keyframe OCR
        if self.use_keyframe_ocr:
            keyframe_result = self._extract_keyframe_text(path)
            if keyframe_result.success and keyframe_result.content.strip():
                content_parts.append(f"Keyframe Text:\n{keyframe_result.content}")
                extraction_methods.append('keyframe_ocr')
                metadata.update(keyframe_result.metadata)
        
        # Get basic video metadata
        video_metadata = self._get_video_metadata(path)
        metadata.update(video_metadata)
        
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
    
    def _extract_transcript(self, path: Path) -> ExtractionResult:
        """Extract transcript using whisper.cpp."""
        if not self.whisper_bin or not self.whisper_model:
            return ExtractionResult(
                content="",
                success=False,
                error_message="Whisper binary or model not configured"
            )
        
        if not os.path.exists(self.whisper_bin):
            return ExtractionResult(
                content="",
                success=False,
                error_message=f"Whisper binary not found: {self.whisper_bin}"
            )
        
        if not os.path.exists(self.whisper_model):
            return ExtractionResult(
                content="",
                success=False,
                error_message=f"Whisper model not found: {self.whisper_model}"
            )
        
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Run whisper.cpp
            cmd = [
                self.whisper_bin,
                '-m', self.whisper_model,
                '-f', str(path),
                '-otxt',
                '-of', tmp_path[:-4]  # Remove .txt extension as whisper adds it
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Read transcript
            transcript_file = tmp_path
            if os.path.exists(transcript_file):
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                
                # Clean up
                os.unlink(transcript_file)
                
                return ExtractionResult(
                    content=transcript,
                    metadata={'whisper_model': self.whisper_model},
                    success=True,
                    extraction_method='whisper_cpp'
                )
            else:
                return ExtractionResult(
                    content="",
                    success=False,
                    error_message="Whisper output file not found",
                    extraction_method='whisper_cpp'
                )
                
        except subprocess.TimeoutExpired:
            return ExtractionResult(
                content="",
                success=False,
                error_message="Whisper transcription timeout",
                extraction_method='whisper_cpp'
            )
        except Exception as e:
            self.logger.error(f"Transcript extraction failed for {path}: {e}")
            return ExtractionResult(
                content="",
                success=False,
                error_message=f"Transcript extraction failed: {e}",
                extraction_method='whisper_cpp'
            )
    
    def _extract_keyframe_text(self, path: Path) -> ExtractionResult:
        """Extract text from video keyframes using OCR."""
        if not CV2_AVAILABLE:
            return ExtractionResult(
                content="",
                success=False,
                error_message="OpenCV not available for video processing"
            )
        
        try:
            keyframes = self._extract_keyframes(path)
            if not keyframes:
                return ExtractionResult(
                    content="",
                    success=False,
                    error_message="No keyframes extracted"
                )
            
            ocr_texts = []
            successful_frames = 0
            
            for i, frame in enumerate(keyframes):
                try:
                    # Save frame as temporary image
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                    
                    # Convert BGR to RGB for PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    pil_image.save(tmp_path)
                    
                    # Extract text using image extractor
                    ocr_result = self.image_extractor._extract_ocr_text(Path(tmp_path))
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                    if ocr_result.success and ocr_result.content.strip():
                        timestamp = i * self.keyframe_interval
                        ocr_texts.append(f"[{timestamp}s] {ocr_result.content.strip()}")
                        successful_frames += 1
                        
                except Exception as e:
                    self.logger.warning(f"OCR failed for keyframe {i}: {e}")
                    continue
            
            if ocr_texts:
                content = "\n".join(ocr_texts)
                return ExtractionResult(
                    content=content,
                    metadata={
                        'keyframes_processed': len(keyframes),
                        'keyframes_with_text': successful_frames
                    },
                    success=True,
                    extraction_method='keyframe_ocr'
                )
            else:
                return ExtractionResult(
                    content="",
                    success=False,
                    error_message="No text found in keyframes",
                    extraction_method='keyframe_ocr'
                )
                
        except Exception as e:
            self.logger.error(f"Keyframe text extraction failed for {path}: {e}")
            return ExtractionResult(
                content="",
                success=False,
                error_message=f"Keyframe extraction failed: {e}",
                extraction_method='keyframe_ocr'
            )
    
    def _extract_keyframes(self, path: Path) -> List:
        """Extract keyframes from video."""
        if not CV2_AVAILABLE:
            return []
        
        try:
            cap = cv2.VideoCapture(str(path))
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0:
                fps = 25  # Default fallback
            
            # Calculate frame interval
            frame_interval = int(fps * self.keyframe_interval)
            
            keyframes = []
            frame_count = 0
            
            while len(keyframes) < self.max_keyframes:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    keyframes.append(frame.copy())
                
                frame_count += 1
            
            cap.release()
            return keyframes
            
        except Exception as e:
            self.logger.error(f"Keyframe extraction failed for {path}: {e}")
            return []
    
    def _get_video_metadata(self, path: Path) -> Dict[str, Any]:
        """Get basic video metadata."""
        metadata = {}
        
        if not CV2_AVAILABLE:
            return metadata
        
        try:
            cap = cv2.VideoCapture(str(path))
            
            metadata.update({
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration_seconds': cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
            })
            
            cap.release()
            
        except Exception as e:
            self.logger.warning(f"Failed to get video metadata for {path}: {e}")
        
        return metadata