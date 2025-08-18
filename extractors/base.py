"""
Base classes for content extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of content extraction."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    extraction_method: Optional[str] = None
    
    def __post_init__(self):
        if not self.success and not self.error_message:
            self.error_message = "Unknown extraction error"


class BaseExtractor(ABC):
    """Base class for all content extractors."""
    
    SUPPORTED_EXTENSIONS: set = set()
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def can_extract(self, file_path: Union[str, Path]) -> bool:
        """Check if this extractor can handle the given file."""
        pass
    
    @abstractmethod
    def extract(self, file_path: Union[str, Path]) -> ExtractionResult:
        """Extract content from the file."""
        pass
    
    def _create_fallback_result(self, file_path: Union[str, Path]) -> ExtractionResult:
        """Create a fallback result with basic file metadata."""
        path = Path(file_path)
        try:
            stat = path.stat()
            content = f"File: {path.name}\nType: {path.suffix}\nSize: {stat.st_size} bytes\nPath: {path}"
            return ExtractionResult(
                content=content,
                metadata={
                    'filename': path.name,
                    'extension': path.suffix,
                    'size': stat.st_size,
                    'path': str(path)
                },
                success=True,
                extraction_method='fallback_metadata'
            )
        except Exception as e:
            return ExtractionResult(
                content=f"File: {path.name}",
                metadata={'filename': path.name, 'path': str(path)},
                success=False,
                error_message=str(e),
                extraction_method='fallback_metadata'
            )
    
    def _safe_extract(self, file_path: Union[str, Path], 
                     extraction_func, method_name: str) -> ExtractionResult:
        """Safely execute extraction function with error handling."""
        try:
            result = extraction_func(file_path)
            if isinstance(result, str):
                return ExtractionResult(
                    content=result,
                    success=True,
                    extraction_method=method_name
                )
            elif isinstance(result, ExtractionResult):
                result.extraction_method = method_name
                return result
            else:
                return ExtractionResult(
                    content=str(result),
                    success=True,
                    extraction_method=method_name
                )
        except Exception as e:
            self.logger.error(f"Extraction failed for {file_path} using {method_name}: {e}")
            return ExtractionResult(
                content="",
                success=False,
                error_message=str(e),
                extraction_method=method_name
            )


class MultiExtractor:
    """Manages multiple extractors and routes files to appropriate extractors."""
    
    def __init__(self, extractors: Optional[List[BaseExtractor]] = None):
        self.extractors = extractors or []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_extractor(self, extractor: BaseExtractor):
        """Add an extractor to the collection."""
        self.extractors.append(extractor)
    
    def get_extractor_for_file(self, file_path: Union[str, Path]) -> Optional[BaseExtractor]:
        """Get the appropriate extractor for a file."""
        for extractor in self.extractors:
            if extractor.can_extract(file_path):
                return extractor
        return None
    
    def extract(self, file_path: Union[str, Path]) -> ExtractionResult:
        """Extract content using the appropriate extractor."""
        extractor = self.get_extractor_for_file(file_path)
        if extractor:
            return extractor.extract(file_path)
        else:
            # Fallback to basic metadata extraction
            self.logger.warning(f"No extractor found for {file_path}, using fallback")
            return BaseExtractor()._create_fallback_result(file_path)