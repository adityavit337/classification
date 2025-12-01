"""OCR text extraction using Qwen3-VL-8B-Instruct vision-language model."""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple
import logging
import re
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)


class OCRExtractor:
    """Extract text from images and PDFs using Qwen3-VL-4B-Instruct vision-language model."""
    
    # Page separator pattern used internally
    PAGE_SEPARATOR = "--- PAGE {page_num} ---"
    PAGE_SEPARATOR_PATTERN = r'---\s*PAGE\s*\d+\s*---'
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct", 
        device: str = None,
        enable_preprocessing: bool = True,
        max_pages: int = None,
        use_flash_attention: bool = True
    ):
        """
        Initialize Qwen3-VL OCR extractor.
        
        Args:
            model_name: Qwen3-VL model name from HuggingFace
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            enable_preprocessing: Whether to apply basic preprocessing to output
            max_pages: Maximum number of pages to process (None for all)
            use_flash_attention: Whether to use Flash Attention 2 (requires flash-attn package)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_preprocessing = enable_preprocessing
        self.max_pages = max_pages
        
        logger.info(f"Loading Qwen3-VL model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Check for Flash Attention
        has_flash_attn = False
        if use_flash_attention:
            try:
                import flash_attn
                has_flash_attn = True
                logger.info(f"Flash Attention detected: v{flash_attn.__version__}")
            except ImportError:
                logger.info("Flash Attention not installed - using standard attention")
        
        try:
            # Prepare model loading arguments
            model_kwargs = {
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            # Configure for GPU with Flash Attention if available
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.bfloat16
                
                if has_flash_attn and use_flash_attention:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using Flash Attention 2 with bfloat16 on GPU")
                else:
                    logger.info("Using standard attention with bfloat16 on GPU")
            else:
                model_kwargs["torch_dtype"] = torch.float32
                logger.info("Using CPU with float32")
            
            # Load model
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            logger.info("Qwen3-VL model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen3-VL model: {str(e)}")
            raise
    
    def _basic_text_cleanup(self, text: str) -> str:
        """
        Apply basic text cleanup immediately after OCR extraction.
        This handles OCR-specific artifacts before full preprocessing.
        
        Args:
            text: Raw OCR output text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove markdown artifacts that model might produce
        
        # Remove image references: ![...](... ) or ! []
        text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
        text = re.sub(r'!\[\]', '', text)
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove markdown code block markers
        text = re.sub(r'```[\w]*\n?', '', text)
        
        # Remove markdown horizontal rules
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        
        # Clean up markdown headers but preserve text
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove markdown bold/italic markers but keep text
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
        text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
        
        # Remove markdown links but keep text: [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove bullet points and list markers
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove escape characters
        text = re.sub(r'\\([*_`#\[\]])', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _enhance_image_for_handwriting(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better handwriting recognition.
        
        Args:
            image: PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        try:
            from PIL import ImageEnhance, ImageFilter
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Increase contrast for handwritten text
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            
            # Increase sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Slight brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.05)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed, using original: {str(e)}")
            return image
    
    def is_pdf(self, file_path: Union[str, Path]) -> bool:
        """Check if the file is a PDF."""
        return str(file_path).lower().endswith('.pdf')
    
    def _pdf_to_images(self, pdf_path: Union[str, Path]) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PIL Images
        """
        try:
            import fitz  # PyMuPDF
            images = []
            pdf_document = fitz.open(str(pdf_path))
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # Render at 2x resolution for better quality
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                logger.info(f"Converted PDF page {page_num + 1}/{len(pdf_document)}")
            
            pdf_document.close()
            return images
        except ImportError:
            logger.warning("PyMuPDF not found, trying pdf2image")
            return convert_from_path(str(pdf_path), dpi=200)
    
    def _extract_from_image(
        self, 
        image: Image.Image, 
        enhance_handwriting: bool = True,
        custom_prompt: str = None
    ) -> str:
        """
        Extract text from a PIL Image using Qwen3-VL.
        
        Args:
            image: PIL Image
            enhance_handwriting: Whether to apply image enhancement
            custom_prompt: Optional custom prompt for extraction
            
        Returns:
            Extracted text as string
        """
        # Enhance image for handwriting if enabled
        if enhance_handwriting:
            image = self._enhance_image_for_handwriting(image)
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        try:
            # Default OCR prompt
            if custom_prompt is None:
                prompt = "Extract all text from this image. Preserve the original text structure, line breaks, and formatting. Output only the extracted text without any additional commentary."
            else:
                prompt = custom_prompt
            
            # Prepare messages in Qwen3-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Process inputs
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate output
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            
            # Decode output, removing input tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            result = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # Apply basic cleanup if enabled
            if result and self.enable_preprocessing:
                result = self._basic_text_cleanup(result)
            
            return result.strip() if result else ""
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return ""
    
    def extract_text_from_pdf(
        self, 
        pdf_path: Union[str, Path],
        return_pages_separately: bool = False
    ) -> Union[str, List[Dict]]:
        """
        Extract text from all pages in a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            return_pages_separately: If True, return list of page dicts
            
        Returns:
            Combined extracted text or list of page dictionaries
        """
        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            images = self._pdf_to_images(pdf_path)
            total_pages = len(images)
            
            # Limit pages if max_pages is set
            if self.max_pages and self.max_pages < total_pages:
                images = images[:self.max_pages]
                logger.info(f"PDF has {total_pages} page(s), processing first {self.max_pages}")
            else:
                logger.info(f"PDF has {total_pages} page(s)")
            
            pages_data = []
            texts = []
            
            for i, image in enumerate(images):
                logger.info(f"Processing PDF page {i + 1}/{len(images)}")
                text = self._extract_from_image(image)
                
                if return_pages_separately:
                    pages_data.append({
                        'page_number': i + 1,
                        'text': text.strip() if text else "",
                        'has_content': bool(text and text.strip())
                    })
                
                if text and text.strip():
                    texts.append(f"{self.PAGE_SEPARATOR.format(page_num=i + 1)}\n{text.strip()}")
            
            if return_pages_separately:
                return pages_data
            
            # Join all pages with double newline separator
            return "\n\n".join(texts)
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            return [] if return_pages_separately else ""
    
    def extract_text(
        self, 
        image: Union[str, Path, Image.Image],
        return_metadata: bool = False,
        custom_prompt: str = None
    ) -> Union[str, Dict]:
        """
        Extract text from a single image or PDF file.
        
        Args:
            image: Image as file path, Path object, or PIL Image
            return_metadata: If True, return dict with text and metadata
            custom_prompt: Optional custom prompt for extraction
            
        Returns:
            Extracted text as string, or dict with text and metadata
        """
        metadata = {
            'source_type': 'unknown',
            'source_path': None,
            'pages': 1,
            'extraction_success': False
        }
        
        try:
            # Check if input is a PDF file
            if isinstance(image, (str, Path)):
                metadata['source_path'] = str(image)
                
                if self.is_pdf(image):
                    metadata['source_type'] = 'pdf'
                    text = self.extract_text_from_pdf(image)
                    # Count pages from separator
                    metadata['pages'] = len(re.findall(self.PAGE_SEPARATOR_PATTERN, text)) or 1
                else:
                    metadata['source_type'] = 'image'
                    image = Image.open(image).convert('RGB')
                    text = self._extract_from_image(image, custom_prompt=custom_prompt)
            elif isinstance(image, Image.Image):
                metadata['source_type'] = 'pil_image'
                text = self._extract_from_image(image, custom_prompt=custom_prompt)
            else:
                logger.error("Invalid image input type")
                text = ""
            
            metadata['extraction_success'] = bool(text and text.strip())
            metadata['char_count'] = len(text) if text else 0
            metadata['word_count'] = len(text.split()) if text else 0
            
            if return_metadata:
                return {'text': text, 'metadata': metadata}
            return text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            if return_metadata:
                return {'text': '', 'metadata': metadata}
            return ""
    
    def extract_from_file(self, image_path: Union[str, Path]) -> str:
        """
        Extract text directly from an image or PDF file.
        
        Args:
            image_path: Path to the image or PDF file
            
        Returns:
            Extracted text as string
        """
        logger.info(f"Extracting text from: {image_path}")
        return self.extract_text(image_path)
    
    def extract_from_multiple_images(
        self, 
        image_paths: List[Union[str, Path]],
        return_metadata: bool = False
    ) -> List[Union[str, Dict]]:
        """
        Extract text from multiple images or PDF files.
        
        Args:
            image_paths: List of image or PDF file paths
            return_metadata: If True, return list of dicts with metadata
            
        Returns:
            List of extracted text strings or dicts
        """
        results = []
        for idx, image_path in enumerate(image_paths):
            file_type = "PDF" if self.is_pdf(image_path) else "image"
            logger.info(f"Processing {file_type} {idx + 1}/{len(image_paths)}: {Path(image_path).name}")
            result = self.extract_text(image_path, return_metadata=return_metadata)
            results.append(result)
        return results
    
    def extract_batch(
        self, 
        image_paths: List[Union[str, Path]], 
        batch_size: int = 4
    ) -> List[str]:
        """
        Extract text from multiple images or PDFs in batches for efficiency.
        
        Args:
            image_paths: List of image or PDF file paths
            batch_size: Number of files to process at once
            
        Returns:
            List of extracted text strings
        """
        texts = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(image_paths) - 1) // batch_size + 1}")
            
            for file_path in batch_paths:
                text = self.extract_text(file_path)
                texts.append(text)
        
        return texts
    
    def extract_and_preprocess(
        self,
        image_path: Union[str, Path],
        preprocessor=None
    ) -> Tuple[str, str]:
        """
        Extract text and apply full preprocessing in one call.
        
        Args:
            image_path: Path to image or PDF file
            preprocessor: OCRTextPreprocessor instance (imports if None)
            
        Returns:
            Tuple of (raw_text, preprocessed_text)
        """
        # Extract raw text
        raw_text = self.extract_text(image_path)
        
        if not raw_text:
            return "", ""
        
        # Apply preprocessing
        if preprocessor is None:
            try:
                from ocr_text_preprocessor import OCRTextPreprocessor
                preprocessor = OCRTextPreprocessor()
            except ImportError:
                logger.warning("OCRTextPreprocessor not found, returning raw text")
                return raw_text, raw_text
        
        preprocessed_text = preprocessor.preprocess(raw_text, aggressive=True)
        
        return raw_text, preprocessed_text
    
    def describe_image(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Get a description of the image content (not just text extraction).
        
        Args:
            image: Image as file path, Path object, or PIL Image
            
        Returns:
            Description of the image
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        return self._extract_from_image(
            image, 
            enhance_handwriting=False,
            custom_prompt="Describe what you see in this image in detail."
        )
