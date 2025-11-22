"""OCR text extraction using DeepSeek-OCR."""

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from pathlib import Path
from typing import Union, List
import logging
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)


class OCRExtractor:
    """Extract text from images and PDFs using DeepSeek-OCR with Flash Attention support."""
    
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR", device: str = None):
        """
        Initialize DeepSeek OCR extractor.
        
        Args:
            model_name: DeepSeek OCR model name from HuggingFace
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading DeepSeek-OCR model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Check for Flash Attention
        try:
            import flash_attn
            has_flash_attn = True
            logger.info(f"Flash Attention detected: v{flash_attn.__version__}")
        except ImportError:
            has_flash_attn = False
            logger.info("Flash Attention not installed - using standard attention")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Prepare model loading arguments
            model_kwargs = {
                "trust_remote_code": True,
                "use_safetensors": True,
            }
            
            # Configure for GPU with Flash Attention if available
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["device_map"] = "cuda"
                
                if has_flash_attn:
                    model_kwargs["_attn_implementation"] = "flash_attention_2"
                    logger.info("Using Flash Attention 2 with bfloat16 on GPU")
                else:
                    logger.info("Using standard attention with bfloat16 on GPU")
            else:
                model_kwargs["torch_dtype"] = torch.float32
                logger.info("Using CPU with float32")
            
            # Load model
            self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
            self.model.eval()
            
            logger.info("DeepSeek-OCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DeepSeek-OCR model: {str(e)}")
            raise
        
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
    
    def _extract_from_image(self, image: Image.Image, output_dir: Path = None) -> str:
        """
        Extract text from a PIL Image using DeepSeek-OCR's infer method.
        
        Args:
            image: PIL Image
            output_dir: Optional directory for temporary files
            
        Returns:
            Extracted text as string
        """
        if output_dir is None:
            output_dir = Path("outputs/temp")
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image temporarily
        temp_image_path = output_dir / f"temp_image_{id(image)}.png"
        image.save(temp_image_path)
        
        try:
            # Use DeepSeek-OCR's official infer() method
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
            
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=str(temp_image_path),
                output_path=str(output_dir),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=False
            )
            
            # Clean up temp file
            if temp_image_path.exists():
                temp_image_path.unlink()
            
            return result.strip() if result else ""
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            # Clean up temp file on error
            if temp_image_path.exists():
                temp_image_path.unlink()
            return ""
    
    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text from all pages in a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Combined extracted text from all pages
        """
        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            images = self._pdf_to_images(pdf_path)
            logger.info(f"PDF has {len(images)} page(s)")
            
            texts = []
            for i, image in enumerate(images):
                logger.info(f"Processing PDF page {i + 1}/{len(images)}")
                text = self._extract_from_image(image)
                if text:
                    texts.append(f"--- PAGE {i + 1} ---\n{text.strip()}")
            
            # Join all pages with double newline separator
            return "\n\n".join(texts)
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            return ""
    
    def extract_text(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Extract text from a single image or PDF file.
        
        Args:
            image: Image as file path, Path object, or PIL Image
            
        Returns:
            Extracted text as string
        """
        try:
            # Check if input is a PDF file
            if isinstance(image, (str, Path)) and self.is_pdf(image):
                return self.extract_text_from_pdf(image)
            
            # Load image if path provided
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                logger.error("Invalid image input type")
                return ""
            
            # Extract text using DeepSeek-OCR
            return self._extract_from_image(image)
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return ""
    
    def extract_from_file(self, image_path: Union[str, Path]) -> str:
        """
        Extract text directly from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string
        """
        logger.info(f"Extracting text from: {image_path}")
        return self.extract_text(image_path)
    
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
    
    def extract_from_multiple_images(self, image_paths: List[Union[str, Path]]) -> List[str]:
        """
        Extract text from multiple images or PDF files.
        
        Args:
            image_paths: List of image or PDF file paths
            
        Returns:
            List of extracted text strings
        """
        texts = []
        for idx, image_path in enumerate(image_paths):
            file_type = "PDF" if self.is_pdf(image_path) else "image"
            logger.info(f"Processing {file_type} {idx + 1}/{len(image_paths)}: {Path(image_path).name}")
            text = self.extract_text(image_path)
            texts.append(text)
        return texts
    
    def extract_batch(self, image_paths: List[Union[str, Path]], batch_size: int = 4) -> List[str]:
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
