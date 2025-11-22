"""Test script to demonstrate DeepSeek-OCR on assignment1.pdf"""

import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
import os

def setup_model():
    """Initialize DeepSeek-OCR model with Flash Attention"""
    print("=" * 80)
    print("Initializing DeepSeek-OCR")
    print("=" * 80)
    print()
    
    model_name = "deepseek-ai/DeepSeek-OCR"
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
    else:
        print("⚠ Warning: No GPU detected. Running on CPU will be very slow!")
    
    # Check if Flash Attention is available
    try:
        import flash_attn
        has_flash_attn = True
        print(f"✓ Flash Attention installed: v{flash_attn.__version__}")
    except ImportError:
        has_flash_attn = False
        print("⚠ Flash Attention not installed - using standard attention")
    
    print()
    print("Loading model... (this may take a few minutes)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    
    # Prepare model loading arguments
    model_kwargs = {
        "trust_remote_code": True,
        "use_safetensors": True,
    }
    
    # Set device and dtype BEFORE loading to avoid warnings
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "cuda"
        
        # Add Flash Attention if available
        if has_flash_attn:
            model_kwargs["_attn_implementation"] = "flash_attention_2"
            print("✓ Using Flash Attention 2 with bfloat16 on GPU")
        else:
            print("✓ Using standard attention with bfloat16 on GPU")
    else:
        model_kwargs["torch_dtype"] = torch.float32
        print("⚠ Using CPU with float32 (will be slow)")
    
    # Load model directly to GPU with proper dtype
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    model = model.eval()
    
    print("✓ Model loaded successfully")
    print()
    return model, tokenizer

def pdf_to_images(pdf_path):
    """Convert PDF pages to PIL Images using PyMuPDF"""
    print(f"Converting PDF to images: {pdf_path}")
    
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("\n✗ ERROR: PyMuPDF not installed!")
        print("Install with: pip install pymupdf")
        print("\nAlternatively, if you have pdf2image installed:")
        print("pip install pdf2image")
        raise
    
    images = []
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        # Render at 2x resolution for better quality
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        print(f"  ✓ Converted page {page_num + 1}/{len(pdf_document)}")
    
    pdf_document.close()
    return images

def extract_text_from_image(image, model, tokenizer, page_num, output_dir):
    """Extract text from a single image using DeepSeek-OCR's infer() method"""
    print(f"\nProcessing page {page_num}...")
    
    # Save image temporarily
    temp_image_path = output_dir / f"temp_page_{page_num}.png"
    image.save(temp_image_path)
    
    try:
        # Use DeepSeek-OCR's official infer() method
        prompt = "<image>\n<|grounding|>Convert the document to markdown."
        
        # Call the model's custom infer method
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(temp_image_path),
            output_path=str(output_dir),
            base_size=1024,      # Resolution control
            image_size=640,      # Token compression level
            crop_mode=True,      # Enable for better handling of large docs
            save_results=False,  # Don't save intermediate files
            test_compress=False  # Don't run compression tests
        )
        
        # Clean up temp file
        if temp_image_path.exists():
            temp_image_path.unlink()
        
        print(f"✓ Page {page_num} processed")
        return result
        
    except Exception as e:
        print(f"✗ Error processing page {page_num}: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up temp file on error
        if temp_image_path.exists():
            temp_image_path.unlink()
        
        return ""

def main():
    # Path to your assignment
    pdf_path = Path("data/raw/assignment1.pdf")
    
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found!")
        print(f"Current directory: {Path.cwd()}")
        return
    
    print("=" * 80)
    print("DeepSeek-OCR Test on assignment1.pdf")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize model
    model, tokenizer = setup_model()
    
    # Convert PDF to images
    print()
    images = pdf_to_images(pdf_path)
    print(f"\n✓ Total pages: {len(images)}")
    print()
    
    # Extract text from each page
    all_text = []
    
    for i, image in enumerate(images, 1):
        print("=" * 80)
        print(f"PAGE {i}/{len(images)}")
        print("=" * 80)
        
        text = extract_text_from_image(image, model, tokenizer, i, output_dir)
        
        if text:
            all_text.append(f"--- PAGE {i} ---\n{text}")
            
            # Show preview
            preview = text[:300] + "..." if len(text) > 300 else text
            print(f"\nPreview:\n{preview}\n")
    
    # Combine all pages
    extracted_text = "\n\n".join(all_text)
    
    # Display summary
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total pages processed: {len(images)}")
    print(f"Total characters extracted: {len(extracted_text)}")
    print(f"Total words (approximate): {len(extracted_text.split())}")
    print("=" * 80)
    
    # Save to file
    output_path = output_dir / "assignment1_extracted.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    
    print()
    print(f"✓ Full text saved to: {output_path}")
    
    # Display full text
    print("\n" + "=" * 80)
    print("FULL EXTRACTED TEXT")
    print("=" * 80)
    print()
    print(extracted_text)
    print()

if __name__ == '__main__':
    main()