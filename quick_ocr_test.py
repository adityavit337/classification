"""
Quick OCR test on first few pages of assignment1.pdf
"""

import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
import fitz  # PyMuPDF

def setup_model():
    """Initialize DeepSeek-OCR model"""
    print("Loading DeepSeek-OCR model...")
    
    model_name = "deepseek-ai/DeepSeek-OCR"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with Flash Attention if available
    try:
        import flash_attn
        has_flash = True
    except ImportError:
        has_flash = False
    
    model_kwargs = {
        "trust_remote_code": True,
        "use_safetensors": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda",
    }
    
    if has_flash:
        model_kwargs["_attn_implementation"] = "flash_attention_2"
        print("✓ Using Flash Attention 2")
    
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    model.eval()
    
    print("✓ Model loaded successfully\n")
    return model, tokenizer

def extract_from_pdf_page(pdf_path, page_num, model, tokenizer, output_dir):
    """Extract text from a single PDF page"""
    # Convert page to image
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    pdf_doc.close()
    
    # Save temp image
    temp_image = output_dir / f"temp_page_{page_num}.png"
    img.save(temp_image)
    
    # Extract text using DeepSeek-OCR with save_results=True to capture output
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    # Capture the output by saving to markdown file
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=str(temp_image),
        output_path=str(output_dir),
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,  # Save to markdown file
        test_compress=False
    )
    
    # Read the saved markdown file
    md_file = output_dir / f"temp_page_{page_num}.md"
    if md_file.exists():
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.read()
        md_file.unlink()  # Clean up markdown file
    else:
        text = result if result else ""
    
    # Clean up temp image
    if temp_image.exists():
        temp_image.unlink()
    
    return text

def main():
    print("="*70)
    print(" QUICK DEEPSEEK OCR TEST")
    print("="*70)
    print()
    
    # Setup
    pdf_path = "data/raw/assignment1.pdf"
    output_dir = Path("outputs/temp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    # Load model
    model, tokenizer = setup_model()
    
    # Extract from first 3 pages only
    num_pages = 3
    all_text = []
    
    print(f"Extracting text from first {num_pages} pages...\n")
    
    for page_num in range(num_pages):
        print(f"Processing page {page_num + 1}/{num_pages}...")
        try:
            text = extract_from_pdf_page(pdf_path, page_num, model, tokenizer, output_dir)
            if text:
                all_text.append(f"--- PAGE {page_num + 1} ---\n{text.strip()}")
                print(f"✓ Page {page_num + 1} complete")
            else:
                all_text.append(f"--- PAGE {page_num + 1} ---\n[No text extracted]")
                print(f"⚠ Page {page_num + 1} returned no text")
        except Exception as e:
            print(f"❌ Error on page {page_num + 1}: {str(e)}")
            all_text.append(f"--- PAGE {page_num + 1} ---\n[Error: {str(e)}]")
    
    # Save results
    output_file = Path("outputs/assignment1_extracted.txt")
    combined_text = "\n\n".join(all_text)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    print(f"\n✓ Saved extracted text to: {output_file}")
    print(f"✓ Total characters: {len(combined_text)}")
    
    # Show preview
    print("\n" + "="*70)
    print(" EXTRACTED TEXT PREVIEW")
    print("="*70)
    print(combined_text[:500] + "..." if len(combined_text) > 500 else combined_text)
    print("="*70)

if __name__ == "__main__":
    main()
