"""
Main Classification Pipeline using Qwen3-4B-Instruct-2507

This pipeline:
1. Extracts text from PDFs using Qwen3-VL-4B OCR
2. Classifies text into questions/answers/metadata using Qwen3-4B-Instruct few-shot learning
3. Saves organized results

Usage:
    python pipeline.py <pdf_path>
    python pipeline.py data/raw/test_2.pdf
    python pipeline.py --text outputs/extracted.txt  # Classify pre-extracted text
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ocr.ocr_extractor import OCRExtractor
from src.classification.qwen_classifier import QwenClassifier


def run_full_pipeline(pdf_path: str, output_dir: str = "outputs") -> dict:
    """
    Run full OCR + Classification pipeline on a PDF.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save outputs
        
    Returns:
        Classification results dictionary
    """
    basename = Path(pdf_path).stem
    
    print("=" * 70)
    print(f"CLASSIFICATION PIPELINE")
    print(f"Input: {pdf_path}")
    print(f"Model: Qwen3-4B-Instruct-2507 (few-shot)")
    print("=" * 70)
    
    # Step 1: OCR Extraction
    print("\n[STEP 1] OCR Extraction (Qwen3-VL-4B)...")
    print("-" * 50)
    
    ocr = OCRExtractor()
    extracted_text = ocr.extract_from_file(pdf_path)
    
    lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
    print(f"  ✓ Extracted {len(lines)} lines from PDF")
    
    # Save extracted text
    os.makedirs(output_dir, exist_ok=True)
    extracted_path = os.path.join(output_dir, f"{basename}_extracted.txt")
    with open(extracted_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    print(f"  ✓ Saved to: {extracted_path}")
    
    # Step 2: Classification
    print("\n[STEP 2] Classification (Qwen3-4B-Instruct)...")
    print("-" * 50)
    
    classifier = QwenClassifier()
    results = classifier.classify_document(lines, merge_lines=True, show_progress=True)
    
    # Step 3: Save Results
    print("\n[STEP 3] Saving Results...")
    print("-" * 50)
    
    _save_results(results, basename, output_dir)
    _print_summary(results)
    
    return results


def run_classification_only(text_path: str, output_dir: str = "outputs") -> dict:
    """
    Run classification on pre-extracted text.
    
    Args:
        text_path: Path to text file
        output_dir: Directory to save outputs
        
    Returns:
        Classification results dictionary
    """
    basename = Path(text_path).stem
    
    print("=" * 70)
    print(f"CLASSIFICATION PIPELINE (Text Only)")
    print(f"Input: {text_path}")
    print(f"Model: Qwen3-4B-Instruct-2507 (few-shot)")
    print("=" * 70)
    
    # Load text
    print("\n[STEP 1] Loading Text...")
    print("-" * 50)
    
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    print(f"  ✓ Loaded {len(lines)} lines")
    
    # Classification
    print("\n[STEP 2] Classification (Qwen3-4B-Instruct)...")
    print("-" * 50)
    
    classifier = QwenClassifier()
    results = classifier.classify_document(lines, merge_lines=True, show_progress=True)
    
    # Save Results
    print("\n[STEP 3] Saving Results...")
    print("-" * 50)
    
    _save_results(results, basename, output_dir)
    _print_summary(results)
    
    return results


def _save_results(results: dict, basename: str, output_dir: str):
    """Save classification results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp
    results["timestamp"] = datetime.now().isoformat()
    
    # Save full JSON results
    results_path = os.path.join(output_dir, f"{basename}_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Full results: {results_path}")
    
    # Save questions
    questions_path = os.path.join(output_dir, f"{basename}_questions.txt")
    with open(questions_path, 'w', encoding='utf-8') as f:
        for q in results["questions"]:
            f.write(q["text"] + "\n\n")
    print(f"  ✓ Questions ({len(results['questions'])}): {questions_path}")
    
    # Save answers
    answers_path = os.path.join(output_dir, f"{basename}_answers.txt")
    with open(answers_path, 'w', encoding='utf-8') as f:
        for a in results["answers"]:
            f.write(a["text"] + "\n\n")
    print(f"  ✓ Answers ({len(results['answers'])}): {answers_path}")


def _print_summary(results: dict):
    """Print classification summary."""
    stats = results["statistics"]
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 STATISTICS:")
    print(f"  Original lines:   {stats['total_original_lines']}")
    print(f"  Merged blocks:    {stats['total_merged_blocks']}")
    print(f"  Questions:        {stats['questions_count']}")
    print(f"  Answers:          {stats['answers_count']}")
    print(f"  Metadata:         {stats['metadata_count']}")
    
    # Sample questions
    if results["questions"]:
        print(f"\n📝 SAMPLE QUESTIONS:")
        for i, q in enumerate(results["questions"][:3], 1):
            text = q["text"][:80] + "..." if len(q["text"]) > 80 else q["text"]
            print(f"  {i}. {text}")
    
    # Sample answers
    if results["answers"]:
        print(f"\n✅ SAMPLE ANSWERS:")
        for i, a in enumerate(results["answers"][:3], 1):
            text = a["text"][:80] + "..." if len(a["text"]) > 80 else a["text"]
            print(f"  {i}. {text}")
    
    print("\n" + "=" * 70)
    print("COMPLETED!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Classification Pipeline using Qwen3-4B-Instruct-2507",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py data/raw/test_2.pdf
  python pipeline.py data/raw/assignment1.pdf
  python pipeline.py --text outputs/test_2_extracted.txt
        """
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to PDF file (or text file with --text flag)"
    )
    parser.add_argument(
        "--text", "-t",
        action="store_true",
        help="Input is a text file (skip OCR)"
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs",
        help="Output directory (default: outputs)"
    )
    
    args = parser.parse_args()
    
    if not args.input:
        # Default: process test_2.pdf if exists
        default_pdf = "data/raw/test_2.pdf"
        if os.path.exists(default_pdf):
            print(f"No input specified. Using default: {default_pdf}")
            run_full_pipeline(default_pdf, args.output)
        else:
            parser.print_help()
            sys.exit(1)
    elif args.text:
        run_classification_only(args.input, args.output)
    else:
        run_full_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()

