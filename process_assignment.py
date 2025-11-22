"""
Process assignment1 extracted text and save results
"""

from src.text_processing.text_processor import TextProcessor
from pathlib import Path
import json

def main():
    print("="*70)
    print(" PROCESSING ASSIGNMENT1 EXTRACTED TEXT")
    print("="*70)
    
    # Initialize processor
    processor = TextProcessor(remove_bounding_boxes=True)
    
    # Input and output paths
    input_file = Path('outputs/assignment1_extracted.txt')
    output_jsonl = Path('data/processed/assignment1.jsonl')
    output_txt = Path('data/processed/assignment1_cleaned.txt')
    
    # Check if input exists
    if not input_file.exists():
        print(f"\n❌ Input file not found: {input_file}")
        print("Please run DeepSeek OCR first to generate the extracted text.")
        return
    
    # Read the extracted text
    print(f"\n📖 Reading: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"✓ Loaded {len(text)} characters")
    
    # Process for classification
    print("\n🔄 Processing with intelligent page break merging...")
    jsonl_data, stats = processor.process_for_classification(
        text,
        output_jsonl=str(output_jsonl),
        merge_page_breaks=True
    )
    
    # Save cleaned text version too
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(entry['text'] + '\n\n')
    
    print(f"\n✓ Saved JSONL to: {output_jsonl}")
    print(f"✓ Saved cleaned text to: {output_txt}")
    
    # Display statistics
    print("\n" + "="*70)
    print(" STATISTICS")
    print("="*70)
    print(f"Total lines extracted: {stats['total_lines']}")
    print(f"Likely questions: {stats['likely_questions']}")
    print(f"Likely answers: {stats['likely_answers']}")
    print(f"Other (metadata/text): {stats['other']}")
    
    # Show first 10 lines
    print("\n" + "="*70)
    print(" FIRST 10 PROCESSED LINES")
    print("="*70)
    for i, entry in enumerate(jsonl_data[:10], 1):
        text_preview = entry['text'][:80] + "..." if len(entry['text']) > 80 else entry['text']
        print(f"{i}. {text_preview}")
    
    # Show some questions
    print("\n" + "="*70)
    print(" SAMPLE QUESTIONS")
    print("="*70)
    questions = [entry for entry in jsonl_data if processor.get_line_type(entry['text']) == 'question']
    for i, entry in enumerate(questions[:5], 1):
        print(f"\nQ{i}: {entry['text']}")
    
    # Show some answers
    print("\n" + "="*70)
    print(" SAMPLE ANSWERS")
    print("="*70)
    answers = [entry for entry in jsonl_data if processor.get_line_type(entry['text']) == 'answer']
    for i, entry in enumerate(answers[:5], 1):
        text_preview = entry['text'][:150] + "..." if len(entry['text']) > 150 else entry['text']
        print(f"\nA{i}: {text_preview}")
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print(f"""
1. Review the processed files:
   - JSONL: {output_jsonl}
   - Text: {output_txt}

2. Use with DeBERTa classifier:
   from src.classification.classifier import DeBERTaClassifier
   classifier = DeBERTaClassifier('path/to/model')
   predictions = classifier.predict_batch([entry['text'] for entry in jsonl_data])

3. Or run full pipeline:
   python pipeline.py --input data/raw/assignment1.pdf
""")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
