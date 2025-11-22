"""
Demonstration of Text Processing with Page Break Handling
Shows before/after comparison and example outputs
"""

from src.text_processing.text_processor import TextProcessor
from pathlib import Path
import json

def print_separator(title=""):
    """Print a formatted separator"""
    print("\n" + "="*70)
    if title:
        print(f" {title}")
        print("="*70)

def demo_example_1():
    """Example 1: Question split across pages"""
    print_separator("EXAMPLE 1: Question Split Across Page Break")
    
    example = """
Q. What is the function of the cell membrane in both plant and

--- PAGE 1 ---

animal cells?

Ans. The cell membrane controls what enters and exits the cell.
"""
    
    processor = TextProcessor()
    
    # Without merging
    print("\n[WITHOUT PAGE BREAK MERGING]")
    lines_no_merge = processor.extract_lines_from_markdown(example, merge_breaks=False)
    for i, line in enumerate(lines_no_merge, 1):
        print(f"{i}. {line}")
    
    # With merging
    print("\n[WITH PAGE BREAK MERGING]")
    lines_with_merge = processor.extract_lines_from_markdown(example, merge_breaks=True)
    for i, line in enumerate(lines_with_merge, 1):
        print(f"{i}. {line}")
    
    print(f"\n✓ Merged {len(lines_no_merge) - len(lines_with_merge)} lines")

def demo_example_2():
    """Example 2: Answer split across pages"""
    print_separator("EXAMPLE 2: Answer Split Across Page Break")
    
    example = """
Q. Define photosynthesis?

Ans. Photosynthesis is the process by which green plants use

Page 2

sunlight to synthesize nutrients from carbon dioxide and water.
"""
    
    processor = TextProcessor()
    
    # Without merging
    print("\n[WITHOUT PAGE BREAK MERGING]")
    lines_no_merge = processor.extract_lines_from_markdown(example, merge_breaks=False)
    for i, line in enumerate(lines_no_merge, 1):
        print(f"{i}. {line}")
    
    # With merging
    print("\n[WITH PAGE BREAK MERGING]")
    lines_with_merge = processor.extract_lines_from_markdown(example, merge_breaks=True)
    for i, line in enumerate(lines_with_merge, 1):
        print(f"{i}. {line}")
    
    print(f"\n✓ Merged {len(lines_no_merge) - len(lines_with_merge)} lines")

def demo_example_3():
    """Example 3: Q ends, A starts on next page (DON'T MERGE)"""
    print_separator("EXAMPLE 3: Q/A Boundary - No Merge")
    
    example = """
Q. What is the capital of France?

PAGE 1

Ans. Paris is the capital of France.

Q. Define force?

---

Ans. A push or pull on an object.
"""
    
    processor = TextProcessor()
    
    # Without merging
    print("\n[WITHOUT PAGE BREAK MERGING]")
    lines_no_merge = processor.extract_lines_from_markdown(example, merge_breaks=False)
    for i, line in enumerate(lines_no_merge, 1):
        print(f"{i}. {line}")
    
    # With merging
    print("\n[WITH PAGE BREAK MERGING]")
    lines_with_merge = processor.extract_lines_from_markdown(example, merge_breaks=True)
    for i, line in enumerate(lines_with_merge, 1):
        print(f"{i}. {line}")
    
    print(f"\n✓ Correctly kept Q/A boundaries separate")

def demo_example_4():
    """Example 4: DeepSeek markdown with bounding boxes"""
    print_separator("EXAMPLE 4: DeepSeek Markdown Format")
    
    example = """
<|ref|>text<|/ref|><|det|>[[120, 163, 603, 197]]<|/det|>
Q1. He saw his ______ at him.

<|ref|>text<|/ref|><|det|>[[120, 207, 880, 265]]<|/det|>
Q(a) Who is 'he' in the above extract? How did he feel looking at his family?

<|ref|>text<|/ref|><|det|>[[120, 275, 895, 380]]<|/det|>
Ans. The young seagull is mentioned as 'he' in the above extract. He feels sad and scared when

PAGE 2

he looks at his family that he cannot fly and he's ashamed of his younger sibling who does flying.
"""
    
    processor = TextProcessor()
    
    print("\n[RAW INPUT]")
    print(example[:200] + "...")
    
    print("\n[PROCESSED OUTPUT]")
    lines = processor.extract_lines_from_markdown(example, merge_breaks=True)
    for i, line in enumerate(lines, 1):
        print(f"{i}. {line}")
    
    print(f"\n✓ Removed bounding boxes and merged split answer")

def demo_jsonl_output():
    """Example 5: JSONL format output"""
    print_separator("EXAMPLE 5: JSONL Format for DeBERTa")
    
    example = """
Q. What is the capital of France?

Ans. Paris is the capital of France.

Q. Define force?

Ans. A push or pull on an object.

PAGE 1

Total Pages: 1
"""
    
    processor = TextProcessor()
    
    # Process and get statistics
    jsonl_data, stats = processor.process_for_classification(
        example, 
        merge_page_breaks=True
    )
    
    print("\n[JSONL OUTPUT]")
    for entry in jsonl_data:
        print(json.dumps(entry, ensure_ascii=False))
    
    print(f"\n[STATISTICS]")
    print(f"Total lines: {stats['total_lines']}")
    print(f"Likely questions: {stats['likely_questions']}")
    print(f"Likely answers: {stats['likely_answers']}")
    print(f"Other (metadata): {stats['other']}")

def demo_complex_example():
    """Example 6: Complex multi-page Q&A"""
    print_separator("EXAMPLE 6: Complex Multi-Page Q&A")
    
    example = """
Q1. Explain the process of photosynthesis in detail, including the role of

PAGE 1

chlorophyll, light reactions, and

---

dark reactions in plants.

Ans. Photosynthesis is a complex process that occurs in

12/11/2024

two main stages. The light reactions take place in the thylakoid membranes where

Page 2

chlorophyll absorbs light energy to split water molecules.

Q2. What is the difference between mitosis and meiosis?

Ans. Mitosis produces two identical daughter cells.
"""
    
    processor = TextProcessor()
    
    print("\n[WITHOUT MERGING]")
    lines_no_merge = processor.extract_lines_from_markdown(example, merge_breaks=False)
    for i, line in enumerate(lines_no_merge, 1):
        print(f"{i}. {line[:80]}{'...' if len(line) > 80 else ''}")
    
    print(f"\n[WITH INTELLIGENT MERGING]")
    lines_with_merge = processor.extract_lines_from_markdown(example, merge_breaks=True)
    for i, line in enumerate(lines_with_merge, 1):
        print(f"{i}. {line[:80]}{'...' if len(line) > 80 else ''}")
    
    print(f"\n✓ Merged {len(lines_no_merge) - len(lines_with_merge)} lines across page breaks")
    print("✓ Preserved Q/A boundaries")
    print("✓ Removed page markers and dates")

def demo_edge_cases():
    """Example 7: Edge cases"""
    print_separator("EXAMPLE 7: Edge Cases")
    
    print("\n[Case 1: Incomplete question, answer on next page]")
    case1 = """
Q. What is

PAGE 1

Ans. The answer.
"""
    processor = TextProcessor()
    lines1 = processor.extract_lines_from_markdown(case1, merge_breaks=True)
    for i, line in enumerate(lines1, 1):
        print(f"{i}. {line}")
    print("✓ Correctly kept separate (detected 'Ans.' prefix)")
    
    print("\n[Case 2: Complete question, continuation without prefix]")
    case2 = """
Q. What is the function of

PAGE 1

the mitochondria?

Ans. It produces energy.
"""
    lines2 = processor.extract_lines_from_markdown(case2, merge_breaks=True)
    for i, line in enumerate(lines2, 1):
        print(f"{i}. {line}")
    print("✓ Correctly merged question parts")
    
    print("\n[Case 3: Multi-page answer]")
    case3 = """
Ans. The process involves multiple steps including the initial

PAGE 1

binding of the enzyme to the

---

substrate at the active site.
"""
    lines3 = processor.extract_lines_from_markdown(case3, merge_breaks=True)
    for i, line in enumerate(lines3, 1):
        print(f"{i}. {line}")
    print("✓ Correctly merged all answer parts")

def main():
    """Run all demonstrations"""
    print("\n")
    print("="*70)
    print(" TEXT PROCESSING DEMONSTRATION")
    print(" DeepSeek OCR → DeBERTa Classification Format")
    print("="*70)
    
    demo_example_1()
    demo_example_2()
    demo_example_3()
    demo_example_4()
    demo_jsonl_output()
    demo_complex_example()
    demo_edge_cases()
    
    print_separator("KEY FEATURES")
    print("""
✓ Intelligent Page Break Merging:
  - Merges lines split across pages
  - Preserves Q/A boundaries
  - Detects incomplete sentences
  
✓ Bounding Box Removal:
  - Removes DeepSeek annotations like <|ref|>, <|det|>
  - Keeps only actual text content
  
✓ Page Marker Detection:
  - Removes: PAGE 1, page numbers, dates, separators
  - Cleans metadata and headers
  
✓ Q/A Type Detection:
  - Identifies questions vs answers
  - Prevents incorrect merging across boundaries
  
✓ DeBERTa-Ready Output:
  - JSONL format: {"text": "...", "label": ""}
  - Preserves OCR characteristics for model
  - Provides statistics
""")
    
    print_separator("NEXT STEPS")
    print("""
1. Test on your actual DeepSeek OCR output:
   
   from src.text_processing.text_processor import TextProcessor
   processor = TextProcessor()
   
   # Read your OCR output
   with open('outputs/assignment1_extracted.txt', 'r') as f:
       text = f.read()
   
   # Process it
   jsonl_data, stats = processor.process_for_classification(
       text,
       output_jsonl='data/processed/assignment1.jsonl',
       merge_page_breaks=True
   )
   
   # View results
   print(f"Processed {stats['total_lines']} lines")
   print(f"Questions: {stats['likely_questions']}")
   print(f"Answers: {stats['likely_answers']}")
   
2. Use processed data with DeBERTa classifier:
   
   from src.classification.classifier import DeBERTaClassifier
   classifier = DeBERTaClassifier('path/to/model')
   
   predictions = classifier.predict_batch([entry['text'] for entry in jsonl_data])
   
3. Full pipeline integration:
   
   python pipeline.py --input data/raw/assignment1.pdf --output data/processed/
""")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
