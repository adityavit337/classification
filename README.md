# OCR + Classification Pipeline

A complete pipeline for extracting text from PDF files using **Qwen3-VL-4B** OCR and classifying it into questions/answers/metadata using **Qwen3-4B-Instruct-2507** few-shot learning.

## Features

- 🔍 **OCR Extraction**: Qwen3-VL-4B vision-language model for accurate text extraction
- 🏷️ **Few-Shot Classification**: Qwen3-4B-Instruct with carefully crafted examples
- 📦 **Smart Line Merging**: Automatically groups related lines into logical blocks
- ⚡ **Pattern Matching**: Fast regex-based classification for obvious cases
- 📊 **Organized Output**: Separate files for questions, answers, and full results

## Project Structure

```
Classification/
├── src/
│   ├── ocr/                   # Qwen3-VL-4B OCR module
│   │   ├── __init__.py
│   │   └── ocr_extractor.py
│   ├── text_processing/       # Text processing and merging
│   │   ├── __init__.py
│   │   └── text_processor.py
│   ├── classification/        # Qwen3-4B-Instruct classifier
│   │   ├── __init__.py
│   │   └── qwen_classifier.py
│   └── utils/                 # Utilities (config, logging)
├── data/
│   └── raw/                   # Input PDF files
├── models/                    # Model storage (optional)
├── config/
│   └── config.yaml           # Configuration
├── outputs/                   # Classification results
├── logs/                     # Log files
├── pipeline.py              # Main entry point
├── requirements.txt          
└── README.md
```

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Install Poppler** (for PDF processing):
   - **Windows**: Download from https://github.com/oschwartz10612/poppler-windows/releases/ and add to PATH
   - **Linux**: `sudo apt-get install poppler-utils`
   - **Mac**: `brew install poppler`

3. Models are automatically downloaded from HuggingFace on first use:
   - `Qwen/Qwen3-VL-4B-Instruct` (~8GB) - OCR
   - `Qwen/Qwen3-4B-Instruct-2507` (~8GB) - Classification

## Usage

### Full Pipeline (PDF → Classification)

```bash
# Process a PDF file
python pipeline.py data/raw/test_2.pdf

# Specify output directory
python pipeline.py data/raw/assignment1.pdf --output results/
```

### Classification Only (Pre-extracted Text)

```bash
# Classify already extracted text
python pipeline.py --text outputs/test_2_extracted.txt
```

### Python API

```python
from src.classification import QwenClassifier

# Initialize classifier
classifier = QwenClassifier()

# Classify text lines
lines = ["Q1: What is AWS?", "Ans: Amazon Web Services", "Page 1"]
results = classifier.classify_document(lines)

print(f"Questions: {len(results['questions'])}")
print(f"Answers: {len(results['answers'])}")
print(f"Metadata: {len(results['metadata'])}")
```

## Output Files

After running the pipeline, you'll find:

```
outputs/
├── {filename}_extracted.txt    # Raw OCR text
├── {filename}_results.json     # Full classification results
├── {filename}_questions.txt    # Extracted questions
└── {filename}_answers.txt      # Extracted answers
```

### Results JSON Format

```json
{
  "questions": [
    {"text": "Q1: What is...", "confidence": 0.98, "reasoning": "Pattern matched"}
  ],
  "answers": [...],
  "metadata": [...],
  "statistics": {
    "total_original_lines": 174,
    "total_merged_blocks": 102,
    "questions_count": 10,
    "answers_count": 29,
    "metadata_count": 63
  }
}
```

## Classification Categories

| Category | Examples |
|----------|----------|
| **question** | `Q1:`, `Ques-1:`, interrogative sentences, problem scenarios |
| **answer** | `Ans:`, `Answer:`, technical explanations, solutions |
| **metadata** | Student names, page numbers, section headers, dates |

## Configuration

Edit `config/config.yaml` for custom settings:

```yaml
ocr:
  model_name: 'Qwen/Qwen3-VL-4B-Instruct'
  
classification:
  model_name: 'Qwen/Qwen3-4B-Instruct-2507'
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- ~16GB disk space for models
