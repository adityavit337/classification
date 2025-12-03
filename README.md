# OCR + Classification Pipeline

A complete pipeline for extracting text from PDF files using **Qwen3-VL-4B** OCR and classifying it into questions/answers/metadata using **Qwen3-4B-Instruct-2507** few-shot learning.


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


## Configuration

Edit `config/config.yaml` for custom settings:

```yaml
ocr:
  model_name: 'Qwen/Qwen3-VL-4B-Instruct'
  
classification:
  model_name: 'Qwen/Qwen3-4B-Instruct-2507'
```

