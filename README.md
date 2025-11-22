# Text Classification Pipeline

A complete pipeline for extracting text from images and PDF files using DeepSeek-OCR and classifying it as questions or answers using a trained DeBERTa model.

## Project Structure

```
Classification/
├── src/
│   ├── ocr/                   # DeepSeek-OCR text extraction module
│   │   ├── __init__.py
│   │   └── ocr_extractor.py
│   ├── text_processing/       # Text processing and merging module
│   │   ├── __init__.py
│   │   └── text_processor.py
│   ├── classification/        # DeBERTa classification module
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logger.py
│   └── __init__.py
├── data/
│   ├── raw/                   # Input images
│   └── processed/             # (Not used - kept for compatibility)
├── models/                    # Trained models directory
├── config/
│   └── config.yaml           # Configuration file
├── outputs/                   # Classification results
├── logs/                     # Log files
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── pipeline.py              # Main pipeline script
└── README.md                # This file
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Poppler (required for PDF processing):
   - **Windows**: Download from https://github.com/oschwartz10612/poppler-windows/releases/ and add to PATH
   - **Linux**: `sudo apt-get install poppler-utils`
   - **Mac**: `brew install poppler`

3. The DeepSeek-OCR model will be automatically downloaded from HuggingFace on first use

4. Place your trained DeBERTa model in the `models/deberta_classifier` directory

## Usage

### Full Pipeline

Run the complete pipeline on your images and/or PDF files:

```bash
python pipeline.py --input data/raw --output outputs/results.json
```

The pipeline automatically detects and processes both image files (.jpg, .jpeg, .png, .tiff, .bmp) and PDF files (.pdf).

### Individual Modules

You can also use individual modules:

```python
from src.ocr import OCRExtractor
from src.text_processing import TextProcessor
from src.classification import DeBERTaClassifier

# Extract text using DeepSeek-OCR (works with both images and PDFs)
ocr = OCRExtractor(model_name='deepseek-ai/DeepSeek-OCR')

# From image
text = ocr.extract_text('image.jpg')

# From PDF (automatically processes all pages)
text = ocr.extract_text('document.pdf')

# Process text
processor = TextProcessor()
processed_text = processor.clean_text(text)

# Classify
classifier = DeBERTaClassifier('models/deberta_classifier')
result = classifier.predict(processed_text)
```

## Configuration

Edit `config/config.yaml` to customize:
- DeepSeek-OCR model settings
- Text processing thresholds
- Classification model settings
- File paths

## Pipeline Steps

1. **OCR Extraction**: Extract text directly from images and PDFs using DeepSeek-OCR
   - State-of-the-art OCR model from DeepSeek AI
   - Supports both image files and multi-page PDF documents
   - No preprocessing required
   - High accuracy on various image types
   - Automatic model download from HuggingFace

2. **Text Processing**: Clean and merge text from multiple pages
   - Remove artifacts
   - Normalize whitespace
   - Merge related pages based on similarity
   - Split into chunks if needed

3. **Classification**: Classify text as question or answer
   - Uses trained DeBERTa model
   - Provides confidence scores
   - Batch processing support

## Testing

Run tests:
```bash
pytest tests/
```

## License

MIT License
