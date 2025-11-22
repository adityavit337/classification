"""Main pipeline script for text classification from images."""

import argparse
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from src.ocr import OCRExtractor
from src.text_processing import TextProcessor
from src.classification import DeBERTaClassifier
from src.utils import Config
from src.utils.logger import setup_logging


def process_single_image(image_path: Path, 
                        ocr_extractor: OCRExtractor,
                        text_processor: TextProcessor,
                        classifier: DeBERTaClassifier) -> Dict:
    """Process a single image or PDF file through the complete pipeline."""
    
    # Step 1: Extract text with DeepSeek-OCR (no preprocessing needed)
    text = ocr_extractor.extract_text(image_path)
    
    # Step 2: Process text
    cleaned_text = text_processor.clean_text(text)
    
    # Step 3: Classify text
    result = classifier.predict(cleaned_text)
    
    # Add metadata
    result['file_path'] = str(image_path)
    result['file_type'] = 'pdf' if str(image_path).lower().endswith('.pdf') else 'image'
    result['extracted_text'] = text
    
    return result


def process_directory(input_dir: Path,
                     output_path: Path,
                     config: Config):
    """Process all images in a directory."""
    
    # Initialize components
    logger.info("Initializing pipeline components...")
    
    ocr_extractor = OCRExtractor(
        model_name=config.get('ocr.model_name', 'deepseek-ai/DeepSeek-OCR')
    )
    
    text_processor = TextProcessor(
        similarity_threshold=config.get('text_processing.similarity_threshold')
    )
    
    classifier = DeBERTaClassifier(
        model_path=config.get('classification.model_path')
    )
    
    # Find all image and PDF files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf'}
    image_files = [
        f for f in input_dir.rglob('*') 
        if f.suffix.lower() in image_extensions
    ]
    
    logger.info(f"Found {len(image_files)} files to process (images and PDFs)")
    
    # Process each image or PDF
    results = []
    for image_path in tqdm(image_files, desc="Processing files"):
        try:
            result = process_single_image(
                image_path,
                ocr_extractor,
                text_processor,
                classifier
            )
            results.append(result)
            logger.info(f"Processed {image_path.name}: {result['predicted_label']} "
                       f"(confidence: {result['confidence']:.2f})")
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            results.append({
                'file_path': str(image_path),
                'error': str(e),
                'predicted_label': None,
                'confidence': None
            })
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    successful = sum(1 for r in results if r.get('predicted_label'))
    questions = sum(1 for r in results if r.get('predicted_label') == 'question')
    answers = sum(1 for r in results if r.get('predicted_label') == 'answer')
    
    logger.info(f"\nProcessing Summary:")
    logger.info(f"Total files: {len(image_files)}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Questions: {questions}")
    logger.info(f"Answers: {answers}")
    logger.info(f"Failed: {len(image_files) - successful}")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Text Classification Pipeline from Images'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory containing images and/or PDF files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/results.json',
        help='Output JSON file path (default: outputs/results.json)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    import logging
    log_level = getattr(logging, args.log_level)
    global logger
    logger = setup_logging(log_level=log_level)
    
    # Load configuration
    config = Config(args.config)
    logger.info(f"Configuration loaded from: {config.config_path}")
    
    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Process directory
    output_path = Path(args.output)
    process_directory(input_dir, output_path, config)


if __name__ == '__main__':
    main()
