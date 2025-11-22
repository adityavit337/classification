"""DeBERTa-based text classifier for question/answer classification."""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from typing import Union, List, Dict
import logging

logger = logging.getLogger(__name__)


class DeBERTaClassifier:
    """Classify text as question or answer using trained DeBERTa model."""
    
    def __init__(self, model_path: Union[str, Path], device: str = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to the trained DeBERTa model
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model from: {self.model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path)
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
    def preprocess_text(self, text: str, max_length: int = 512) -> Dict:
        """
        Preprocess text for model input.
        
        Args:
            text: Input text to classify
            max_length: Maximum sequence length
            
        Returns:
            Tokenized input dictionary
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Classify a single text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with predicted label and confidence score
        """
        # Preprocess
        inputs = self.preprocess_text(text)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        predicted_label = self.id2label[predicted_class]
        
        return {
            'text': text,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'all_probabilities': {
                self.id2label[i]: probabilities[0][i].item() 
                for i in range(len(self.id2label))
            }
        }
    
    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """
        Classify multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1)
            
            # Process results
            for j, text in enumerate(batch_texts):
                predicted_class = predicted_classes[j].item()
                confidence = probabilities[j][predicted_class].item()
                predicted_label = self.id2label[predicted_class]
                
                results.append({
                    'text': text,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'all_probabilities': {
                        self.id2label[k]: probabilities[j][k].item() 
                        for k in range(len(self.id2label))
                    }
                })
        
        return results
    
    def classify_with_threshold(self, text: str, 
                               threshold: float = 0.5) -> Dict[str, Union[str, float]]:
        """
        Classify text with confidence threshold.
        
        Args:
            text: Input text to classify
            threshold: Minimum confidence required for classification
            
        Returns:
            Prediction dictionary with 'uncertain' label if below threshold
        """
        result = self.predict(text)
        
        if result['confidence'] < threshold:
            result['predicted_label'] = 'uncertain'
            logger.warning(f"Low confidence prediction: {result['confidence']:.2f}")
        
        return result
