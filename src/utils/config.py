"""Configuration settings for the classification pipeline."""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_path: str = None):
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'ocr': {
                'model_name': 'deepseek-ai/DeepSeek-OCR'
            },
            'text_processing': {
                'similarity_threshold': 0.8,
                'max_chunk_length': 512,
                'chunk_overlap': 50
            },
            'classification': {
                'model_path': 'models/deberta_classifier',
                'batch_size': 8,
                'confidence_threshold': 0.5,
                'max_length': 512
            },
            'paths': {
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'models': 'models',
                'outputs': 'outputs',
                'logs': 'logs'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def save(self, output_path: str = None):
        """Save configuration to YAML file."""
        output_path = output_path or self.config_path
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
