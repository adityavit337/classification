"""Logging configuration."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = 'logs', log_level: int = logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
    """
    # Create logs directory
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'classification_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger
