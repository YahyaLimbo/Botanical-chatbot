import logging
import sys
from pathlib import Path

from src.config.settings import settings


def setup_logging():
    """Configure application logging"""
    log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
   
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
  
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
   
    file_handler = logging.FileHandler(log_dir / "cinny_ai.log")
    file_handler.setLevel(log_level)
 
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
 
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module"""
    return logging.getLogger(name)

setup_logging()
