import logging
import sys
from typing import Optional
from pathlib import Path
from src.config.settings import get_settings

def setup_logger(
    name: str,
    log_level: Optional[int] = None,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Set up a logger with console and file handlers using singleton pattern.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: DEBUG if DEBUG_MODE is True, else INFO)
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """

    settings = get_settings()
    if log_level is None:
        log_level = logging.DEBUG if settings.DEBUG_MODE else logging.INFO
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    if logger.hasHandlers(): # to ensure that only one logger handler is active
        logger.handlers.clear()

    logger.propagate = False
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger 