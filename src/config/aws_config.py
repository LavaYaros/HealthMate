import sys
from pathlib import Path

# add project root (parent of "src") to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import boto3
from src.logging.logger import setup_logger
from src.config.settings import get_settings

logger = setup_logger(__name__)
settings = get_settings()

def configure_aws():
    """Configure AWS credentials from settings."""
    try:
        # Configure AWS credentials
        boto3.setup_default_session(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        logger.info("AWS credentials configured successfully")
    except Exception as e:
        logger.error(f"Error configuring AWS credentials: {str(e)}")
        raise 