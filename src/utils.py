import os
import json
import logging
from dotenv import load_dotenv

# Load local .env if running outside Lambda
load_dotenv()

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_env_var(key: str, default=None, required=True):
    """
    Fetch environment variable with optional default.
    If required and missing, raises ValueError.
    """
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value

def log_event(event):
    """
    Pretty-print incoming Lambda event for debugging.
    Avoid logging sensitive values.
    """
    try:
        safe_event = dict(event)
        if "body" in safe_event:
            safe_event["body"] = "<omitted>"
        logger.info(f"Event received: {json.dumps(safe_event)}")
    except Exception as e:
        logger.warning(f"Could not log event: {e}")
