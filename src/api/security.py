import os

from fastapi import Header, HTTPException, status, Depends
from src.utils import get_env_var

API_KEY = os.getenv("API_KEY", "change-me-please")

def api_key_auth(x_api_key: str = Header(default="")):
    if not API_KEY:
        # Auth disabled if API_KEY is empty
        return True
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    return True


def check_api_key(headers: dict) -> bool:
    """
    Validate API key from headers.
    Returns True if valid, False otherwise.
    """
    if not headers:
        return False
    key = headers.get("x-api-key")
    return key == API_KEY
