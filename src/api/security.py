# src/api/security.py
from fastapi import Header, HTTPException, status
from src.utils import get_env_var

API_KEY = get_env_var("API_KEY", "", required=False)

def api_key_auth(x_api_key: str = Header(default="")) -> bool:
    """
    FastAPI dependency. If API_KEY env var is empty, auth is disabled (useful for dev).
    """
    if not API_KEY:
        return True
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return True
