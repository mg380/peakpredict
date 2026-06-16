"""
Credential management for the Sports Data Platform.
"""

import os
from typing import Dict, Optional

def get_credentials() -> Dict[str, str]:
    """
    Get credentials from environment variables.
    
    Returns:
        Dictionary with username and password
    """
    username = os.environ.get("SPORTS_DATA_USER")
    password = os.environ.get("SPORTS_DATA_PASS")
    
    if not username or not password:
        raise ValueError(
            "Credentials not found in environment variables. "
            "Please set SPORTS_DATA_USER and SPORTS_DATA_PASS."
        )
    
    return {
        "username": username,
        "password": password
    }

def set_credentials_env(username: str, password: str) -> None:
    """
    Set credentials as environment variables.
    
    Args:
        username: Username for authentication
        password: Password for authentication
    """
    os.environ["SPORTS_DATA_USER"] = username
    os.environ["SPORTS_DATA_PASS"] = password
    
def get_api_key(service: str) -> Optional[str]:
    """
    Get API key for a specific service from environment variables.
    
    Args:
        service: Service name
        
    Returns:
        API key if available, None otherwise
    """
    env_var = f"SPORTS_DATA_{service.upper()}_API_KEY"
    return os.environ.get(env_var)
