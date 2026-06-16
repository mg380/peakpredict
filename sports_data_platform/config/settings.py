"""
Configuration settings for the Sports Data Platform.
"""

import os
from typing import Dict, Any

def get_config(profile: str = "default") -> Dict[str, Any]:
    """
    Get configuration settings for the specified profile.
    
    Args:
        profile: Configuration profile name
        
    Returns:
        Configuration dictionary
    """
    # Base configuration
    base_config = {
        "base_url": "https://tilastopaja.info",
        "username_env": "SPORTS_DATA_USER",
        "password_env": "SPORTS_DATA_PASS",
        "headless": False,  # Changed to False to show the browser window
        "max_retries": 3,
        "retry_delay": 5,
        "use_database": True,
        "db_connection_string": os.environ.get("SPORTS_DATA_DB", "sqlite:///sports_data.db"),
        "use_hdf5": True,
        "hdf5_dir": "data",
        "concurrency": {
            "max_workers": 4,
            "chunk_size": 10
        }
    }
    
    # Profile-specific configurations
    profiles = {
        "default": {},
        
        "development": {
            "headless": False,
            "max_retries": 1,
            "db_connection_string": "sqlite:///data/dev/sports_data_dev.db",
            "hdf5_dir": "data/dev",
            # For development, use a mock URL that doesn't require real credentials
            "base_url": "https://httpbin.org",
        },
        
        "testing": {
            "base_url": "https://httpbin.org",
            "db_connection_string": "sqlite:///data/test/sports_data_test.db",
            "hdf5_dir": "data/test"
        },
        
        "production": {
            "max_retries": 5,
            "retry_delay": 10,
            "db_connection_string": os.environ.get("SPORTS_DATA_DB_PROD"),
            "hdf5_dir": "/var/data/sports_data",
            "concurrency": {
                "max_workers": 8,
                "chunk_size": 20
            }
        }
    }
    
    # Get profile configuration or use default
    profile_config = profiles.get(profile, profiles["default"])
    
    # Merge base config with profile config
    config = {**base_config, **profile_config}
    
    return config
