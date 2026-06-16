"""
Error handling utilities for the Sports Data Platform.
"""

import logging
import time
import functools
from typing import Callable, Any, TypeVar, Optional

F = TypeVar('F', bound=Callable[..., Any])

def retry(max_retries: int = 3, delay: int = 1, backoff: int = 2, 
         exceptions: tuple = (Exception,), logger: Optional[logging.Logger] = None):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
        logger: Logger instance
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            local_logger = logger or logging.getLogger(func.__module__)
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        local_logger.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    wait_time = delay * (backoff ** (retries - 1))
                    local_logger.warning(f"Retry {retries}/{max_retries} for {func.__name__} in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
        return wrapper  # type: ignore
    return decorator

class ErrorTracker:
    """Track and manage errors during processing."""
    
    def __init__(self, max_errors: int = 10):
        """
        Initialize the error tracker.
        
        Args:
            max_errors: Maximum number of errors before raising an exception
        """
        self.max_errors = max_errors
        self.errors = []
        self.logger = logging.getLogger(__name__)
    
    def add_error(self, error: Exception, context: Optional[str] = None):
        """
        Add an error to the tracker.
        
        Args:
            error: Exception object
            context: Optional context information
            
        Raises:
            RuntimeError: If max_errors is reached
        """
        error_info = {
            "error": error,
            "type": type(error).__name__,
            "message": str(error),
            "context": context,
            "timestamp": time.time()
        }
        
        self.errors.append(error_info)
        self.logger.error(f"Error {len(self.errors)}/{self.max_errors}: {error_info['message']} [{context or 'unknown'}]")
        
        if len(self.errors) >= self.max_errors:
            raise RuntimeError(f"Maximum error threshold reached ({self.max_errors})")
    
    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return len(self.errors) > 0
    
    def get_error_summary(self) -> dict:
        """Get a summary of recorded errors."""
        error_types = {}
        for error in self.errors:
            error_type = error["type"]
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        return {
            "total": len(self.errors),
            "types": error_types
        }
