"""
Concurrency utilities for the Sports Data Platform.
"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable, Any, TypeVar, Generic, Iterable

T = TypeVar('T')
R = TypeVar('R')

class ParallelProcessor:
    """Utility for parallel processing of tasks."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    def process_batch(self, items: List[T], process_func: Callable[[T], R]) -> List[R]:
        """
        Process a batch of items in parallel.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            
        Returns:
            List of results
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_func, item) for item in items]
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel processing: {str(e)}")
        
        return results
    
    def process_chunks(self, items: List[T], process_func: Callable[[List[T]], List[R]], 
                      chunk_size: int = 10) -> List[R]:
        """
        Process items in chunks.
        
        Args:
            items: List of items to process
            process_func: Function to process each chunk
            chunk_size: Size of each chunk
            
        Returns:
            Combined list of results
        """
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_func, chunk) for chunk in chunks]
            for future in futures:
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    self.logger.error(f"Error in chunk processing: {str(e)}")
        
        return all_results

async def process_async(items: Iterable[T], process_func: Callable[[T], R], 
                       max_concurrency: int = 10) -> List[R]:
    """
    Process items asynchronously with concurrency limit.
    
    Args:
        items: Iterable of items to process
        process_func: Async function to process each item
        max_concurrency: Maximum number of concurrent tasks
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_with_semaphore(item):
        async with semaphore:
            return await process_func(item)
    
    return await asyncio.gather(*(process_with_semaphore(item) for item in items))
