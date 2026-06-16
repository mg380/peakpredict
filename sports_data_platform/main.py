"""
Main entry point for the Sports Data Platform.
"""

import argparse
import asyncio
import logging
import sys
from dotenv import load_dotenv

# Add the project root directory to the Python path
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.settings import get_config
from utils.logging_utils import setup_logging
from pipeline.orchestrator import PipelineOrchestrator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sports Data Platform")
    parser.add_argument("--events", action="store_true", help="Scrape events")
    parser.add_argument("--athletes", action="store_true", help="Scrape athletes")
    parser.add_argument("--performances", action="store_true", help="Scrape performances")
    parser.add_argument("--year", type=int, help="Year filter for events")
    parser.add_argument("--event-type", type=str, help="Event type filter")
    parser.add_argument("--config", type=str, default="default", 
                      help="Configuration profile")
    parser.add_argument("--log-level", type=str, default="INFO", 
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
    return parser.parse_args()

async def main():
    """Main entry point for the application."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Sports Data Platform")
    
    # Load configuration
    config = get_config(args.config)
    
    # Create pipeline orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    # Setup pipeline components
    if not orchestrator.setup():
        logger.error("Failed to set up pipeline components")
        return {"success": False}
    
    try:
        # Run requested pipelines
        results = {}
        
        if args.events:
            logger.info("Running events pipeline")
            events = orchestrator.run_event_pipeline(year=args.year, event_type=args.event_type)
            results["events"] = len(events)
        
        if args.athletes:
            logger.info("Running athletes pipeline")
            athletes = orchestrator.run_athlete_pipeline()
            results["athletes"] = len(athletes)
        
        if args.performances:
            logger.info("Running performances pipeline")
            performances = orchestrator.run_performance_pipeline()
            results["performances"] = len(performances)
        
        # If no specific pipeline was requested, run all
        if not (args.events or args.athletes or args.performances):
            logger.info("Running all pipelines")
            events = orchestrator.run_event_pipeline(year=args.year, event_type=args.event_type)
            athletes = orchestrator.run_athlete_pipeline()
            performances = orchestrator.run_performance_pipeline()
            
            results = {
                "events": len(events),
                "athletes": len(athletes),
                "performances": len(performances)
            }
        
        logger.info(f"Pipeline execution completed: {results}")
        return {"success": True, "results": results}
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return {"success": False, "error": str(e)}
    
    finally:
        # Clean up resources
        orchestrator.close()

def main_cli():
    """Entry point for command-line interface."""
    return asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())
