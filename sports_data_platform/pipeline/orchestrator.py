"""
Pipeline orchestrator for the Sports Data Platform.
"""

import logging
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import h5py
from datetime import datetime

from core.session_manager import SessionManager
from scrapers.event_scraper import EventScraper
from scrapers.athlete_scraper import AthleteScraper
from scrapers.performance_scraper import PerformanceScraper
from storage.database_manager import DatabaseManager
from storage.database import Event, Athlete, Performance

class PipelineOrchestrator:
    """Orchestrates the entire data pipeline."""
    
    def __init__(self, config):
        """
        Initialize the orchestrator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Components will be initialized in setup()
        self.session_manager = None
        self.event_scraper = None
        self.athlete_scraper = None
        self.performance_scraper = None
        self.db_manager = None
        
    def setup(self):
        """
        Set up all components needed for the pipeline.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            # Initialize session manager
            self.logger.info("Initializing session manager")
            self.session_manager = SessionManager(
                base_url=self.config['base_url'],
                username_env=self.config.get('username_env', 'SPORTS_DATA_USER'),
                password_env=self.config.get('password_env', 'SPORTS_DATA_PASS'),
                headless=self.config.get('headless', True)
            )
            
            # Initialize scrapers
            self.logger.info("Initializing scrapers")
            self.event_scraper = EventScraper(
                self.session_manager,
                max_retries=self.config.get('max_retries', 3),
                retry_delay=self.config.get('retry_delay', 5)
            )
            
            self.athlete_scraper = AthleteScraper(
                self.session_manager,
                max_retries=self.config.get('max_retries', 3),
                retry_delay=self.config.get('retry_delay', 5)
            )
            
            self.performance_scraper = PerformanceScraper(
                self.session_manager,
                max_retries=self.config.get('max_retries', 3),
                retry_delay=self.config.get('retry_delay', 5)
            )
            
            # Initialize database manager
            if self.config.get('use_database', True):
                self.logger.info("Initializing database manager")
                
                # Ensure the directory exists for the database
                db_path = self.config['db_connection_string']
                if db_path.startswith('sqlite:///'):
                    import os
                    db_file = db_path.replace('sqlite:///', '')
                    os.makedirs(os.path.dirname(os.path.abspath(db_file)), exist_ok=True)
                
                self.db_manager = DatabaseManager(self.config['db_connection_string'])
                
                # Create tables if they don't exist
                self.db_manager.create_tables()
            
            return True
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            return False
    
    def run_event_pipeline(self, year: Optional[int] = None, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run the event scraping pipeline.
        
        Args:
            year: Optional year to filter events
            event_type: Optional event type to filter events
            
        Returns:
            List of event dictionaries
        """
        self.logger.info(f"Running event pipeline for year={year}, type={event_type}")
        
        # Scrape all events instead of just one category
        events = self.event_scraper.scrape_all_events()
        self.logger.info(f"Scraped {len(events)} events")
        
        # Store in database if enabled
        if self.db_manager and events:
            self._store_events_in_db(events)
        
        # Store in HDF5 if enabled
        if self.config.get('use_hdf5', True) and events:
            self._store_events_in_hdf5(events)
        
        return events
    
    def run_athlete_pipeline(self) -> List[Dict[str, Any]]:
        """
        Run the athlete scraping pipeline.
        
        Returns:
            List of athlete dictionaries
        """
        self.logger.info("Running athlete pipeline")
        
        # Get list of athletes
        athletes = self.athlete_scraper.get_athlete_list()
        self.logger.info(f"Found {len(athletes)} athletes")
        
        # Store in database if enabled
        if self.db_manager and athletes:
            self._store_athletes_in_db(athletes)
        
        # Store in HDF5 if enabled
        if self.config.get('use_hdf5', True) and athletes:
            self._store_athletes_in_hdf5(athletes)
        
        return athletes
    
    def run_performance_pipeline(self, athlete_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Run the performance scraping pipeline.
        
        Args:
            athlete_ids: Optional list of athlete IDs to scrape performances for
            
        Returns:
            List of performance dictionaries
        """
        self.logger.info("Running performance pipeline")
        
        all_performances = []
        
        # If no athlete IDs provided, get all athletes
        if not athlete_ids:
            athletes = self.athlete_scraper.get_athlete_list()
            athlete_ids = [athlete.get('id') for athlete in athletes if athlete.get('id')]
        
        # Process each athlete
        for athlete_id in athlete_ids:
            try:
                self.logger.info(f"Scraping performances for athlete {athlete_id}")
                
                # Get event types for this athlete
                event_types = self.performance_scraper.get_athlete_events(athlete_id)
                
                # Scrape performances for each event type
                for event_type in event_types:
                    performances = self.performance_scraper.scrape_athlete_performances(
                        athlete_id, event_type=event_type
                    )
                    
                    if performances:
                        self.logger.info(f"Found {len(performances)} performances for athlete {athlete_id} in {event_type}")
                        all_performances.extend(performances)
                        
                        # Store in database if enabled
                        if self.db_manager:
                            self._store_performances_in_db(performances)
                        
                        # Store in HDF5 if enabled
                        if self.config.get('use_hdf5', True):
                            self._store_performances_in_hdf5(performances, athlete_id, event_type)
            
            except Exception as e:
                self.logger.error(f"Error processing athlete {athlete_id}: {str(e)}")
        
        self.logger.info(f"Total performances scraped: {len(all_performances)}")
        return all_performances
    
    def _store_events_in_db(self, events: List[Dict[str, Any]]) -> None:
        """
        Store events in the database.
        
        Args:
            events: List of event dictionaries
        """
        session = self.db_manager.Session()
        try:
            # Recreate tables to ensure schema is up to date
            from storage.database import Base
            Base.metadata.drop_all(self.db_manager.engine)
            Base.metadata.create_all(self.db_manager.engine)
            
            for event_data in events:
                # Create new event
                event = Event(
                    name=event_data['name'],
                    date=event_data['date'],
                    location=event_data.get('location', ''),
                    event_type=event_data.get('event_type', ''),
                    discipline=event_data.get('discipline', ''),
                    gender=event_data.get('gender', '')
                )
                session.add(event)
            
            session.commit()
            self.logger.info(f"Stored events in database")
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to store events in database: {str(e)}")
        finally:
            session.close()
    
    def _store_athletes_in_db(self, athletes: List[Dict[str, Any]]) -> None:
        """
        Store athletes in the database.
        
        Args:
            athletes: List of athlete dictionaries
        """
        session = self.db_manager.Session()
        try:
            for athlete_data in athletes:
                # Check if athlete already exists
                existing = session.query(Athlete).filter_by(name=athlete_data['name']).first()
                if existing:
                    continue
                
                # Create new athlete
                athlete = Athlete(
                    name=athlete_data['name'],
                    country=athlete_data.get('country'),
                    birth_date=self._parse_date(athlete_data.get('birth_date')) if athlete_data.get('birth_date') else None,
                    gender=athlete_data.get('gender')
                )
                session.add(athlete)
            
            session.commit()
            self.logger.info(f"Stored athletes in database")
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to store athletes in database: {str(e)}")
        finally:
            session.close()
    
    def _store_performances_in_db(self, performances: List[Dict[str, Any]]) -> None:
        """
        Store performances in the database.
        
        Args:
            performances: List of performance dictionaries
        """
        session = self.db_manager.Session()
        try:
            for perf_data in performances:
                # Get athlete and event
                athlete = session.query(Athlete).filter_by(name=perf_data['athlete_name']).first()
                event = session.query(Event).filter_by(name=perf_data['event_name']).first()
                
                if not athlete or not event:
                    continue
                
                # Check if performance already exists
                existing = session.query(Performance).filter_by(
                    athlete_id=athlete.id,
                    event_id=event.id
                ).first()
                
                if existing:
                    continue
                
                # Create new performance
                performance = Performance(
                    athlete_id=athlete.id,
                    event_id=event.id,
                    result=perf_data['result'],
                    position=perf_data.get('position'),
                    points=perf_data.get('points')
                )
                session.add(performance)
            
            session.commit()
            self.logger.info(f"Stored performances in database")
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to store performances in database: {str(e)}")
        finally:
            session.close()
    
    def _store_events_in_hdf5(self, events: List[Dict[str, Any]]) -> None:
        """
        Store events in HDF5 format.
        
        Args:
            events: List of event dictionaries
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(events)
            
            # Ensure directory exists
            os.makedirs(self.config.get('hdf5_dir', 'data'), exist_ok=True)
            
            # Save to HDF5 using pandas' HDF5 store which handles object dtypes better
            file_path = os.path.join(
                self.config.get('hdf5_dir', 'data'),
                f"events_{datetime.now().strftime('%Y%m%d')}.h5"
            )
            
            # Use pandas to_hdf instead of h5py directly
            df.to_hdf(file_path, key='events', mode='w')
            
            self.logger.info(f"Stored events in HDF5: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to store events in HDF5: {str(e)}")
    
    def _store_athletes_in_hdf5(self, athletes: List[Dict[str, Any]]) -> None:
        """
        Store athletes in HDF5 format.
        
        Args:
            athletes: List of athlete dictionaries
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(athletes)
            
            # Ensure directory exists
            os.makedirs(self.config.get('hdf5_dir', 'data'), exist_ok=True)
            
            # Save to HDF5 using pandas' HDF5 store
            file_path = os.path.join(
                self.config.get('hdf5_dir', 'data'),
                f"athletes_{datetime.now().strftime('%Y%m%d')}.h5"
            )
            
            # Use pandas to_hdf instead of h5py directly
            df.to_hdf(file_path, key='athletes', mode='w')
            
            self.logger.info(f"Stored athletes in HDF5: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to store athletes in HDF5: {str(e)}")
    
    def _store_performances_in_hdf5(self, performances: List[Dict[str, Any]], 
                                   athlete_id: str, event_type: str) -> None:
        """
        Store performances in HDF5 format.
        
        Args:
            performances: List of performance dictionaries
            athlete_id: Athlete ID
            event_type: Event type
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(performances)
            
            # Ensure directory exists
            hdf5_dir = os.path.join(self.config.get('hdf5_dir', 'data'), 'performances')
            os.makedirs(hdf5_dir, exist_ok=True)
            
            # Save to HDF5 - organize by athlete
            file_path = os.path.join(
                hdf5_dir,
                f"athlete_{athlete_id}.h5"
            )
            
            # Use pandas to_hdf with a key that includes the event type
            group_name = event_type.replace(' ', '_').lower()
            df.to_hdf(file_path, key=f'{group_name}/performances', mode='a')
            
            self.logger.info(f"Stored performances in HDF5: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to store performances in HDF5: {str(e)}")
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        Parse date string into datetime object.
        
        Args:
            date_str: Date string
            
        Returns:
            Datetime object or None if parsing fails
        """
        if not date_str:
            return None
        
        formats = [
            '%Y-%m-%d',
            '%d.%m.%Y',
            '%m/%d/%Y',
            '%d %b %Y',
            '%B %d, %Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        self.logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def close(self):
        """
        Close all resources.
        """
        if self.session_manager:
            self.session_manager.close()
            self.logger.info("Closed session manager")
