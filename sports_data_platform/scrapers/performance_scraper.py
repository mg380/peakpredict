"""
Performance scraper for the Sports Data Platform.
"""

from typing import List, Dict, Any, Optional
from core.base_scraper import BaseScraper

class PerformanceScraper(BaseScraper):
    """Scraper for athlete performance data."""
    
    def __init__(self, session_manager, max_retries=3, retry_delay=5):
        super().__init__(session_manager, max_retries, retry_delay)
    
    def scrape(self, athlete_id=None, event_type=None, year=None):
        """
        Scrape performance data.
        
        Args:
            athlete_id: Optional athlete ID to filter performances
            event_type: Optional event type filter
            year: Optional year to filter performances
            
        Returns:
            List of performance dictionaries
        """
        if athlete_id:
            return self.scrape_athlete_performances(athlete_id, event_type, year)
        else:
            return []
    
    def scrape_athlete_performances(self, athlete_id: str, event_type: Optional[str] = None, 
                                   year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scrape performances for a specific athlete.
        
        Args:
            athlete_id: Athlete ID
            event_type: Optional event type filter
            year: Optional year filter
            
        Returns:
            List of performance dictionaries
        """
        # Construct the URL for tilastopaja.info
        url = f"{self.session_manager.base_url}/db/athlpages.php"
        params = {"athleteid": athlete_id}
        
        if event_type:
            params["event"] = event_type
        if year:
            params["year"] = year
        
        # Use Selenium for JavaScript-heavy pages
        html = self.session_manager.get_page(url, params)
        return self._parse_performances(html, athlete_id)
    
    def _parse_performances(self, html: str, athlete_id: str) -> List[Dict[str, Any]]:
        """
        Parse performances from HTML.
        
        Args:
            html: HTML content
            athlete_id: Athlete ID
            
        Returns:
            List of performance dictionaries
        """
        soup = self._parse_html(html)
        performances = []
        
        # Get athlete name
        athlete_name_elem = soup.find("h1") or soup.find("title")
        athlete_name = athlete_name_elem.text.strip() if athlete_name_elem else "Unknown"
        
        # Find performance tables - typically organized by event type
        performance_tables = soup.find_all("table", class_="maintable") or soup.find_all("table")
        
        for table in performance_tables:
            # Get event type from table header or nearby element
            event_type_elem = table.find_previous("h2") or table.find_previous("h3") or table.find_previous("b")
            event_type = event_type_elem.text.strip() if event_type_elem else "Unknown"
            
            # Process each row in the table
            for row in table.find_all("tr")[1:]:  # Skip header row
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue
                
                # Extract performance data
                try:
                    performance = {
                        "athlete_id": athlete_id,
                        "athlete_name": athlete_name,
                        "event_type": event_type,
                    }
                    
                    # Extract data based on table structure
                    # The structure might vary, so we need to be flexible
                    if len(cells) >= 1:
                        performance["date"] = cells[0].text.strip()
                    
                    if len(cells) >= 2:
                        performance["location"] = cells[1].text.strip()
                    
                    if len(cells) >= 3:
                        performance["result"] = cells[2].text.strip()
                    
                    # Add position if available
                    if len(cells) >= 4:
                        position_text = cells[3].text.strip()
                        try:
                            performance["position"] = int(position_text)
                        except (ValueError, TypeError):
                            performance["position"] = position_text
                    
                    # Add event name if available
                    if len(cells) >= 5:
                        performance["event_name"] = cells[4].text.strip()
                    
                    # Add wind if available (for sprint events)
                    if len(cells) >= 6:
                        wind_text = cells[5].text.strip()
                        if wind_text:
                            performance["wind"] = wind_text
                    
                    performances.append(performance)
                except Exception as e:
                    self.logger.warning(f"Failed to parse performance row: {str(e)}")
        
        self.logger.info(f"Parsed {len(performances)} performances for athlete {athlete_id}")
        return performances
    
    def get_athlete_events(self, athlete_id: str) -> List[str]:
        """
        Get list of events for an athlete.
        
        Args:
            athlete_id: Athlete ID
            
        Returns:
            List of event types
        """
        url = f"{self.session_manager.base_url}/db/athlpages.php"
        params = {"athleteid": athlete_id}
        
        html = self.session_manager.get_page(url, params)
        soup = self._parse_html(html)
        
        # Find event headers
        event_elements = soup.find_all(["h2", "h3"]) or soup.find_all("b")
        events = []
        
        for element in event_elements:
            event_name = element.text.strip()
            if event_name and not event_name.startswith("Personal"):  # Filter out non-event headers
                events.append(event_name)
        
        self.logger.info(f"Found {len(events)} events for athlete {athlete_id}")
        return events
