"""
Athlete scraper for the Sports Data Platform.
"""

from typing import List, Dict, Any, Optional
from core.base_scraper import BaseScraper

class AthleteScraper(BaseScraper):
    """Scraper for athlete data."""
    
    def __init__(self, session_manager, max_retries=3, retry_delay=5):
        super().__init__(session_manager, max_retries, retry_delay)
    
    def scrape(self, event_id=None, indoor=False, sex="1", *args, **kwargs):
        """
        Main scraping method implementation required by BaseScraper.
        
        Args:
            event_id: Event ID to filter athletes (e.g., 390 for running events)
            indoor: Whether to scrape indoor events (True) or outdoor events (False)
            sex: Gender filter ("1" for men, "2" for women)
            
        Returns:
            List of athlete dictionaries
        """
        return self.get_athlete_list(event_id, indoor, sex)
    
    def get_athlete_list(self, event_id=None, indoor=False, sex="1"):
        """
        Get list of athletes to scrape.
        
        Args:
            event_id: Event ID to filter athletes (e.g., 390 for running events)
            indoor: Whether to scrape indoor events (True) or outdoor events (False)
            sex: Gender filter ("1" for men, "2" for women)
            
        Returns:
            List of athlete dictionaries
        """
        # Default to event_id 390 (running events) if not specified
        event_id = event_id or "390"
        
        # Set indoor/outdoor parameter
        ind_param = "1" if indoor else "0"
        
        # Construct the URL for tilastopaja.info
        url = f"{self.session_manager.base_url}/db/alltfull.php"
        params = {
            "Ind": ind_param,
            "Event": event_id,
            "Sex": sex,
            "area": "",
            "All": "0",
            "Age": "99"
        }
        
        self.logger.info(f"Scraping athletes for {'indoor' if indoor else 'outdoor'} event_id={event_id}, sex={sex}")
        
        # Get the page content
        response = self._make_request(url, params=params)
        return self._parse_athlete_list(response.text)
    
    def _parse_athlete_list(self, html):
        """
        Parse athlete list from HTML.
        
        Args:
            html: HTML content
            
        Returns:
            List of athlete dictionaries
        """
        soup = self._parse_html(html)
        athletes = []
        
        # Find the main athlete table
        athlete_table = soup.find("table", class_="maintable") or soup.find("table")
        
        if not athlete_table:
            self.logger.warning("No athlete table found in the HTML")
            return athletes
        
        # Process each row in the table
        for row in athlete_table.find_all("tr")[1:]:  # Skip header row
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            
            try:
                # Extract athlete data
                athlete = {"name": cells[0].text.strip()}
                
                # Try to extract athlete ID from links
                athlete_link = cells[0].find('a')
                if athlete_link and 'href' in athlete_link.attrs:
                    href = athlete_link['href']
                    import re
                    id_match = re.search(r'id=(\d+)', href)
                    if id_match:
                        athlete["id"] = id_match.group(1)
                
                # Try to extract country if available
                if len(cells) > 1:
                    country_cell = cells[1]
                    country = country_cell.text.strip()
                    if country:
                        athlete["country"] = country
                
                # Try to extract birth date if available
                if len(cells) > 2:
                    birth_date = cells[2].text.strip()
                    if birth_date:
                        athlete["birth_date"] = birth_date
                
                # Add gender based on the page we're scraping
                # This would need to be passed in from the calling function
                # For now, we'll try to infer from the URL or table structure
                
                athletes.append(athlete)
            except Exception as e:
                self.logger.warning(f"Failed to parse athlete row: {str(e)}")
        
        self.logger.info(f"Parsed {len(athletes)} athletes")
        return athletes
    
    def scrape_all_athletes(self, event_ids=None):
        """
        Scrape athletes for multiple events, both indoor and outdoor, men and women.
        
        Args:
            event_ids: List of event IDs to scrape (defaults to [390] for running events)
            
        Returns:
            List of all athletes
        """
        all_athletes = []
        event_ids = event_ids or ["390"]  # Default to running events
        
        for event_id in event_ids:
            # Scrape outdoor men
            outdoor_men = self.get_athlete_list(event_id=event_id, indoor=False, sex="1")
            all_athletes.extend(outdoor_men)
            
            # Scrape indoor men
            indoor_men = self.get_athlete_list(event_id=event_id, indoor=True, sex="1")
            all_athletes.extend(indoor_men)
            
            # Scrape outdoor women
            outdoor_women = self.get_athlete_list(event_id=event_id, indoor=False, sex="2")
            all_athletes.extend(outdoor_women)
            
            # Scrape indoor women
            indoor_women = self.get_athlete_list(event_id=event_id, indoor=True, sex="2")
            all_athletes.extend(indoor_women)
        
        # Remove duplicates based on athlete ID
        unique_athletes = []
        seen_ids = set()
        
        for athlete in all_athletes:
            athlete_id = athlete.get("id")
            if athlete_id and athlete_id not in seen_ids:
                seen_ids.add(athlete_id)
                unique_athletes.append(athlete)
            elif not athlete_id:
                # If no ID, use name as fallback for deduplication
                name = athlete.get("name")
                if name and name not in seen_ids:
                    seen_ids.add(name)
                    unique_athletes.append(athlete)
        
        return unique_athletes
