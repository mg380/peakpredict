"""
Event scraper for the Sports Data Platform.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from core.base_scraper import BaseScraper

class EventScraper(BaseScraper):
    """Scraper for event data."""
    
    def __init__(self, session_manager, max_retries=3, retry_delay=5):
        super().__init__(session_manager, max_retries, retry_delay)
    
    def discover_events(self):
        """
        Discover available events from the website by parsing the event dropdown.
        
        Returns:
            DataFrame with event names, IDs, and indoor/outdoor flag
        """
        event_database = pd.DataFrame(columns=['Event Name', 'Event ID', 'Indoor'])
        
        # Check both indoor (1) and outdoor (0) events
        for io in ["0", "1"]:
            # Start with a generic event ID (390 for running events)
            url = f"{self.session_manager.base_url}/db/alltfull.php?Ind={io}&Event=390&Sex=1&area=&All=0&Age=99"
            
            self.logger.info(f"Discovering {'indoor' if io == '1' else 'outdoor'} events from {url}")
            
            # Get the page content
            html = self.session_manager.get_page(url)
            soup = self._parse_html(html)
            
            # Find the event dropdown in the info-left div
            info_left_divs = soup.find_all('div', class_="info-left")
            
            if not info_left_divs:
                self.logger.warning("No info-left div found for event discovery")
                
                # Try to find the select element directly
                select_elements = soup.find_all('select', {'name': 'Event'})
                if select_elements:
                    self.logger.info(f"Found select element for events")
                    options = select_elements[0].find_all('option')
                    
                    for idx, option in enumerate(options):
                        # Skip the first option (usually a placeholder)
                        if idx == 0:
                            continue
                            
                        event_name = option.text.strip()
                        
                        try:
                            # Extract event ID from the option value
                            event_id = option['value'].split("&")[1].split('=')[1]
                        except (IndexError, KeyError):
                            self.logger.warning(f"Could not extract event ID for {event_name}")
                            continue
                            
                        # Check if this event is already in our database
                        if len(event_database[(event_database['Event ID'] == event_id) & 
                                            (event_database['Indoor'] == io)]) == 0:
                            # Add the event to our database
                            event_database.loc[len(event_database)] = [event_name, event_id, io]
                            self.logger.info(f"Discovered event: {event_name} - {event_id} - {'indoor' if io == '1' else 'outdoor'}")
                    
                    continue
                
                # For development/testing with mock data
                if "httpbin.org" in self.session_manager.base_url:
                    # Add some mock events for testing
                    mock_events = [
                        ["100m", "100", io],
                        ["200m", "200", io],
                        ["400m", "400", io],
                        ["800m", "800", io],
                        ["High Jump", "HJ", io],
                        ["Long Jump", "LJ", io]
                    ]
                    for event in mock_events:
                        event_database.loc[len(event_database)] = event
                    self.logger.info(f"Added {len(mock_events)} mock events for testing")
                
                continue
                
            # Find all options in the dropdown
            options = info_left_divs[0].find_all('option')
            
            for idx, option in enumerate(options):
                # Skip the first option (usually a placeholder)
                if idx == 0:
                    continue
                    
                event_name = option.text.strip()
                
                try:
                    # Extract event ID from the option value
                    event_id = option['value'].split("&")[1].split('=')[1]
                except (IndexError, KeyError):
                    self.logger.warning(f"Could not extract event ID for {event_name}")
                    continue
                    
                # Check if this event is already in our database
                if len(event_database[(event_database['Event ID'] == event_id) & 
                                    (event_database['Indoor'] == io)]) == 0:
                    # Add the event to our database
                    event_database.loc[len(event_database)] = [event_name, event_id, io]
                    self.logger.info(f"Discovered event: {event_name} - {event_id} - {'indoor' if io == '1' else 'outdoor'}")
        
        self.logger.info(f"Total events discovered: {len(event_database)}")
        return event_database
    
    def get_page(self, url, params=None):
        """
        Get page content using the session manager.
        
        Args:
            url: URL to get
            params: URL parameters
            
        Returns:
            Page HTML content
        """
        # Use the session manager to get the page
        html = self.session_manager.get_page(url, params)
        
        # Save the HTML to a file for debugging
        import os
        debug_dir = "debug_html"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create a filename based on the URL and params
        filename = url.split('/')[-1].split('.')[0]
        if params:
            for key, value in params.items():
                filename += f"_{key}_{value}"
        
        debug_path = os.path.join(debug_dir, f"{filename}.html")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        self.logger.info(f"Saved HTML to {debug_path} for debugging")
        
        return html
        
    def scrape(self, year=None, event_type=None, indoor=False, sex="1"):
        """
        Scrape events.
        
        Args:
            year: Optional year filter
            event_type: Optional event type filter (e.g., 390 for running events)
            indoor: Whether to scrape indoor events (True) or outdoor events (False)
            sex: Gender filter ("1" for men, "2" for women)
            
        Returns:
            List of event dictionaries
        """
        # Default to event_id 390 (running events) if not specified
        event_id = event_type or "390"
        
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
        
        # Add year filter if provided
        if year:
            params["Year"] = str(year)
        
        self.logger.info(f"Scraping {'indoor' if indoor else 'outdoor'} events for event_id={event_id}, sex={sex}")
        
        # Get the page content using our custom method that saves debug HTML
        html = self.get_page(url, params)
        events = self._parse_events(html, indoor, event_id)
        
        return events
    
    def _parse_events(self, html, indoor=False, event_id="390"):
        """
        Parse events from HTML.
        
        Args:
            html: HTML content
            indoor: Whether these are indoor events
            event_id: The event ID being scraped
            
        Returns:
            List of event dictionaries
        """
        soup = self._parse_html(html)
        events = []
        
        # Find the main event table
        event_table = soup.find("table", class_="maintable") or soup.find("table")
        
        if not event_table:
            self.logger.warning("No event table found in the HTML")
            # Save the HTML to a file for debugging
            with open(f"debug_html_{event_id}_{indoor}.html", "w", encoding="utf-8") as f:
                f.write(html)
            self.logger.info(f"Saved HTML to debug_html_{event_id}_{indoor}.html for debugging")
            return events
        
        # Process each row in the table
        for row in event_table.find_all("tr")[1:]:  # Skip header row
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            
            try:
                # Extract event data - for development mode, the mock data has a different structure
                # than what we'd expect from tilastopaja.info
                if "httpbin.org" in self.session_manager.base_url:
                    # For development/testing with mock data
                    event = {
                        "name": cells[0].text.strip(),
                        "date": "2023-08-01",  # Use a fixed date for testing
                        "location": "Test Location",
                        "event_type": "Indoor" if indoor else "Outdoor",
                        "discipline": f"Event {event_id}"
                    }
                else:
                    # For real tilastopaja.info data
                    # The structure might vary based on the event type
                    # We need to adapt based on the actual HTML structure
                    
                    # Try to extract competition name
                    competition_name = cells[0].text.strip() if cells[0].text.strip() else "Unknown"
                    
                    # Try to extract date
                    date_text = cells[1].text.strip() if len(cells) > 1 else ""
                    
                    # Try to extract location
                    location_text = cells[2].text.strip() if len(cells) > 2 else ""
                    
                    event = {
                        "name": competition_name,
                        "date": date_text,
                        "location": location_text,
                        "event_type": "Indoor" if indoor else "Outdoor",
                        "discipline": f"Event {event_id}"
                    }
                
                # Try to extract event ID from links
                event_link = cells[0].find('a')
                if event_link and 'href' in event_link.attrs:
                    href = event_link['href']
                    import re
                    id_match = re.search(r'id=(\d+)', href)
                    if id_match:
                        event["id"] = id_match.group(1)
                
                events.append(event)
            except Exception as e:
                self.logger.warning(f"Failed to parse event row: {str(e)}")
        
        self.logger.info(f"Parsed {len(events)} events for Event {event_id} ({'indoor' if indoor else 'outdoor'})")
        return events
    
    def scrape_all_events(self):
        """
        Scrape all athletics events for both indoor and outdoor, men and women.
        
        Returns:
            List of all events
        """
        all_events = []
        
        # First, discover all available events
        event_database = self.discover_events()
        
        # If no events were discovered, use hardcoded event types for fallback
        if len(event_database) == 0:
            self.logger.warning("No events discovered, using fallback event types")
            
            # Define fallback event types
            fallback_events = [
                ["100m", "100", "0"],
                ["200m", "200", "0"],
                ["400m", "400", "0"],
                ["800m", "800", "0"],
                ["1500m", "1500", "0"],
                ["5000m", "5000", "0"],
                ["10000m", "10000", "0"],
                ["110m Hurdles", "110H", "0"],
                ["400m Hurdles", "400H", "0"],
                ["3000m Steeplechase", "3000SC", "0"],
                ["High Jump", "HJ", "0"],
                ["Pole Vault", "PV", "0"],
                ["Long Jump", "LJ", "0"],
                ["Triple Jump", "TJ", "0"],
                ["Shot Put", "SP", "0"],
                ["Discus Throw", "DT", "0"],
                ["Hammer Throw", "HT", "0"],
                ["Javelin Throw", "JT", "0"],
                ["100m", "100", "1"],
                ["200m", "200", "1"],
                ["400m", "400", "1"],
                ["800m", "800", "1"],
                ["1500m", "1500", "1"],
                ["3000m", "3000", "1"],
                ["60m Hurdles", "60H", "1"],
                ["High Jump", "HJ", "1"],
                ["Pole Vault", "PV", "1"],
                ["Long Jump", "LJ", "1"],
                ["Triple Jump", "TJ", "1"],
                ["Shot Put", "SP", "1"]
            ]
            
            for event in fallback_events:
                event_database.loc[len(event_database)] = event
            
            self.logger.info(f"Added {len(fallback_events)} fallback events")
        
        self.logger.info(f"Scraping data for {len(event_database)} events")
        
        # Loop through all discovered events
        for index, row in event_database.iterrows():
            event_name = row['Event Name']
            event_id = row['Event ID']
            is_indoor = row['Indoor'] == "1"
            
            self.logger.info(f"Scraping events for {event_name} (ID: {event_id}, Indoor: {is_indoor})")
            
            try:
                # Scrape events for men
                men_events = self.scrape(indoor=is_indoor, sex="1", event_type=event_id)
                for event in men_events:
                    event["discipline"] = event_name
                    event["gender"] = "Men"
                all_events.extend(men_events)
                
                # Scrape events for women
                women_events = self.scrape(indoor=is_indoor, sex="2", event_type=event_id)
                for event in women_events:
                    event["discipline"] = event_name
                    event["gender"] = "Women"
                all_events.extend(women_events)
            except Exception as e:
                self.logger.error(f"Error scraping events for {event_name}: {str(e)}")
        
        self.logger.info(f"Total events scraped: {len(all_events)}")
        return all_events
