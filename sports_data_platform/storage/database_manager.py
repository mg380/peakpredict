"""
Database manager for the Sports Data Platform.
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .database import Base

class DatabaseManager:
    """Database manager for the Sports Data Platform."""
    
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create database tables."""
        Base.metadata.create_all(self.engine)
