"""
Database models for the Sports Data Platform.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Event(Base):
    """Event database model."""
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    date = Column(String(50), nullable=False)  # Changed from DateTime to String for flexibility
    location = Column(String(255))
    event_type = Column(String(50))  # Indoor/Outdoor
    discipline = Column(String(50))  # 100m, High Jump, etc.
    gender = Column(String(10))  # Men/Women
    
    performances = relationship("Performance", back_populates="event")

class Athlete(Base):
    """Athlete database model."""
    __tablename__ = 'athletes'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    country = Column(String(50))
    birth_date = Column(String(50))  # Changed from DateTime to String for flexibility
    gender = Column(String(10))
    
    performances = relationship("Performance", back_populates="athlete")

class Performance(Base):
    """Performance database model."""
    __tablename__ = 'performances'
    
    id = Column(Integer, primary_key=True)
    athlete_id = Column(Integer, ForeignKey('athletes.id'), nullable=False)
    event_id = Column(Integer, ForeignKey('events.id'), nullable=False)
    result = Column(String(50), nullable=False)
    position = Column(String(20))  # Changed from Integer to String for flexibility
    points = Column(Float)
    wind = Column(String(10))  # Added for sprint events
    
    athlete = relationship("Athlete", back_populates="performances")
    event = relationship("Event", back_populates="performances")
