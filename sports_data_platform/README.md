# Sports Data Platform

A modular Python framework for scraping, processing, and analyzing sports data.

## Features

- Robust error handling with automatic retries
- Parallel processing for improved performance
- Modular architecture for easy extension
- Database storage for efficient data management
- Command-line interface for easy operation
- HDF5 storage for performance data

## Architecture

The platform is organized into the following components:

- **Core**: Base classes and session management
- **Scrapers**: Data collection from web sources
- **Processors**: Data cleaning and validation
- **Storage**: Database models and operations
- **Utils**: Utility functions for error handling, logging, and concurrency
- **Pipeline**: Orchestration of the data processing workflow
- **CLI**: Command-line interface

## Installation

### Prerequisites

- Python 3.8 or higher
- Chrome browser (for Selenium-based scraping)
- ChromeDriver (for Selenium)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sports_data_platform.git
   cd sports_data_platform
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package:
   ```
   pip install -e .
   ```

4. Create a `.env` file with your credentials:
   ```
   SPORTS_DATA_USER=your_username
   SPORTS_DATA_PASS=your_password
   SPORTS_DATA_DB=sqlite:///sports_data.db
   ```

## Usage

### Command Line Interface

Run the platform with various options:

```bash
# Scrape events for a specific year
python main.py --events --year 2023

# Scrape athletes and their performances
python main.py --athletes --performances

# Use a specific configuration profile
python main.py --config production --events --athletes

# Set logging level
python main.py --log-level DEBUG
```

### As a Library

```python
from pipeline.orchestrator import PipelineOrchestrator
from config.settings import get_config

# Load configuration
config = get_config("default")

# Create and set up orchestrator
orchestrator = PipelineOrchestrator(config)
orchestrator.setup()

# Run pipelines
events = orchestrator.run_event_pipeline(year=2023)
athletes = orchestrator.run_athlete_pipeline()
performances = orchestrator.run_performance_pipeline()

# Clean up
orchestrator.close()
```

## Data Storage

The platform supports two storage methods:

1. **SQL Database**: Using SQLAlchemy ORM with models for events, athletes, and performances
2. **HDF5 Files**: Hierarchical data format for efficient storage of performance data

## Configuration

Configuration profiles are defined in `config/settings.py`:

- **default**: Standard configuration
- **development**: For local development (non-headless browser, etc.)
- **testing**: For running tests
- **production**: Optimized for production use

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
