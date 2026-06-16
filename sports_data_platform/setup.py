"""
Setup script for the Sports Data Platform.
"""

from setuptools import setup, find_packages

setup(
    name="sports_data_platform",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "beautifulsoup4>=4.9.0",
        "h5py>=3.1.0",
        "pandas>=1.3.0",
        "python-dotenv>=0.15.0",
        "requests>=2.25.0",
        "selenium>=4.0.0",
        "sqlalchemy>=1.4.0",
        "tables>=3.6.0",  # PyTables for HDF5 support with pandas
    ],
    entry_points={
        "console_scripts": [
            "sports-data=main:main_cli",
        ],
    },
    python_requires=">=3.8",
    author="Sports Data Team",
    description="A modular Python framework for scraping, processing, and analyzing sports data",
    keywords="sports, data, scraping, analytics",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
