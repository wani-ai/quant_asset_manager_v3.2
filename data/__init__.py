# /data/__init__.py

"""
The data package is responsible for all data-related operations,
including connecting to external APIs via the SmartDataManager and
interacting with the local PostgreSQL database.

It abstracts the data sourcing and storage logic, providing a clean and
consistent interface for other parts of the application (e.g., analysis,
backtesting) to access financial and market data.

This __init__.py file exposes the main data access classes and functions,
acting as the public API for the data layer.
"""

# -----------------------------------------------------------------------------
# Import the main data fetching class from the connectors module.
# This makes the SmartDataManager directly accessible from the 'data' package.
# -----------------------------------------------------------------------------
from .connectors import SmartDataManager

# -----------------------------------------------------------------------------
# Import utility functions for database interaction from the database module.
# This provides a centralized way to handle database connections and queries.
# -----------------------------------------------------------------------------
from .database import get_database_engine, load_data_from_db

# -----------------------------------------------------------------------------
# Define the public API of the 'data' package using __all__.
# This explicitly lists the names that will be imported when a user
# executes 'from data import *'. It's a standard practice for creating
# clean and maintainable packages.
# -----------------------------------------------------------------------------
__all__ = [
    # The intelligent, multi-source API data fetcher
    'SmartDataManager',

    # Utility function to create a reusable database connection engine
    'get_database_engine',

    # Utility function to load data from the database into a pandas DataFrame
    'load_data_from_db',
]
