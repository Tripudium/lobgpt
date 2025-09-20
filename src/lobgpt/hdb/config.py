"""
Configuration settings for historical database (HDB) module.

This module contains configuration constants and API credentials for accessing
external data sources. It centralizes all configuration parameters used by
the HDB data loaders to ensure consistent setup across the application.

Key Components:
    - API Keys: Environment-based credential loading
    - Data Paths: Default storage locations for different data sources
    - Configuration Constants: Shared settings for data loading operations

Example:
    Accessing configuration in data loaders:

    >>> from triplob.hdb.config import TARDIS_API_KEY, TARDIS_DATA_PATH
    >>> if TARDIS_API_KEY:
    ...     # Initialize Tardis client with API key
    ...     client = TardisClient(api_key=TARDIS_API_KEY)

Environment Variables:
    TARDIS_API_KEY: Required for accessing Tardis historical market data

Notes:
    API keys should be set as environment variables and never committed to
    version control. Use .env files for local development with proper gitignore.

See Also:
    triplob.hdb.tardis_dataloader: Uses TARDIS_API_KEY for data access
    triplob.hdb.base: Uses common data path constants
"""

import os
from pathlib import Path

# API credentials - loaded from environment variables
TARDIS_API_KEY = os.getenv("TARDIS_API_KEY", "your-tardis-api-key-here")
TARDIS_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "tardis"
