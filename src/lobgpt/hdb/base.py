"""
Abstract base class for historical data loaders in Tripudium LOB.

This module defines the DataLoader abstract base class that provides a consistent
interface for loading and streaming financial market data from various sources.
All data loader implementations inherit from this class to ensure uniform API.

Key Features:
    - Abstract interface for data loading operations
    - Support for both order book and trade data
    - Streaming capabilities for real-time processing
    - Configurable batch sizes for memory-efficient processing
    - Time-based data filtering and selection

Example:
    Implementing a custom data loader:

    >>> class CustomLoader(DataLoader):
    ...     def load_book(self, symbol, times, depth=5):
    ...         # Implementation details
    ...         return polars_dataframe

    Using a data loader:

    >>> loader = get_dataset("custom")
    >>> df = loader.load_book("BTCUSDT", times=['250120.000000', '250120.235959'])

Notes:
    All concrete implementations must implement the abstract methods defined
    in this class. The streaming methods provide efficient memory usage for
    large datasets by processing data in configurable batches.

See Also:
    triplob.hdb.tardis_dataloader.TardisData: Tardis-specific implementation
    triplob.hdb.registry: Registry system for accessing data loaders
"""

import logging
from pathlib import Path
from typing import Generator

import polars as pl

# Local imports

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"


class DataLoader:
    """
    Base class for loading and processing financial data.

    This class provides common functionality for downloading, processing,
    and accessing financial data from various sources.
    """

    def __init__(self, root: str | Path = DATA_PATH, cache: bool = True):
        """
        Initialize the DataLoader with a path to the data.
        """
        logger.info("Initializing DataLoader with path %s" % root)
        self.root = root
        self._raw_path = Path(root) / "raw"
        self._processed_path = Path(root) / "processed"
        # Create the directories if they do not exist
        self._raw_path.mkdir(parents=True, exist_ok=True)
        self._processed_path.mkdir(parents=True, exist_ok=True)
        # maintain a cache of dataframes
        if cache:
            self.cache = {}
        else:
            self.cache = None

    @property
    def raw_path(self):
        """
        Get the path to raw data files.

        Returns:
            Path: The directory containing raw data files.
        """
        return str(self._raw_path)

    @property
    def processed_path(self):
        """
        Get the path to processed data files.

        Returns:
            Path: The directory containing processed data files.
        """
        return str(self._processed_path)

    @raw_path.setter
    def raw_path(self, path: str | Path):
        """
        Set the path to raw data files.

        Args:
            path (str | Path): The directory to store raw data files.
        """
        self._raw_path = path

    @processed_path.setter
    def processed_path(self, path: str | Path):
        """
        Set the path to processed data files.

        Args:
            path (str | Path): The directory to store processed data files.
        """
        self._processed_path = path

    def load_trades(
        self, products: list[str] | str, times: list[str], lazy=False
    ) -> pl.DataFrame:
        """
        Load trades data for a given product and times.
        """
        if isinstance(products, str):
            products = [products]
        dfs = []
        for product in products:
            df = self._load_data(product, times, "trade", lazy)
            df = df.with_columns(pl.lit(product).alias("product"))
            dfs.append(df)
        df = pl.concat(dfs).sort("ts")
        return df

    def load_book(
        self, _product: str, _times: list[str], _depth: int = 10, _lazy: bool = False
    ) -> pl.DataFrame:
        """
        Load book data for a given product and times.
        """
        raise NotImplementedError

    def load_sync(
        self,
        _products: list[str] | str,
        _times: list[str],
        _col: str = "mid",
        _freq: str = "1s",
        _lazy=False,
    ) -> pl.DataFrame:
        """
        Load data for a given set of products and times, sampled at fixed frequency.
        """
        raise NotImplementedError

    def download(self, _product: str, _month: str, _type: str):
        raise NotImplementedError

    def process(self, _product: str, _month: str, _type: str):
        raise NotImplementedError

    def stream_book(
        self,
        _product: str,
        _times: list[str],
        _depth: int = 10,
        _batch_size: int = 10000,
    ) -> Generator[pl.DataFrame, None, None]:
        """
        Stream book data for a given product and times in batches.

        Args:
            _product: Product symbol
            _times: List of two strings in format '%y%m%d.%H%M%S'
            _depth: Order book depth
            _batch_size: Number of rows per batch

        Yields:
            Polars DataFrame batches
        """
        raise NotImplementedError

    def stream_trades(
        self, _product: str, _times: list[str], _batch_size: int = 10000
    ) -> Generator[pl.DataFrame, None, None]:
        """
        Stream trade data for a given product and times in batches.

        Args:
            _product: Product symbol
            _times: List of two strings in format '%y%m%d.%H%M%S'
            _batch_size: Number of rows per batch

        Yields:
            Polars DataFrame batches
        """
        raise NotImplementedError
