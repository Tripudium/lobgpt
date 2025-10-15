"""
MIC historical market data loader for cryptocurrency exchanges.

This module provides efficient loading and streaming of historical order book
and trade data from MIC.

Example:
    Basic data loading:

    >>> from lobgpt.hdb.registry import get_dataset
    >>> mic = get_dataset("mic")
    >>> df = mic.load_book("RTSX", times=['250120.000000', '250120.235959'])

See Also:
    lobgpt.hdb.base.DataLoader: Base class for all data loaders
    lobgpt.hdb.registry: Registry for accessing different data sources
"""

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

# Local imports
from lobgpt.hdb.base import DataLoader
from lobgpt.hdb.registry import register_dataset
from lobgpt.hdb.utils import get_days
from lobgpt.utils.time import nanoseconds

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MIC_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "mic"

SUBMISSION = 1
DELETION = 2


def generate_schema(type: str = "incremental_book_L3") -> dict[str, pl.DataType]:
    """
    Generate a schema for a given type of data from tardis
    """
    schema = {}
    if type == "incremental_book_L3":
        schema = {
            "tst": pl.Int64,
            "tsr": pl.Int64,
            "delivery_delay_ms": pl.Int64,
            "event_code": pl.Int64,
            "is_buy": pl.Boolean,
            "prc": pl.Float64,
            "prc_diff": pl.Float64,
            "vol": pl.Float64,
            "order_id": pl.Int64,
            "order_id_offset": pl.Int64
        }
    else:
        raise ValueError(f"Unknown type: {type}")
    return schema


@register_dataset("mic")
class MicData(DataLoader):
    """
    Dataloader for MIC data
    """

    def __init__(self, root: str | Path = MIC_DATA_PATH):
        logger.info("Initializing MicDataLoader with path %s", root)
        self.market = "mic"
        if not isinstance(root, Path):
            root = Path(root)
        super().__init__(root)

    def _load_data(
        self, product: str, times: list[str], type: str = "incremental_book_L3", lazy=False
    ) -> pl.DataFrame:
        """
        Load data for a given product and times.
        """
        assert type in ["incremental_book_L3"]

        if len(times) != 2:
            raise ValueError(
                "Times must be a list of two strings in the format '%y%m%d.%H%M'"
            )
        try:
            dtimes = [datetime.strptime(t, "%y%m%d.%H%M%S") for t in times]
        except ValueError:
            raise ValueError("Times must be in the format '%y%m%d.%H%M%S'")
        days = get_days(dtimes[0], dtimes[1])

        dfs = []
        for day in days:
            filename = f"{str(self.processed_path)}/{self.market}_{day}_{product}.parquet"
            if not Path(filename).exists():
                logger.info(f"File {filename} not found")
                return None
            else:
                # check if the dataframe is already in the cache
                if self.cache is not None and filename in self.cache and not lazy:
                    df = self.cache[filename]
                else:
                    if lazy:
                        df = pl.scan_parquet(filename)
                    else:
                        df = pl.read_parquet(filename)
                        if self.cache is not None:
                            self.cache[filename] = df
            dfs.append(df)

        df = pl.concat(dfs)
        df = df.filter(
            pl.col("tst").is_between(nanoseconds(times[0]), nanoseconds(times[1]))
        )
        return df

    def load_book(
        self,
        product: str,
        times: list[str],
        depth: int = 10,
        lazy: bool = False,
        type: str = "incremental_book_L3",
    ) -> pl.DataFrame:
        """
        Load book data for a given product and times.
        """
        df = self._load_data(product, times, type, lazy)
        if type == "incremental_book_L3":
            df = df.with_columns(
                (pl.col("tsr") - pl.col("tst")).alias("delivery_delay_ms")//1_000_000,
                (pl.col("order_id") - df[0, "order_id"] + 1).alias("order_id_offset"),
                pl.col("prc").diff().alias("prc_diff").fill_null(0))
            columns = [
                "tst", 
                "tsr", 
                "delivery_delay_ms", 
                "event_code", 
                "is_buy", 
                "prc", 
                "prc_diff", 
                "vol", 
                "order_id", 
                "order_id_offset"]
            df = df.select(columns)
        else:
            raise ValueError(f"Invalid type: {type}")
        return df
    
    def load_trades(
        self,
        products: list[str] | str,
        times: list[str],
        lazy=False,
    ) -> pl.DataFrame:
        """
        Load trade data for a given product and times.
        """
        raise NotImplementedError("Load trades is not implemented for MIC")

    def load_inc(
        self,
        product: str,
        times: list[str],
        lazy: bool = False,
    ) -> pl.DataFrame:
        raise NotImplementedError("Load incremental book is not implemented for MIC")

    def download(self, product: str, day: str, type: str):
        """
        Download data for a given product and day.
        """
        raise NotImplementedError("Download is not implemented for MIC")

    def process(self, product: str, day: str, type: str):
        """
        Process data for a given product and day.
        """
        raise NotImplementedError("Process is not implemented for MIC")
