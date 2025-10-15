"""
LOBSTER historical market data loader for US equity markets.

This module provides efficient loading and processing of historical order book
and message data from LOBSTER (Limit Order Book System: The Efficient Reconstructor).

LOBSTER provides high-quality limit order book data for US equity markets with
nanosecond precision and full market depth.

Example:
    Basic data loading:

    >>> from lobgpt.hdb.registry import get_dataset
    >>> lobster = get_dataset("lobster")
    >>> df = lobster.load_book("AAPL", dates=['20240101', '20240102'])

See Also:
    lobgpt.hdb.base.DataLoader: Base class for all data loaders
    lobgpt.hdb.registry: Registry for accessing different data sources
    https://lobsterdata.com/info/DataStructure.php: Data format specification
"""

import gzip
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import polars as pl

from lobgpt.hdb.base import DataLoader
from lobgpt.hdb.registry import register_dataset
from lobgpt.hdb.utils import get_days
from lobgpt.utils.time import nanoseconds

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOBSTER_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "lobster"

# LOBSTER event type constants
SUBMISSION = 1
DELETION = 2
PARTIAL_CANCEL = 3
VISIBLE_EXECUTION = 4
HIDDEN_EXECUTION = 5
CROSS_TRADE = 6
TRADING_HALT = 7


def generate_message_schema() -> dict[str, pl.DataType]:
    """Generate schema for LOBSTER message files."""
    return {
        "time": pl.Float64,        # Seconds after midnight with decimal precision
        "event_type": pl.Int8,     # Event type code (1-7)
        "order_id": pl.Int64,      # Unique order reference number
        "size": pl.Int32,          # Number of shares
        "price": pl.Int32,         # Price in cents (dollar * 10000)
        "direction": pl.Int8,      # -1 for sell, 1 for buy
    }


@register_dataset("lobster")
class LobsterData(DataLoader):
    """
    Dataloader for LOBSTER limit order book data.

    LOBSTER provides reconstructed limit order book data from NASDAQ with
    nanosecond precision timestamps and full market depth.
    """

    def __init__(self, root: str | Path = LOBSTER_DATA_PATH, cache: bool = True):
        logger.info("Initializing LobsterDataLoader with path %s", root)
        self.market = "lobster"
        if not isinstance(root, Path):
            root = Path(root)
        super().__init__(root, cache=cache)

    def _parse_message_file(self, filepath: Path, symbol: str, date: str) -> pl.DataFrame:
        """
        Parse a LOBSTER message file.

        Message file format:
        - Column 1: Time (seconds after midnight)
        - Column 2: Event type (1-7)
        - Column 3: Order ID
        - Column 4: Size (shares)
        - Column 5: Price (price * 10000)
        - Column 6: Direction (-1 sell, 1 buy)
        """
        logger.info(f"Parsing message file: {str(filepath)}")

        # Read CSV file directly with polars
        df = pl.read_csv(
            str(filepath),
            has_header=False,
            new_columns=["time", "event_type", "order_id", "size", "price", "direction"],
            schema={
                "time": pl.Float64,
                "event_type": pl.Int8,
                "order_id": pl.Int64,
                "size": pl.Int32,
                "price": pl.Int32,
                "direction": pl.Int8,
            }
        )

        session_start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        base_ns = int(session_start.timestamp() * 1_000_000_000)

        df = df.with_columns([
            pl.when(pl.col("event_type") == PARTIAL_CANCEL)
            .then(pl.lit(DELETION))
            .when(pl.col("event_type") == DELETION)
            .then(pl.lit(PARTIAL_CANCEL))
            .otherwise(pl.col("event_type"))
            .alias("event_code"),
            (pl.lit(base_ns) + pl.col("time").mul(1_000_000_000).round(0).cast(pl.Int64)).alias("tst"),
            # Convert price from cents representation to actual price
            (pl.col("price") / 10000.0).alias("prc"),
            pl.col("size").alias("vol"),
            pl.col("direction").replace_strict({1: True, -1: False}).alias("is_buy"),
        ]).select([
            "tst", "event_code", "is_buy", "prc", "vol", "direction", "order_id"
        ])
        return df

    def _parse_orderbook_file(
        self,
        filepath: Path,
        messages_df: pl.DataFrame,
        levels: int,
    ) -> pl.DataFrame:
        """
        Parse a LOBSTER order book snapshot file and align it with message timestamps.

        The order book file contains, for each event, the resulting book state with
        columns ordered as:
            ask_price_1, ask_volume_1, bid_price_1, bid_volume_1, ..., ask_price_n, ask_volume_n, bid_price_n, bid_volume_n
        """
        raw_columns: list[str] = []
        schema: dict[str, pl.DataType] = {}
        for level in range(levels):
            raw_columns.extend(
                [
                    f"ask_price_{level}",
                    f"ask_volume_{level}",
                    f"bid_price_{level}",
                    f"bid_volume_{level}",
                ]
            )
            schema[f"ask_price_{level}"] = pl.Int64
            schema[f"ask_volume_{level}"] = pl.Int32
            schema[f"bid_price_{level}"] = pl.Int64
            schema[f"bid_volume_{level}"] = pl.Int32

        orderbook_df = pl.read_csv(
            str(filepath),
            has_header=False,
            new_columns=raw_columns,
            schema=schema,
        ).with_row_count("row_idx")

        messages_with_idx = messages_df.with_row_count("row_idx").select(["row_idx", "tst"])
        orderbook_df = orderbook_df.join(messages_with_idx, on="row_idx", how="inner")

        # Convert raw columns into structured ask/bid columns and remove dummy values
        dummy_value = 9_999_999_999
        bid_dummy = -9_999_999_999
        transformed_cols = [pl.col("tst")]
        for level in range(levels):
            ask_price_col = pl.col(f"ask_price_{level}")
            bid_price_col = pl.col(f"bid_price_{level}")
            transformed_cols.extend(
                [
                    (
                        pl.when(ask_price_col == dummy_value)
                        .then(None)
                        .otherwise(ask_price_col.cast(pl.Float64) / 10000.0)
                        .cast(pl.Float64)
                        .alias(f"asks[{level}].price")
                    ),
                    pl.col(f"ask_volume_{level}").cast(pl.Float64).alias(f"asks[{level}].amount"),
                    (
                        pl.when(bid_price_col == bid_dummy)
                        .then(None)
                        .otherwise(bid_price_col.cast(pl.Float64) / 10000.0)
                        .cast(pl.Float64)
                        .alias(f"bids[{level}].price")
                    ),
                    pl.col(f"bid_volume_{level}").cast(pl.Float64).alias(f"bids[{level}].amount"),
                ]
            )

        orderbook_df = orderbook_df.select(transformed_cols)
        return orderbook_df

    def process(self, symbol: str, date: str, levels: int = 50) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """
        Process raw LOBSTER files for a given symbol and date.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            date: Date in YYYY-MM-DD format
            levels: Number of price levels to process

        Returns:
            Tuple of (messages_df, orderbook_df). The order book dataframe may be
            None if the corresponding snapshot file is unavailable.
        """
        # Check for raw files
        raw_message_file = self._raw_path / f"{symbol}_{date}_message_{levels}.csv"
        raw_orderbook_file = self._raw_path / f"{symbol}_{date}_orderbook_{levels}.csv"

        # Also check for gzipped versions
        if not raw_message_file.exists():
            raw_message_gz = self._raw_path / f"{symbol}_{date}_message_{levels}.csv.gz"
            if raw_message_gz.exists():
                logger.info(f"Decompressing {str(raw_message_gz)}")
                with gzip.open(str(raw_message_gz), 'rb') as f_in:
                    with open(str(raw_message_file), 'wb') as f_out:
                        f_out.write(f_in.read())

        if not raw_orderbook_file.exists():
            raw_orderbook_gz = self._raw_path / f"{symbol}_{date}_orderbook_{levels}.csv.gz"
            if raw_orderbook_gz.exists():
                logger.info(f"Decompressing {str(raw_orderbook_gz)}")
                with gzip.open(str(raw_orderbook_gz), 'rb') as f_in:
                    with open(str(raw_orderbook_file), 'wb') as f_out:
                        f_out.write(f_in.read())

        if not raw_message_file.exists():
            raise FileNotFoundError(
                f"Raw files not found for {symbol} on {date}. "
                f"Expected: {str(raw_message_file)}"
            )

        
        messages_df = self._parse_message_file(raw_message_file, symbol, date)
        processed_message_file = self._processed_path / f"{symbol}_{date}_messages.parquet"

        messages_df.write_parquet(str(processed_message_file))
        logger.info(f"Processed and saved {symbol} data for {date}")

        orderbook_df = None
        if raw_orderbook_file.exists():
            try:
                orderbook_df = self._parse_orderbook_file(raw_orderbook_file, messages_df, levels)
                processed_orderbook_file = self._processed_path / f"{symbol}_{date}_orderbook.parquet"
                orderbook_df.write_parquet(str(processed_orderbook_file))
            except Exception as exc:
                logger.warning(f"Failed to process orderbook snapshots for {symbol} on {date}: {exc}")
        else:
            logger.info(f"No orderbook snapshot file found for {symbol} on {date}")

        return messages_df, orderbook_df

    def _load_data(
        self,
        symbol: str,
        times: List[str],
        data_type: str = "messages",
        lazy: bool = False
    ) -> pl.DataFrame:
        """
        Load processed data for given symbol and times.

        Args:
            symbol: Stock ticker symbol
            times: List of times in YYMMDD.HHMM format
            data_type: 'messages' or 'orderbook'
            lazy: If True, return lazy dataframe
        """
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
            filename = f"{self.processed_path}/{symbol}_{day}_{data_type}.parquet"
            df = None

            if not Path(filename).exists():
                logger.info(f"Processed file not found, attempting to process raw data for {day}")
                try:
                    self.process(symbol, day)
                    if not Path(filename).exists():
                        logger.warning(f"Processing completed but {filename} still missing")
                        continue
                    if lazy:
                        df = pl.scan_parquet(filename)
                    else:
                        df = pl.read_parquet(filename)
                        if self.cache is not None:
                            self.cache[str(filename)] = df
                except FileNotFoundError as e:
                    logger.error(str(e))
                    continue
            else:
                if self.cache is not None and str(filename) in self.cache and not lazy:
                    df = self.cache[str(filename)]
                else:
                    if lazy:
                        df = pl.scan_parquet(filename)
                    else:
                        df = pl.read_parquet(filename)
                        if self.cache is not None:
                            self.cache[str(filename)] = df
            if df is not None:
                dfs.append(df)

        if not dfs:
            logger.warning(f"No data found for {symbol} on dates {days}")
            return None

        # Concatenate all dataframes
        if lazy:
            df = pl.concat(dfs, how="vertical_relaxed")
        else:
            df = pl.concat(dfs)
        df = df.filter(
            pl.col("tst").is_between(nanoseconds(times[0]), nanoseconds(times[1]))
        )
        return df

    def load_book(
        self,
        symbol: str,
        dates: List[str],
        depth: int = 10,
        lazy: bool = False,
        type: str = "incremental_book_L3"
    ) -> pl.DataFrame:
        """
        Load orderbook data for a given symbol and dates.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            dates: List of dates in YYYYMMDD format
            depth: Number of price levels to include (max available in file)
            lazy: If True, return lazy dataframe
            type: Data type - "snapshot" for book states, "incremental_book_L3" for messages

        Returns:
            DataFrame with orderbook snapshots or incremental messages
        """
        if type == "incremental_book_L3":
            # Return message data when incremental_book_L3 is requested
            df = self._load_data(symbol, dates, "messages", lazy)
        elif type == "snapshot":
            df = self._load_data(symbol, dates, "orderbook", lazy)
            if df is not None:
                columns = ["tst"]
                for level in range(depth):
                    columns.extend(
                        [
                            f"asks[{level}].price",
                            f"asks[{level}].amount",
                            f"bids[{level}].price",
                            f"bids[{level}].amount",
                        ]
                    )
                if lazy:
                    schema = df.schema
                    select_expr = [pl.col(col) for col in columns if col in schema]
                    if select_expr:
                        df = df.select(select_expr)
                else:
                    existing_columns = [col for col in columns if col in df.columns]
                    if existing_columns:
                        df = df.select(existing_columns)
        else:
            raise ValueError(f"Unknown type: {type}. Supported types: 'snapshot', 'incremental_book_L3'")
        return df

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols in the data directory."""
        symbols = set()

        # Check raw directory
        for file in self.raw_path.glob("*_message_*.csv*"):
            symbol = file.name.split("_")[0]
            symbols.add(symbol)

        # Check processed directory
        for file in self.processed_path.glob("*_messages.parquet"):
            symbol = file.name.split("_")[0]
            symbols.add(symbol)

        return sorted(list(symbols))

    def get_available_dates(self, symbol: str) -> List[str]:
        """Get list of available dates for a given symbol."""
        dates = set()

        # Check raw directory
        for file in self.raw_path.glob(f"{symbol}_*_message_*.csv*"):
            parts = file.name.split("_")
            if len(parts) >= 2:
                dates.add(parts[1])

        # Check processed directory
        for file in self.processed_path.glob(f"{symbol}_*_messages.parquet"):
            parts = file.name.split("_")
            if len(parts) >= 2:
                dates.add(parts[1])

        return sorted(list(dates))

    def download(self, symbol: str, date: str, levels: int = 10):
        """
        Download LOBSTER data for a given symbol and date.

        Note: LOBSTER is a commercial data provider. This method is a placeholder
        for users who have their own LOBSTER data access.
        """
        raise NotImplementedError(
            "LOBSTER data must be obtained directly from https://lobsterdata.com/. "
            "Place downloaded files in the raw directory with naming convention: "
            f"{symbol}_{date}_message_{levels}.csv and {symbol}_{date}_orderbook_{levels}.csv"
        )
