"""
Polars extensions.
"""

import polars as pl


@pl.api.register_dataframe_namespace("ds")
@pl.api.register_lazyframe_namespace("ds")
class DatetimeMethods:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def add_datetime(self, ts_col: str = "ts") -> pl.DataFrame:
        """
        Add a datetime column to the DataFrame.
        """

        return self._df.with_columns(
            [pl.from_epoch(ts_col, time_unit="ns").alias("dts")]
        )