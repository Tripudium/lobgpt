"""
Utilities for converting raw LOB loader outputs into tokenised training data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import polars as pl

from lobgpt.hdb.base import DataLoader
from lobgpt.preprocessing import (
    prepare_message_volume_features,
    preprocess_messages_for_tokenization,
)
from lobgpt.tokenizer import ConfigurableTokenizer, DEFAULT_TOKENIZER_CONFIG_PATH


def prepare_tokenised_sequence(
    loader: DataLoader,
    product: str,
    times: Sequence[str],
    *,
    depth: int = 10,
    tick_size: float = 0.01,
    tokenizer_config: Optional[Path] = None,
    include_reference_info: bool = False,
) -> dict[str, object]:
    """Load messages/snapshots and convert to tokenised training sequence."""

    messages = loader.load_book(product, list(times), type="incremental_book_L3")
    if messages is None or messages.is_empty():
        raise ValueError("No messages returned by loader.")

    snapshots = loader.load_book(product, list(times), depth=depth, type="snapshot")
    if snapshots is None or snapshots.is_empty():
        raise ValueError("No snapshots returned by loader.")

    messages_features = preprocess_messages_for_tokenization(
        messages,
        tick_size=tick_size,
        include_reference_info=include_reference_info,
    )

    feature_table = prepare_message_volume_features(
        messages_features,
        snapshots,
        tick_size=tick_size,
        depth=depth,
    )

    config_path = tokenizer_config or DEFAULT_TOKENIZER_CONFIG_PATH
    tokenizer = ConfigurableTokenizer.from_config_file(config_path)

    token_ids = tokenizer.encode_dataframe(messages_features)

    record = {
        "config_path": str(config_path),
        "token_ids": token_ids.tolist(),
        "features": feature_table.to_dicts(),
        "columns": feature_table.columns,
    }
    return record


def save_tokenised_sequence(record: dict[str, object], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        token_ids=np.asarray(record["token_ids"], dtype=np.int64),
        features=json.dumps(record["features"]).encode("utf-8"),
        columns=np.asarray(record["columns"], dtype="U"),
        config_path=np.asarray(record["config_path"], dtype="U"),
    )


def load_tokenised_sequence(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    features = pl.from_dicts(json.loads(data["features"].tobytes().decode("utf-8")))
    return {
        "token_ids": data["token_ids"],
        "features": features,
        "columns": list(data["columns"]),
        "config_path": Path(str(data["config_path"][()])),
    }
