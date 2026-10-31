"""Append model evaluations to one queryable CSV scoreboard."""

from pathlib import Path

import pandas as pd


def save_scores(path, rows):
    """Append a batch of candidate rows while preserving earlier runs."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(destination, dtype={"run_id": str}) if destination.exists() else pd.DataFrame()
    incoming = pd.DataFrame(rows)
    combined = pd.concat([previous, incoming], ignore_index=True)
    temporary = destination.with_suffix(".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(destination)
    return combined
