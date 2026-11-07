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


def save_run_report(scoreboard_path, run_id, model, selection, features, metric, cv_score):
    """Save a compact human-readable summary beside the scoreboard."""
    report = Path(scoreboard_path).with_name(f"run_{run_id}.txt")
    report.write_text(
        f"Run: {run_id}\nModel: {model}\nFeature selection: {selection}\n"
        f"Features kept: {', '.join(features)}\nNested CV {metric}: {cv_score:.4f}\n",
        encoding="utf-8",
    )
    return report
