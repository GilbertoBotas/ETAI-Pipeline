"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full report (accuracy, classification report, fairness table) to a timestamped file in `results/` instead, so youcan open it again later, or compare two runs side by side after changing something in config.yaml.
"""
import os
from datetime import datetime


def save_run(results_dir: str, config: dict, report_text: str, selected_model: dict) -> str:
    """
    Writes one run's full report to a timestamped .txt file inside
    `results_dir` (created automatically if it doesn't exist yet) and
    returns the path that was written.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    header = (
        f"Run: {timestamp}\n"
        f"Selected model: {selected_model['type']}  params={selected_model.get('params', {})}\n"
        f"Test size: {config['test_set']['size']}  "
        f"random_state: {config['test_set']['random_state']}\n"
        f"CV: {config['cv']['n_splits']} stratified folds, "
        f"scoring={config['cv']['scoring']}, random_state={config['cv']['random_state']}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)

    return path
