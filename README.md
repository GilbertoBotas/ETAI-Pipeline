# Estudante 
Gilberto Antero Botas Júnior - 20260523

# Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | Data diagnosis and preprocessing | Clean invalid values and categories; impute, encode, and scale features within the model pipeline. |
| 4 | Cross-validation and model comparison | Compare four complete preprocessing-and-model pipelines on the same five stratified folds of the development set. Select by mean validation accuracy, refit on all development rows, then report the held-out test result. |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, cleans the data, separates the test rows, and prints:
- a cross-validation comparison of all candidates in `config.yaml`'s `models` list, including mean validation accuracy, standard deviation across folds, and the mean train-validation gap
- **train accuracy and test accuracy** for the candidate selected by mean cross-validation accuracy
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this, plus each model's individual fold scores, is saved to a timestamped file in `results/` (e.g. `results/run_20260916_143012.txt`). Adjust the candidates in `config.yaml`'s `models` list to compare a different set. Keep the `test_set` settings fixed, and use the cross-validation scores for model choices rather than the test score.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.

# Week 2 - Result Progress
After running the code for the first time, the logistic regression model is a better model for this dataset, because although the train accuracy (67,9%) is lower than the decision tree model (82,9%), the test accuracy is bigger and the gap between the train and test accuracies is smaller, becoming less prone to overfitting.

# Week 3 - Data Cleaning Results
After running the code with data cleaning implemented in the pipeline, the logistic regression model gap between the train accuracy and test accuracy increases from +0.001 to +0.019 (train accuracy of .679 to .676 and test from .678 to .657), and for the decision tree, the gap went from +0.197 to +0.181 (train accuracy of 0.829 to 0.792 and test from 0.632 to 0.611). Overall the logistic regression model is still between because the test accuracy is higher and the gap is smaller than the decision tree model, making it's generalization factor still better.

# Week 4 - Data Processing Results
After running the code on the 4 models (dummy, decision tree, logistic regression and random_forest), logistic regression is still the best model to pick out of all of them. With imputation, Encoding and Robust Scaling applied, the model achieved the highest test accuracy of 65,7%, on validation with a gap of +0.018.

# Cross-Validation Results

The development set is evaluated with the same five stratified folds for every model. Preprocessing is inside the pipeline, so each fold learns its imputation, encoding, and scaling only from that fold's training rows. The model with the highest mean validation accuracy is refit on all development rows before one held-out test evaluation.

| Model | CV accuracy (mean ± sample SD) | Mean train–validation gap |
|---|---:|---:|
| Dummy | 0.549 ± 0.000 | 0.000 |
| Logistic regression | **0.673 ± 0.014** | 0.002 |
| Decision tree | 0.625 ± 0.017 | 0.176 |
| Random forest | 0.631 ± 0.016 | 0.170 |

Logistic regression has the best mean validation accuracy and a small train–validation gap. The selected model scored 0.657 on the held-out test set. The tree models have much larger gaps, suggesting overfitting. These numbers use accuracy, so the classification report and false-positive rates should also be considered when judging practical performance.

**Methodological note:** Earlier weeks already inspected the same test split and used repeated splits to select preprocessing. This test result is useful for the class comparison, but it is not a fully untouched final estimate. Future choices should be made from development-set cross-validation only; a genuinely new external dataset would be needed for a fresh final estimate.
