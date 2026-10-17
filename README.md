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
│   ├── preprocessing.py    # cleaning + development/test split
│   ├── model.py             # model and full pipeline construction
│   ├── evaluate.py         # default-model CV + final evaluation
│   ├── tuning.py           # Optuna search + nested CV
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
| 5 | Hyperparameter tuning | Search model settings with Optuna inside three-fold CV, and compare each tuning procedure with five outer folds. Refit the selected settings on all development rows. |
| 6 | Ensembles and feature challenges | Add an oblique forest and AdaBoost, tune the ensemble candidates, then check three fixed-formula features and AdaBoost with a logistic base learner. |

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
- a five-fold cross-validation comparison of the default models in `config.yaml`'s `models` list
- when `tuning.enabled` is true, a nested cross-validation comparison of tuned models; inner folds choose hyperparameters, outer folds evaluate the tuning procedure
- **train accuracy and test accuracy** for the candidate selected by mean nested-CV accuracy
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this, plus each model's individual fold scores, search results and selected hyperparameters, is saved to a timestamped file in `results/` (e.g. `results/run_20261017_232940.txt`). Adjust `models` and `tuning.search_spaces` in `config.yaml` to change candidates. `tuning.n_trials` controls the search budget per model; `tuning.n_splits` controls the inner folds. Set `tuning.enabled: false` to reproduce the Week 4 default-model workflow. Keep the `test_set` settings fixed, and use the nested-CV scores for model choices rather than the test score.
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

# Week 5 - Hyperparameter Tuning Results

This week I wanted to find out whether the trees did badly in Week 4 because of the algorithms themselves or because I had left their settings at the defaults. I used Optuna to try different settings for the decision tree, random forest and logistic regression. I left the dummy classifier as a baseline.

For each model, the inner three folds chose the settings and the outer five folds measured how well that tuning process worked on unseen rows. I compared models using the outer scores. The best Optuna trial is useful for choosing settings, but its score is likely to look a little too good after trying many options.

| Model | Default CV mean ± SD | Tuned nested CV mean ± SD | Mean train–validation gap | Trials |
|---|---:|---:|---:|---:|
| Dummy | 0.549 ± 0.000 | 0.549 ± 0.000 | 0.000 | 0 |
| Logistic regression | 0.673 ± 0.014 | 0.672 ± 0.016 | 0.004 | 20 |
| Decision tree | 0.625 ± 0.017 | 0.674 ± 0.010 | 0.011 | 30 |
| Random forest | 0.631 ± 0.016 | **0.681 ± 0.014** | 0.008 | 15 |

The two tree models improved quite a lot after tuning. The decision tree went from 0.625 to 0.674, and the random forest from 0.631 to 0.681. Logistic regression barely changed. The random forest came first, although the difference from the other two models is small compared with how much the scores vary between folds. I would call it the best model in this run, but not a clear winner in general.

The final search chose a random forest with `n_estimators=50`, `max_depth=12`, `min_samples_leaf=46` and `max_features=0.5`. After training it on all the development rows, I got 0.673 accuracy on the test split. We had already looked at this same test split in earlier practicals, so I would use the nested-CV scores for the comparison rather than treat the test score as a completely fresh estimate. The full fold scores and reports are in `results/`.

# Week 6 - Ensembles and Challenges

This week I added AdaBoost and a simple oblique forest. The oblique forest is the notebook's scikit-learn construction: each bagged tree sees its own sparse random projection of the features. It is not the `treeple` implementation of SPORF. Both models still go through the same preprocessing pipeline and cross-validation as the earlier models.

I ran six Optuna trials per tuned model, with three inner and five outer folds. This smaller budget kept the run manageable tonight; the settings may change with a wider search. Every model used the same outer folds and the same three engineered features listed below.

| Model | Default CV mean ± SD (gap) | Tuned nested CV mean ± SD (gap) | Best trial's inner score | Final chosen settings |
|---|---:|---:|---:|---|
| Dummy | 0.549 ± 0.000 (0.000) | 0.549 ± 0.000 (0.000) | — | None |
| Logistic regression | 0.675 ± 0.010 (0.001) | 0.675 ± 0.012 (0.001) | 0.676 | `C=0.0177` |
| Decision tree | 0.630 ± 0.015 (0.172) | 0.673 ± 0.006 (0.010) | 0.675 | depth 4, leaf 1, entropy |
| Random forest | 0.632 ± 0.015 (0.171) | 0.676 ± 0.010 (0.019) | 0.677 | 100 trees, depth 4, leaf 2, `max_features=sqrt` |
| Oblique random forest | 0.641 ± 0.010 (0.161) | **0.677 ± 0.013 (0.010)** | 0.678 | 50 trees, 8 projected features, depth 9, leaf 77 |
| AdaBoost | 0.671 ± 0.009 (0.007) | 0.673 ± 0.008 (0.015) | 0.675 | depth 2, 94 rounds, learning rate 0.391 |

The oblique forest ranks first by mean outer-fold accuracy, so it is the model selected by the pipeline. Its lead over the ordinary forest is only 0.001, far smaller than either model's fold-to-fold standard deviation. I would report **0.677 ± 0.013 nested-CV accuracy** as the estimate for this tuned oblique-forest procedure, while saying plainly that these models are too close to call a clear overall winner. The selected forest got 0.674 on the reused test split. The chosen tree settings make sense on this dataset: both forests restrict tree size rather than leaving every tree fully grown. The full fold scores and classification and fairness reports are in `results/`.

## Challenge 1: fixed-formula features

I added `priors_per_adult_year` from the notebook, plus `juvenile_offense_total` (the sum of the three juvenile counts) and `age_priors_interaction` (age multiplied by prior count). These formulas use values from each row only. They do not need a mean, category frequency, or any other statistic learned from the dataset, so I can calculate them before the split. Missing inputs remain missing and are imputed inside each training fold.

For a quick challenge check, I compared each model with and without these columns using the same three outer folds, two inner folds and four Optuna trials. These are smaller runs than the main comparison, so I treat the differences as exploratory.

| Model | Without features | With features | Difference |
|---|---:|---:|---:|
| Logistic regression | 0.672 ± 0.010 | 0.675 ± 0.007 | +0.003 |
| Random forest | 0.679 ± 0.011 | 0.676 ± 0.002 | -0.003 |
| AdaBoost | 0.680 ± 0.014 | 0.674 ± 0.005 | -0.006 |

Logistic regression gained a little. The ensembles did not, and none of the differences is larger than the fold-to-fold variation. The three extra features stay in the pipeline as an experiment, but this run does not give me a strong reason to expect them to improve an ensemble.

## Challenge 2: boosting logistic regression

On the same three development folds, ordinary logistic regression scored 0.674 ± 0.005 and AdaBoost with logistic regression as its base learner scored 0.673 ± 0.010. Boosting did not help here. A weighted sum of linear *scores* would still be linear; AdaBoost combines the base classifiers' hard votes, so its boundary need not be one straight line. In this dataset, that extra flexibility did not improve accuracy in this small check.
