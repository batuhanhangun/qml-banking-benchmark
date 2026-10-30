# Quantum, Hybrid, and Classical Machine Learning for Credit Scoring, Churn Prediction, and Fraud Detection

Code for a benchmark of quantum kernel, variational, and hybrid quantum-classical models against tuned classical machine learning on three banking tasks. Quantum circuits are simulated exactly (noiseless statevector simulation), and everything runs on an ordinary CPU.

## Datasets

| Task | Dataset | Samples | Features | Minority class | Source |
|---|---|---|---|---|---|
| Credit scoring | Statlog German Credit | 1,000 | 20 | 30.0% | [UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| Customer churn | Bank Customer Churn (Churn_Modelling) | 10,000 | 10 | 20.4% | [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) |
| Fraud detection | Credit Card Fraud (ULB), deduplicated | 283,726 | 30 | 0.167% | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

The datasets are not redistributed here. Download them and place `german.data`, `Churn_Modelling.csv`, and `creditcard.csv` in `data/raw/`. Exact duplicate rows of the ULB file are removed before cross-validation.

## Models

- **Quantum:** QSVC with a ZZ fidelity kernel, variational quantum classifier (VQC)
- **Hybrid:** dressed QNN (classical encoder, variational quantum layer, classical head), compared with a parameter-matched MLP
- **Classical:** logistic regression, linear and RBF SVM, random forest, XGBoost, LightGBM, MLP

Quantum models use 4, 6, and 8 qubits. Classical models run both on the same reduced inputs and on all features. Every model gets the same tuning budget.

## Evaluation

Stratified 5-fold cross-validation with an inner 3-fold random search; encoding, scaling, PCA, and resampling are fitted on training folds only. For fraud, legitimate transactions are undersampled 1:4 in training folds, and test folds keep the natural fraud rate. Metrics are ROC-AUC, PR-AUC, F1, G-mean, and recall at fixed false positive rates. Models are compared with the corrected resampled t-test and Holm adjustment.

## Usage

```bash
pip install -r requirements.txt
pip install -e .
pytest
python -m qmlfin.check_data    # dataset and fold summary
python -m qmlfin.experiment --dataset german_credit --models lr rf xgboost --regimes full 4 6 8
```

Full grid (760 jobs) and tables:

```bash
python -m qmlfin.grid --list
python -m qmlfin.grid --run-all       # finished jobs are skipped, so a run can be resumed
python -m qmlfin.report               # tables, statistical tests, and circuit resources in results/tables/
```

To split the grid across processes or machines, run single jobs with `python -m qmlfin.grid --group sklearn --task-id N` (and `--group torch`). The code runs on CPU by default; set `torch.device` to `cuda` in `configs/default.yaml` to use a GPU.

Circuits are simulated with a batched statevector simulator written in PyTorch (`src/qmlfin/quantum/sim.py`). The tests check it against PennyLane, which is also used to count circuit depth and two-qubit gates.
