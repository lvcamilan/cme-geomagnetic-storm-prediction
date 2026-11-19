# Geomagnetic Storm Prediction from CME Events (ML2 Project)

A reproducible Machine Learning 2 project predicting Earth-impacting geomagnetic storms (`target_gst`) from Coronal Mass Ejection (CME) data (`cme_events_2year.csv`, derived from NASA DONKI), with emphasis on temporal leakage prevention, extreme class imbalance (`62.36:1`), model comparison, feature engineering, and interpretability.

## Key Methodology & Design Choices

- **Leakage-Free Target & Chronological Split:** Avoids leaky target definitions (`potentially_geoeffective`) by constructing `target_gst` from `linked_events`. Data is sorted by `start_time` and split chronologically (`80/20` train/test: `n = 2,280` train, `n = 571` test, `10` positive test cases) without shuffling or splitting identical timestamps across train and test.
- **Train-Only Preprocessing & Imbalance Handling:** Imputation, scaling, class weighting (`class_weight`, `scale_pos_weight`, positive-weighted `BCEWithLogitsLoss`), and chronological early stopping for the neural network are fit strictly on training data.
- **Imbalance-Aware Evaluation:** Evaluates models primarily via **PR-AUC** (against a `0.018` random positive-rate baseline) alongside precision, recall, F1-score, and confusion matrices on the minority class.
- **Models & Ensemble:** Compares a `LogisticRegression` baseline against `RandomForestClassifier`, `XGBClassifier`, a custom PyTorch `TorchMLP`, and an unweighted soft-voting ensemble (`XGBoost` + `RandomForest` + `NeuralNet`).
- **Feature Engineering & Interpretability:** Compares `BASE_FEATURES` (`speed`, `half_angle`, `latitude`, `longitude`) against past-only engineered features (`earth_alignment_angle`, `prev_24h_cme_count`, `prev_24h_max_speed`, using `closed="left"` rolling windows), followed by native XGBoost importance and 20-repeat test permutation importance on PR-AUC.

## Results Summary (Test Set, `n = 571`, `10` Positives)

| Model | PR-AUC (4 Feat.) | PR-AUC (7 Feat.) | Delta | Precision (Class 1, 4 Feat.) | Recall (Class 1, 4 Feat.) | F1 (Class 1, 4 Feat.) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LogReg** (Baseline) | `0.183` | `0.307` | `+0.124` | `0.082` | `0.400` | `0.136` |
| **RandomForest** | `0.203` | `0.261` | `+0.058` | `0.167` | `0.100` | `0.125` |
| **XGBoost** | `0.343` | `0.305` | `-0.038` | `0.500` | `0.300` | `0.375` |
| **NeuralNet (`TorchMLP`)** | `0.194` | `0.256` | `+0.061` | `0.238` | `0.500` | `0.323` |
| **Soft-Voting Ensemble** | `0.284` | `0.269` | `-0.014` | `0.333` | `0.300` | `0.316` |

- **Main Takeaways:** `XGBoost` achieves the highest PR-AUC (`0.343`) on the base feature set, while `TorchMLP` reaches the highest recall (`0.500`) on rare positive events. Adding geometric and historical features substantially improves `LogReg`, `RandomForest`, and `TorchMLP`, whereas `XGBoost` sees slight noise from `prev_24h_cme_count` (`-0.0904` mean permutation PR-AUC drop). Permutation importance highlights CME geometry and width (`longitude` / `earth_alignment_angle` and `half_angle`) as key drivers of PR-AUC.

## Project Structure

```text
.
├── ml2_space_weather.py   # End-to-end training, evaluation, feature comparison, and importance script
├── README.md
└── .gitignore
```

## Setup & Usage

1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn xgboost torch
   ```
2. Place `cme_events_2year.csv` in your working directory (or update `DATA_PATH` in `ml2_space_weather.py`).
3. Run the pipeline:
   ```bash
   python ml2_space_weather.py
   ```
