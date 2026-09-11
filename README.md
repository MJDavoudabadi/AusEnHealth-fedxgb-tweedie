# AusEnHealth-fedxgb-tweedie

Federated histogram-based XGBoost with a Tweedie objective, for predicting a
non-negative, right-skewed, zero-inflated outcome (mortality rate) across
multiple data-holding clients (Australian states/territories), benchmarked
against an equivalent centralised model trained on pooled data.

This is a horizontal FL simulator: all clients share the same feature set but
hold different, non-overlapping observations. It runs as a single-machine
simulation, but is structured so that no step ever requires more than a
client's own raw rows plus small, already-aggregated numbers from other
clients — categorical levels, histogram bin edges, gradient/Hessian sums at
each split, evaluation metrics, and the early-stopping criterion are all
computed this way. See the comment block at the top of the script for a full
description of what is and isn't privacy-preserving about this design.

## Requirements

R (developed on R 4.4.3) with the following packages, installed automatically
by the script if missing:

- `data.table`
- `xgboost`
- `ggplot2`
- `gridExtra`

## Data layout

The script expects one subfolder per client under a single root directory,
each containing a chronological three-way split:

data_root/
├── ClientA/
│   ├── train.csv
│   ├── es_valid.csv
│   └── test.csv
├── ClientB/
│   ├── train.csv
│   ├── es_valid.csv
│   └── test.csv
└── ...


- `train.csv` — grows trees (both the federated and centralised models)
- `es_valid.csv` — used only for early stopping / boosting-round selection,
  never for final reported metrics
- `test.csv` — touched once, at the end, for all reported metrics and plots

Each CSV must contain the response column (`mortality_rate` by default) and
the predictor columns listed in `x_cols` near the top of the script.

## Usage

1. Open `fedxgb_tweedie.R` and set `data_root` (path to your client folders)
   and `output_dir` (where results are written).
2. Adjust `y_col` and `x_cols` if your predictor set differs from the
   default.
3. Adjust hyperparameters if needed (defaults below).
4. Run the script (e.g. `Rscript fedxgb_tweedie.R`, or source it in RStudio).

### Hyperparameters (defaults)

| Hyperparameter          | Symbol   | Value |
|--------------------------|----------|-------|
| Maximum number of bins   | B        | 256   |
| Tree depth                | T        | 5     |
| Maximum boosting rounds  | N        | 300   |
| Early stopping rounds     | K        | 30    |
| Learning rate              | η        | 0.05  |
| L2 regularisation          | λ        | 1.0   |
| Tweedie variance power     | ρ        | 1.5   |

## Outputs

Written to `output_dir`:

- `fl_xgboost_model.rds`, `centralised_xgboost_model.rds` — fitted models
- `fl_client_test_sufficient_stats.csv` — per-client (n, sum-of-loss)
  statistics the federated metrics are built from
- `fl_vs_central_per_state_metrics.csv` — per-state and pooled NLL / RMSSE /
  MASE, federated vs. centralised
- `fl_training_curve.csv` — early-stopping validation loss per round
- `central_test_predictions.csv` — centralised model's test-set predictions
- `plot_training_curve.png`, `plot_client_tweedie_nll.png`,
  `plot_per_state_fl_vs_central.png`, `plot_residual_histogram.png`,
  `plot_predicted_vs_actual.png`, `plot_fl_vs_central_residuals.png`,
  `plot_gradient_histograms.png` — diagnostic and comparison plots

## Notes

- Bin-edge construction, split-finding, and metric aggregation are federated
  (no client's raw rows are pooled); a small number of visualisation-only
  plots near the end of the script do pool individual predictions across
  clients purely for plotting convenience on a single machine — see the
  comment above `diag_pooled_fl_test_pred` in the script.
- This script does not implement encryption, secure aggregation, or
  differential privacy, and does not run over a real network — it implements
  the statistical/algorithmic structure a distributed deployment would sit
  on top of, not the network or cryptographic layer around it.
