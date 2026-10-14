# Backpropagation vs. Predictive Coding

This repository contains a runnable notebook comparing a feedforward neural
network trained with backpropagation against a predictive-coding (PC) model on
California housing-value regression. The notebook includes the mathematical
setup, preprocessing, training, validation-based inference-step selection,
held-out evaluation, and artifact export.

## Repository contents

- [`notebooks/icoglabs_backprop_vs_predictive_coding_house_values.ipynb`](notebooks/icoglabs_backprop_vs_predictive_coding_house_values.ipynb) —
  main experiment and analysis.
- [`notebooks/artifacts/house_value_regression/`](notebooks/artifacts/house_value_regression/) —
  model checkpoints, fitted preprocessing pipeline, metadata, predictions,
  training histories, and plots from the recorded run.
- [`doc/`](doc/) — background papers, notes, and presentation material used
  for this project.

## Dataset

The notebook downloads the `hosammhmdali/house-price-dataset` dataset through
KaggleHub and predicts `median_house_value`. It uses the eight numeric
predictors and the `ocean_proximity` categorical feature. Missing values are
imputed, categorical values are one-hot encoded, and numeric features are
standardized. The train/test split is made before fitting preprocessing, and
the preprocessing pipeline is fit using training data only.

The dataset is downloaded at notebook runtime; it is not included in this
repository. KaggleHub may require Kaggle credentials depending on the local
environment.

## Run the notebook

Use Python 3.12 or a compatible Python environment with Jupyter installed.
From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install jupyter
jupyter lab
```

Open the main notebook and run its cells from top to bottom. The notebook uses
seed `42`; training and validation results can still vary across hardware,
library versions, and nondeterministic kernels. Generated files are written to
`notebooks/artifacts/house_value_regression/`.

## Recorded results

The checked-in run used a 20% held-out test split and selected 10 predictive-
coding inference steps based on validation RMSE. On that test split:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Backpropagation | 38,755.10 | 57,255.46 | 0.7498 |
| Predictive coding | 39,903.80 | 57,636.65 | 0.7465 |

For this run, the backpropagation model performed slightly better on all three
reported metrics. This is a result for one seeded split and model setup, not
evidence that either training method is universally superior.

## Artifacts

The generated artifact directory contains:

- `backprop_model.pt` and `predictive_coding_model.pt` — model weights and
  architecture metadata.
- `preprocessor.pkl` — fitted feature preprocessing pipeline.
- `metadata.json` — target scaling, feature names, validation sweep, and test
  metrics.
- `test_predictions.csv` — actual and predicted held-out values.
- `training_history.csv` and `predictive_coding_layer_energies.csv` — saved
  training traces.
- `training_curves.png` — visualization of training objectives and layer
  energies.

## References

The PDFs in [`doc/`](doc/) provide supporting reading and project notes.
Please retain their original authorship and licensing information when
redistributing them.
