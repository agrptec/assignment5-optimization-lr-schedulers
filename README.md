# Assignment 5 — Optimization and Learning Rate Schedulers

Comparison of mini-batch Gradient Descent, Momentum and Adam with a constant learning rate, the decay formula supplied in the tutorial, and a custom Step LR Decay scheduler. The shallow network uses the **Raisin** dataset and the **7–20–3–1 network with L1 λ = 0.7** selected in [Assignment 4](https://github.com/agrptec/assignment4-raisin-regularization).

## Run in Colab

[Open the completed notebook in Colab](https://colab.research.google.com/github/agrptec/assignment5-optimization-lr-schedulers/blob/main/Assignment_5_Optimization_LR_Schedulers.ipynb), then choose **Runtime → Run all**. The first code cell retrieves the repository helpers and data automatically. CPU runtime is enough; no Drive mount is needed.

`Optimization_methods copy.ipynb` is the team's original upload. `Assignment_5_Optimization_LR_Schedulers.ipynb` preserves its tutorial and adds the Raisin experiment. `Optimization_methods.ipynb` is the original tutorial reference.

## Reproducibility and results

- Dataset: 900 raisins, seven features, balanced binary classes; stratified 60/20/20 train/validation/test split with random state 42. Standardization is fitted on training data only.
- Architecture and regularization: 7–20–3–1 network, L1 λ = 0.7, as selected in Assignment 4.
- All nine runs use the same weight initialization, mini-batch order, 5,000 epochs and batch size 64. GD and Momentum start at 0.1; Adam starts at 0.01. Each optimizer keeps its initial rate across scheduler conditions.
- The supplied “exponential” formula is mathematically inverse-time decay, `lr0 / (1 + 0.001 × epoch)`. Step LR Decay multiplies the rate by 0.7 every 1,000 epochs. Curves show binary cross-entropy without the L1 term, evaluated on full train and validation splits.
- The notebook writes `results/summary.csv`, `results/loss_curves.png`, and `results/learning_rates.png` when run. A report draft and its editable text are in `report/`. It selects a model by final validation loss before reporting held-out test accuracy.

**Canvas submission:** download the executed `.ipynb` and submit the file itself, along with the PDF report. A notebook link is not accepted by the assignment instructions. The repository link is for collaboration and reproducibility.

## Dataset attribution

Çinar, İ., Koklu, M., & Tasdemir, S. (2020). *Raisin* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5660T](https://doi.org/10.24432/C5660T). The dataset is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `data/raisin.csv` is converted from the UCI ARFF file without changing observations.
