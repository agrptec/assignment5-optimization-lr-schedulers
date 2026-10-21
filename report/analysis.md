# Assignment 5 — Optimization and Learning Rate Schedulers

## Experimental setup

We used the UCI Raisin dataset (900 observations, seven numerical features, two balanced classes). As in Assignment 4, we kept a stratified 60/20/20 train/validation/test split (`random_state=42`), fit the standardizer on the training split only, and used a shallow 7–20–3–1 fully connected network with L1 regularization (`λ=0.7`). The tutorial optimizer implementations were used with batch size 64 and 5,000 epochs. Each run began from the same weights and saw the same mini-batch ordering. Initial learning rates were 0.1 for GD and Momentum and 0.01 for Adam; these remained identical across the three scheduler choices within each optimizer.

The no-scheduler condition kept its rate constant. The supplied tutorial decay used `lr₀/(1+0.001×epoch)`, which is **inverse-time decay** mathematically, despite its “exponential” label in the assignment. Our Step LR Decay multiplied the initial rate by 0.7 every 1,000 epochs. We evaluated binary cross-entropy on the complete train and validation sets after each recorded epoch. The reported losses exclude the L1 penalty, so they can be compared directly across schedules.

## Findings

| Optimizer | Scheduler | Final validation loss | Validation accuracy |
| --- | --- | ---: | ---: |
| Adam | Constant | 0.5515 | 85.56% |
| GD | Tutorial decay | 0.5534 | 85.56% |
| Adam | Step | 0.5582 | 86.11% |
| Momentum | Tutorial decay | 0.5588 | 86.67% |
| Momentum | Step | 0.5640 | 86.67% |
| Momentum | Constant | 0.5706 | 86.11% |
| GD | Step | 0.5731 | 86.67% |
| Adam | Tutorial decay | 0.5774 | 84.44% |
| GD | Constant | 0.5821 | 85.56% |

- Relative to a constant rate, tutorial decay lowered final validation loss for GD (0.5821 → 0.5534) and Momentum (0.5706 → 0.5588), while it raised Adam's loss (0.5515 → 0.5774). The same scheduler did not help every optimizer.
- Step decay improved GD and Momentum validation accuracy to 86.67%, but their validation losses did not rank first. Accuracy depends on the 0.5 decision threshold; cross-entropy also reflects confidence, so the rankings can differ.
- Adam with a constant rate had the lowest final validation loss. After selecting it using validation data only, its held-out test accuracy was **87.22%**.
- Final training loss was roughly 0.26–0.27 for every run, while validation loss was 0.55–0.58. This gap suggests that the models fit the training split better than the validation split. The small validation set (180 examples) also makes small accuracy differences uncertain.

## What we learned

Learning-rate schedules change the path and end point of training, even with the same optimizer, data, initialization and epoch count. Decay helped GD and Momentum in this setup, but the constant-rate Adam run was best by validation loss. We should choose an evaluation metric before selecting a model, keep the test set held out, and avoid claiming that one optimizer or scheduler is universally best from one split and one set of hyperparameters.

## References

- Çinar, İ., Koklu, M., & Tasdemir, S. (2020). *Raisin* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5660T (CC BY 4.0).
- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. https://arxiv.org/abs/1412.6980.
- Course notebook: *Optimization Methods*, provided with Assignment 5.
