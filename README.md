# Explanations Go Linear: Post-hoc Explainability for Tabular Data with Interpretable Meta-Encoding

Code and data for running [ILLUME](https://arxiv.org/abs/2504.20667), a novel post-hoc explanation method for tabular data that generate local interpretable surrogates from globally-trained meta-encoder networks. 

To run the method, see [the quickstart](https://github.com/simonepiaggesi/illume/blob/main/examples/quickstart.py) or the
[notebook](https://github.com/simonepiaggesi/illume/blob/main/examples/illume_example.ipynb). Run the notebook from the repository
root so it can find the example datasets. The [Appendix](https://github.com/simonepiaggesi/illume/blob/main/appendix.pdf) describes
the experimental settings.

Paper: Simone Piaggesi, Riccardo Guidotti, Fosca Giannotti, and Dino Pedreschi. "Explanations go linear: Post-hoc explainability for tabular data with interpretable meta-encoding." In 2025 IEEE International Conference on Data Mining (ICDM), pp. 663-672. IEEE, 2025. [[link]](https://ieeexplore.ieee.org/abstract/document/11391937/)

## Installation

```bash
pip install "mimosa-illume[model]"
```

From a local clone, use `pip install -e ".[model,dev]"` to run the smoke test and
quickstart. Use `pip install -e ".[notebook]"` for the notebook or
`pip install -e ".[paper]"` for the experiment scripts.

## Basic usage 

```python
import numpy as np
from illume import ILLUME

X_train = np.random.default_rng(1).normal(size=(8, 3))
X_val = np.random.default_rng(2).normal(size=(4, 3))
model = ILLUME(latent_dim=2, max_epochs=1, batch_size=4)
model.fit(
    X=(X_train, X_val),
    idx_num_cat=[[0], [1], [2]],
    params_dict={"num_k": None, "l_rec": 0.0, "l_kld": 1.0,
                 "l_so": 0.0, "l_co": 0.0, "l_st": 0.0},
)
W, Z = model.transform(X_val)
```

Full example: `examples/illume_example.ipynb`.

## Reproduce paper experiments

After installing the `paper` extra, run scripts from the repository root, for
example:

```bash
python -m paper.scripts.train_eval_feature_importance --dataset breast --bb xgb --seed 0
python -m paper.scripts.baseline_eval_feature_importance --dataset breast --bb xgb --seed 0
```

Run a `train_eval_*.py` script for the same dataset, black-box model, and seed
before a real-data `baseline_eval_*.py` script. The baseline scripts load the
black-box model saved by the training scripts. The SENECA scripts generate their
own data and can run independently.

Generated files are stored under `paper/`: `blackboxes/` contains
saved black-box estimators (`.p`), `models/` contains ILLUME weights (`.pt`)
and surrogate descriptions (`.pkl`), and `results/` contains explanation and
evaluation pickles (`.pkl`). These generated directories are excluded by
`.gitignore`.

For the experiments with feature importance and real data:
- ILLUME: [train_eval_feature_importance.py](https://github.com/simonepiaggesi/illume/blob/main/paper/scripts/train_eval_feature_importance.py)
- LIME, SHAP, INP-LR: [baseline_eval_feature_importance.py](https://github.com/simonepiaggesi/illume/blob/main/paper/scripts/baseline_eval_feature_importance.py)

For the experiments with decision rules and real data:
- ILLUME: [train_eval_decision_rule.py](https://github.com/simonepiaggesi/illume/blob/main/paper/scripts/train_eval_decision_rule.py)
- LORE, ANCHOR, INP-DT: [baseline_eval_decision_rule.py](https://github.com/simonepiaggesi/illume/blob/main/paper/scripts/baseline_eval_decision_rule.py)

For the experiments with SENECA synthetic data:
- ILLUME: [seneca_train_eval_feature_importance.py](https://github.com/simonepiaggesi/illume/blob/main/paper/scripts/seneca_train_eval_feature_importance.py) and [seneca_train_eval_decision_rule.py](https://github.com/simonepiaggesi/illume/blob/main/paper/scripts/seneca_train_eval_decision_rule.py) 
- Competitors: [seneca_baseline_eval_feature_importance.py](https://github.com/simonepiaggesi/illume/blob/main/paper/scripts/seneca_baseline_eval_feature_importance.py) and [seneca_baseline_eval_decision_rule.py](https://github.com/simonepiaggesi/illume/blob/main/paper/scripts/seneca_baseline_eval_decision_rule.py) 

## Cite us

If you use the code in this repository, please consider citing us:

```bibtex
@inproceedings{piaggesi2025explanations,
  title={Explanations go linear: Post-hoc explainability for tabular data with interpretable meta-encoding},
  author={Piaggesi, Simone and Guidotti, Riccardo and Giannotti, Fosca and Pedreschi, Dino},
  booktitle={2025 IEEE International Conference on Data Mining (ICDM)},
  pages={663--672},
  year={2025},
  organization={IEEE}
}
```
