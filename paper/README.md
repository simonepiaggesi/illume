# Explanations Go Linear: Post-hoc Explainability for Tabular Data with Interpretable Meta-Encoding

## Reproduce paper experiments

Use a repository checkout containing `paper/` and `datasets/`. From the repository root (the directory containing `pyproject.toml`), install the paper extra dependencies and run the scripts as modules:

```bash
python -m pip install -e ".[paper]"
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

