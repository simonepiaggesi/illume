# Explanations Go Linear: Post-hoc Explainability for Tabular Data with Interpretable Meta-Encoding
Code and data for running [ILLUME](https://arxiv.org/abs/2504.20667). The code has been tested with Python 3.8.12.

To run the method, please look at [main.ipynb](illume/main.ipynb) to get familiar with data loading, model training, and explanation inference. Please refer also to the [Appendix](appendix.pdf) for detailed experimental settings and additional insights. 

## Reproducibility
First, set up the conda environment using the provided YAML file:
   ```bash
   conda env create -f env.yml
   conda activate illume
   ```
Then, navigate to the [scripts/](illume/scripts/) folder and run the scripts.

For the experiments with feature importance and real data:
- ILLUME: [train_eval_feature_importance.py](illume/scripts/train_eval_feature_importance.py)
- LIME, SHAP, INP-LR: [baseline_eval_feature_importance.py](illume/scripts/baseline_eval_feature_importance.py)

For the experiments with decision rules and real data:
- ILLUME: [train_eval_decision_rule.py](illume/scripts/train_eval_decision_rule.py)
- LORE, ANCHOR, INP-DT: [baseline_eval_decision_rule.py](illume/scripts/baseline_eval_decision_rule.py)

For the experiments with SENECA synthetic data:
- ILLUME: [seneca_train_eval_feature_importance.py](illume/scripts/seneca_train_eval_feature_importance.py) and [seneca_train_eval_decision_rule.py](illume/scripts/seneca_train_eval_decision_rule.py) 
- Competitors: [seneca_baseline_eval_feature_importance.py](illume/scripts/seneca_baseline_eval_feature_importance.py) and [seneca_train_baseline_decision_rule.py](illume/scripts/seneca_baseline_eval_decision_rule.py) 
