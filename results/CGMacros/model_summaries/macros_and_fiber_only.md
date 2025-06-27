

**Model: XGBOOST** | n_estimators: 1000, learning_rate: 0.05, max_depth: 8

**Features:** Energy, Carbohydrate, Protein, Fat, Fiber

**R (100 iterations):** Mean 0.364, Std 0.043
## SHAP Analysis ##

SHAP Analysis Plot in: results/CGMacros/SHAP_PDP/macros_and_fiber_only_shap.png

Consider using shap.dependence_plot for individual feature analysis
## PDP Analysis ##

PDP Analysis Plot in: results/CGMacros/SHAP_PDP/macros_and_fiber_only_pdp.png

For categorical features consider running a PDP plot with categorical_features parameter.