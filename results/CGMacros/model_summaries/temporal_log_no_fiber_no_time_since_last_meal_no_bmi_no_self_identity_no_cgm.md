

**Model: XGBOOST** | n_estimators: 1000, learning_rate: 0.05, max_depth: 8

**Features:** Sex, Body weight, Height, Energy, Carbohydrate, Protein, Fat, meal_hour

**R (100 iterations):** Mean 0.640, Std 0.037
## SHAP Analysis ##

SHAP Analysis Plot in: results/CGMacros/SHAP_PDP/temporal_log_no_fiber_no_time_since_last_meal_no_bmi_no_self_identity_no_cgm_shap.png

Consider using shap.dependence_plot for individual feature analysis
## PDP Analysis ##

PDP Analysis Plot in: results/CGMacros/SHAP_PDP/temporal_log_no_fiber_no_time_since_last_meal_no_bmi_no_self_identity_no_cgm_pdp.png

For categorical features consider running a PDP plot with categorical_features parameter.