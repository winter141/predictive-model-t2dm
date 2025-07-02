

**Model: XGBOOST** | n_estimators: 1000, learning_rate: 0.05, max_depth: 8

**Features:** Sex, BMI, Body weight, Height, Energy, Carbohydrate, Protein, Fat, Fiber, Self-identity_African American, Self-identity_Black, African American, Self-identity_Hispanic/Latino, Self-identity_White

**R (100 iterations):** Mean 0.628, Std 0.043
## SHAP Analysis ##

SHAP Analysis Plot in: results/CGMacros/SHAP_PDP/basic_log_with_personal_info_shap.png

Consider using shap.dependence_plot for individual feature analysis
## PDP Analysis ##

PDP Analysis Plot in: results/CGMacros/SHAP_PDP/basic_log_with_personal_info_pdp.png

For categorical features consider running a PDP plot with categorical_features parameter.