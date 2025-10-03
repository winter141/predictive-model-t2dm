

**Model: XGBOOST** | n_estimators: 1000, learning_rate: 0.05, max_depth: 8

**Features:** Sex, Body weight, Height, Energy, Carbohydrate, Protein, Fat, cgm_p30, cgm_p60, cgm_p120, meal_hour, time_since_last_meal, Self-identity_African American, Self-identity_Black, African American, Self-identity_Hispanic/Latino, Self-identity_White, UserID_1, UserID_10, UserID_11, UserID_12, UserID_13, UserID_14, UserID_15, UserID_16, UserID_17, UserID_18, UserID_19, UserID_2, UserID_20, UserID_21, UserID_22, UserID_23, UserID_26, UserID_27, UserID_28, UserID_29, UserID_3, UserID_30, UserID_31, UserID_32, UserID_33, UserID_34, UserID_35, UserID_36, UserID_38, UserID_39, UserID_4, UserID_41, UserID_42, UserID_43, UserID_44, UserID_45, UserID_46, UserID_47, UserID_48, UserID_49, UserID_5, UserID_6, UserID_7, UserID_8, UserID_9

**R (100 iterations):** Mean 0.727, Std 0.035
## SHAP Analysis ##

SHAP Analysis Plot in: results/CGMacros/SHAP_PDP/SuUiFmTgTf_shap.png

Consider using shap.dependence_plot for individual feature analysis
## PDP Analysis ##

PDP Analysis Plot in: results/CGMacros/SHAP_PDP/SuUiFmTgTf_pdp.png

For categorical features consider running a PDP plot with categorical_features parameter.