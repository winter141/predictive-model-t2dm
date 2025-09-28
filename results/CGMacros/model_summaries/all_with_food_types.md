

**Model: XGBOOST** | n_estimators: 1000, learning_rate: 0.05, max_depth: 8

**Features:** Sex, Body weight, Height, Energy, Carbohydrate, Protein, Fat, cgm_p30, cgm_p60, cgm_p120, meal_hour, time_since_last_meal, UserID_1, UserID_2, UserID_3, UserID_4, UserID_5, UserID_6, UserID_7, Self-identity_Hispanic/Latino, Self-identity_White, Food Types_Undefined, Food Types_dairy_products_meat_fish_eggs_tofu, Food Types_grains_potatoes_pulses, Food Types_non-_alcoholic_beverages, Food Types_sweets_salty_snacks_alcohol, Food Types_vegetables_fruits

**R (10 iterations):** Mean 0.603, Std 0.096
## SHAP Analysis ##

SHAP Analysis Plot in: results/CGMacros/SHAP_PDP/all_with_food_types_shap.png

Consider using shap.dependence_plot for individual feature analysis
## PDP Analysis ##

PDP Analysis Plot in: results/CGMacros/SHAP_PDP/all_with_food_types_pdp.png

For categorical features consider running a PDP plot with categorical_features parameter.