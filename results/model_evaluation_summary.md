# Model Evaluation


**Evaluation Method:**
- R is calculated using `scipy.stats.pearsonr`
- **100 simulations** are run, and the **mean and variation** are measured.
- Each simulation uses an **80/20 split**:  
  - 80% training  
  - 20% test
---

### Viewing Data
- All data is within this folder `results`
- Folder for each dataset, e.g. CGMacros which contain
  - `model_summaries`
    - .md files containing a short summary of features, hyperparameters, SHAP/PDP image locations, r score
  - `SHAP_PDP`
    - .png plots of SHAP/PDP analysis for each model


### Model Evaluation Summary
**CGMacros**
- Just food macro and fiber, got a R score of 0.364
- Adding personal info (from bio), this increased to 0.628
- Adding temporal data (cgm_p30, cgm_p60, cgm_p120, meal_hour, time_since_last_meal) improved this to 0.708
- Adding user id as a categorical feature increased to 0.743
  - Note: certain changes need to be made to make sure this scales well with multiple users.
