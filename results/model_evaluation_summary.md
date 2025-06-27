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
- Adding personal info (from bio), this increased to 0.624
