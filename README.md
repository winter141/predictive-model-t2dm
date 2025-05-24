# Version 1

## Data

### Raw

#### TEI_Cleaned
*Food log data including metadata and nutrient values.*

- `Sex`
- `UserID`
- `Date`
- `Time`
- `Timestamp`
- `FoodItem`: *str*
- `Energy`
- `Carbohydrate`
- `Protein`
- `Fat`
- `Tag`: `"scan"` | `"text"`
- `Weight`: `"default"` | `"changed"`

#### CGM_Cleaned
*Continuous Glucose Monitoring (CGM) timeseries data.*

- `UserID`
- `NZT`: *timestamp*
- `value`: *float*

## Current Implementation

**Date:** 23/05/2025

First we must join data by adding the `cgm_window`: incremental area under curve 2 hours after each food log.

**Thinking**
Now we have many options we can look at the raw log or normalise, 
does this matter with blood glucose levels? ...probably.

Ok but we also need to consider other logs during two hours after logged.

Maybe have a visualisation of CGM levels noting food logs for each individual.

And then lets do some simple 2d graphs for each macronutrient, before running full XGBoost.

- Consider as group
- Consider separate data per individual



