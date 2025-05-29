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

## Important Notes
Can't actually digest fiber.

Look at digestible carbohydrates which equals: total_carb - fiber

Rather than iAUC, we could look at the max, rate of change, etc.

Fiber slows down absorbtion of glucose


Insulin helps sugar move from your blood into your cells.


Cells in your body need sugar for energy, Sugar from foods makes your blood sugar levels go up.



### What is Glycemic Index
Unique to each individual. No food in ~10hours, no exercise.

Complex

1. Drink sugar drink
2. Measure AUC for 2 hours
3. Wait
4. Eat Food
5. Measure AUC for 2 hours
6. Return: (Food_AUC / SugarDrink_AUC) * 100

For example: Bread is spikes blood glucose 34% as much as sugar drink.

### Low vs High blood sugar

Low: Hypoglycemia <70
- hunger, irritability, fatigue, trouble concentrating, sweating, confusion, fast heartbeat, shaking, headache

High: Hyperglycemia
- extreme thirst, dry mouth, weakness, nausea, blurry vision, need to pee

# Predicting the Goodness of Foods from CGM Data

This project aims to predict the "goodness" of foods based on an individual's continuous glucose monitor (CGM) response after consuming a meal. Due to the complexity of glycemic responses, we explore custom metrics beyond standard incremental Area Under the Curve (iAUC) to create a more robust and personalized prediction model.

---

## Problem Statement

Glycemic responses to food are highly individualized and affected by a wide range of factors, including:

- Time of day
- Previous meals
- Physical activity
- Sleep
- Stress
- Hormonal fluctuations

Standard metrics like iAUC do not account for this variability, making them potentially unreliable as standalone target variables.

---

## Project Goal

To develop a machine learning model that predicts the glycemic impact—or "goodness"—of foods using CGM data, enriched with contextual and physiological features.

---

## Target Variable Definition

### Challenges with iAUC:
- Sensitive to baseline glucose
- Affected by time of day and recent activity
- Doesn't capture duration or speed of glucose rise/fall

### Alternatives Considered:

| Metric | Description |
|--------|-------------|
| iAUC (0–120 min) | Area under the curve above baseline |
| Peak Glucose | Maximum glucose level post-meal |
| Delta Glucose | Peak minus baseline |
| Duration > threshold | Time glucose spends above a set level (e.g. 140 mg/dL) |
| Slope | Speed of glucose increase |

### Custom Goodness Score

A weighted composite metric is used as the target:

```python
goodness_score = (
    -0.4 * iAUC_120
    - 0.3 * (peak_glucose - baseline)
    - 0.2 * duration_above_140
    - 0.1 * time_to_peak
)
```

# TODO
- Create PDP Plots
- Calculate new iAUC, which considers baseline

The baseline should probably be the closet reading BEFORE not AFTER.
Need to change this, but it shouldn't really make a big difference.
