# Data Format

```python
import pandas
data = {
    "static_user": pandas.DataFrame,
    "dynamic_user": pandas.DataFrame,
    "log": pandas.DataFrame,
    "cgm": pandas.DataFrame
}
```

### static_user

| Column         | Required | Description                      |
|----------------|----------|----------------------------------|
| UserID         | Yes      | Unique identifier                |
| Sex, BMI, etc. | No       | Other demographic information    |

### dynamic_user (optional)

| Column                          | Required | Description                     |
|---------------------------------|----------|---------------------------------|
| UserID                          | Yes      | Unique identifier               |
| Timestamp                       | Yes      | Time of data capture            |
| Heart rate, exercise data, etc. | No       | Time-varying physiological data |

### log

| Column                 | Required | Description                               |
|------------------------|----------|-------------------------------------------|
| UserID                 | Yes      | Unique identifier                         |
| Timestamp              | Yes      | Time of meal or log                       |
| Energy                 | Yes      | Energy in kilocalories                    |
| Carbohydrates          | Yes      | Grams of carbohydrates                    |
| Fat                    | Yes      | Grams of fat                              |
| Protein                | Yes      | Grams of protein                          |
| Fiber, Meal Type, etc. | No       | Additional nutritional or contextual info |

### cgm

| Column    | Required | Description                  |
|-----------|----------|------------------------------|
| UserID    | Yes      | Unique identifier            |
| Timestamp | Yes      | Time of glucose reading      |
| Reading   | Yes      | Glucose value (e.g., mmol/L) |
