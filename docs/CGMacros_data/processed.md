## static_user

| Column        | Type / Range                                                            | Description                                       |
|---------------|-------------------------------------------------------------------------|---------------------------------------------------|
| UserID        | –                                                                       | Unique identifier for the participant             |
| Sex           | Numeric (0, 1)                                                          | Participant’s biological sex. 0 Female, 1 Male    |
| BMI           | Numeric: 20.69–49.09                                                    | Body Mass Index of participant at start of study  |
| Body weight   | Numeric: 116.8–284.6 (lbs)                                              | Weight of participant at start of study in pounds |
| Height        | Numeric: 59–72 (inches)                                                 | Height of participant in inches                   |
| Self-identify | Factor: African American Black, African American, Hispanic/Latino, White | Participant's self-identified race/ethnicity      |

---

## dynamic_user

| Column              | Type / Range            | Description                                                                 |
|---------------------|-------------------------|-----------------------------------------------------------------------------|
| UserID              | –                       | Unique identifier for the participant                                       |
| Timestamp           | –                       | Time of data capture                                                        |
| HR                  | Numeric: 30–176         | Average heart rate over the last minute from FitBit Sense (beats per minute) |
| Calories (Activity) | Numeric: 0–16.178       | Estimated calories burned in past minute from FitBit Sense                 |
| Mets                | Numeric: 10–176         | Metabolic Equivalent of Task estimate ×10 from FitBit Sense                |

---

## log

| Column           | Type / Range                                         | Description                                                       |
|------------------|------------------------------------------------------|-------------------------------------------------------------------|
| UserID           | –                                                    | Unique identifier for the participant                             |
| Timestamp        | –                                                    | Time of meal or food log                                          |
| Energy           | Numeric                                              | Energy content in kilocalories                                    |
| Carbohydrates    | Numeric                                              | Grams of carbohydrates                                            |
| Fat              | Numeric                                              | Grams of fat                                                      |
| Protein          | Numeric                                              | Grams of protein                                                  |
| Meal Type        | Factor: Breakfast, Lunch, Dinner                     | Type of meal                                                      |
| Fiber            | Numeric: 0–176                                       | Grams of fiber in the consumed meal                               |
| Amount Consumed  | Numeric: 0–100                                       | Estimated % of the meal that was consumed                         |

---

## cgm
*(Contains two CGM readings using Libre GL)*

| Column    | Type / Range   | Description                        |
|-----------|----------------|------------------------------------|
| UserID    | –              | Unique identifier for the participant |
| Timestamp | –              | Time of glucose reading            |
| Reading   | Numeric mmol/L | Glucose value                     |
