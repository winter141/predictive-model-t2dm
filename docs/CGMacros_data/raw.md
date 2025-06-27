# Bio

| Field                      | Range                                                                    | Description                                                                                            |
|:---------------------------|:-------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------|
| Age                        | Numeric: 18-69                                                           | Age of participant in years at start of study                                                          |
| Gender                     | Factor: F or M                                                           | Male or Female                                                                                         |
| BMI                        | Numeric: 20.69-49.09                                                     | Body Mass Index of Participant at start of study                                                       |
| Body weight                | Numeric: 116.8-284.6                                                     | Weigth of participant at start of study in pounds                                                      |
| Height                     | Numeric: 59-72                                                           | Height of Participant in inches                                                                        |
| Self-identify              | Factor: African American Black, African American, Hispanic/Latino, White | Participant Self-Identification of Race/Ethnicity                                                      |
| A1c PDL (Lab)              | Numeric: 4.6-8.5                                                         | Range of hbA1c reading taken at start of study in mmol/mol                                             |
| Fasting GLU - PDL (Lab)    | Numeric: 79-218                                                          | Range of fasting glucose taken at start of study in mg/dL                                              |
| Insulin                    | Numeric: 2.5-46.4                                                        | Range of fasting insulin taken at start of study in mcU/mL                                             |
| Triglycerides              | Numeric: 40-1150                                                         | Range of fasting tryglicerides taken at start of study in mg/dL                                        |
| Cholesterol                | Numeric: 91-345                                                          | Range of fasting cholesterol taken at start of study in mg/dL                                          |
| HDL                        | Numeric: 24-106                                                          | Range of fasting HDL cholesterol taken at start of study in mg/dL                                      |
| Non HDL                    | Numeric: 38-283                                                          | Range of fasting Non-HDL cholesterol taken at start of study in mg/dL                                  |
| LDL (Cal)                  | Numeric: 21-260                                                          | Range of fasting LDL cholesterol taken at start of study in mg/dL, 800 is an error in calculation      |
| VLDL (Cal)                 | Numeric: 8-78                                                            | Range of fasting VLDL cholesterol taken at start of study in mg/dL, 400 indicates an erroneous reading |
| Cho/HDL Ratio              | Numeric: 1.7-400                                                         | Range of ratio of cholesterol to HDL taken at start of study in mg/dL                                  |
| Collection time PDL (Lab)  | Time: HH:MM                                                              | Time the fasting lab measurements were taken                                                           |
| #1 Contour Fingerstick GLU | Numeric: 80-220                                                          | Fingerstick glucose measurement reading #1 in mg/dL                                                    |
| Time (t)                   | Time: HH:MM                                                              | Time Fingerstick Reading Measurement 1 is taken                                                        |
| #2 Contour Fingerstick GLU | Numeric: 73-314                                                          | Fingerstick glucose measurement reading #2 in mg/dL                                                    |
| Time (t)                   | Time: HH:MM                                                              | Time Fingerstick Reading Measurement 2 is taken                                                        |
| #3 Contour Fingerstick GLU | Numeric: 67-247                                                          | Fingerstick glucose measurement reading #3 in mg/dL                                                    |
| Time (t)                   | Time: HH:MM                                                              | Time Fingerstick Reading Measurement 3 is taken                                                        |

# CGMacros-0XX
| Field               | Range                                         | Description                                                                                                                            |
|:--------------------|:----------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| Timestamp           | Date: Month/Day/Year HH:MM                    | Randomly shifted time stamps for days in the study that provide the incremental time stamp the CGM readings occurred                   |
| Libre GL            | Numeric: 40-400 mg/dL                         | The blood glucose reading provided by the Libre Pro sensor - with a minimum possible value of 40 and maximum possible value of 400     |
| Dexcom GL           | Numeric: 40-400 mg/dL                         | The blood glucose reading provided by the Dexcom G6 Pro sensor - with a minimum possible value of 40 and maximum possible value of 400 |
| HR                  | Numeric: 30-176                               | average heart rate over the last minute provided by the FitBit Sense Smartwatch in beats per minute                                    |
| Calories (Activity) | Numeric: 0-16.178                             | Fitbit estimate of calories burned in the past minute as captured by the FitBit Sense Smartwatch                                       |
| Mets                | Numeric: 10-176                               | The Metabolic Equivalent of Task estimate for the past minute as provided by the FitBIt Sense multiplied by 10                         |
| Meal Type           | Factor: Breakfast, Lunch, Dinner              | The indication of a meal start and which meal                                                                                          |
| Calories            | Numeric: 30-1180                              | Calories (estimated) in the consumed meal                                                                                              |
| Carbs               | Numeric: 0-176                                | Estimate of carbohydrate quantity (grams) in the consumed meal                                                                         |
| Protein             | Numeric: 3-176                                | Estimate of protein quantity (grams) in the consumed meal                                                                              |
| Fat                 | Numeric: 0-176                                | Estimate of fat quantity (grams) in the consumed meal                                                                                  |
| Fiber               | Numeric: 0-176                                | Estimate of fiber  quantity (grams) in the consumed meal                                                                               |
| Amount Consumed     | numeric: 0-100                                | Estimate of % of meal consumed                                                                                                         |
| Image Path          | Path to the specific meal in folder structure | indicates where the photo associated with the start or end of this meal is located in the dataset                                      |