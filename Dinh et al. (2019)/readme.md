# A data-driven approach to predicting diabetes and cardiovascular disease with machine learning (2019)

## Brief Summary

Dinh et al. (2019) uses different ML models (logistic regression, support vector machines, random forest, and gradient boosting) on NHANES dataset to predict i) Diabetes and Cardiovascular disease ("CVD").

**Goal**: Identification mechanism for patients at risk of diabetes and cardiovascular diseases and key contributors to diabetes .

**Results**:

Best scores:

- CVB prediction based on 131 NHANES variables achieved an AU-ROC score of 83.9% .
- Diabetes prediction based on 123 NHANES variables achieved an AU-ROC score of 95.7% .
- Pre-diabetic prediction based on 123 NHANES variables achieved an AU-ROC score of 84.4% .

- Top 5 predictors in diabetes patients were 1) `waist size`, 2) `age`, 3) `self-reported weight`, 4) `leg length`, 5) `sodium intake`.

## Data

[NHANES](https://www.cdc.gov/nchs/index.htm).

- 123 (unspecified) variables from NHANES data from 1999-2014.
- 168 (unspecified) variables from NHANES data from 2003-2014

![Tables from Dinh et al. 2019](https://raw.githubusercontent.com/pipegalera/ml_diabetes/main/images/dinh_2019_Table4.png)

## Preprocessing and data decisions

  > The preprocessing stage also converted any undecipherable values (errors in datatypes and standard formatting) from the database to null representations.

  > Normalization was performed on the data using the following standardization model: x' = x−x^/σ ,

  > Downsampling was used to produce a balanced 80/20 train/test split.

  > Grid-search

  > 10-fold CV

## Target

`Diabetes = 1` if

- Glucose >= 126 mg/dL. OR;
- "Yes" to the question "Have you ever been told by a doctor that you have diabetes?"

`undiagnosed diabetes = 1` if

- Glucose >= 126 mg/dL. AND;
- "No" to the question "Have you ever been told by a doctor that you have diabetes?" and had a blood glucose level greater than or equal

`pre diabetes = 1` if

- Glucose 125 >= 100 mg/dL

`CVD = 1` if

- "Yes" to any of the the questions "Have you ever been told by a doctor that you had congestive heart failure, coronary heart disease, a heart attack, or a stroke?"

![Tables from Dinh et al. 2019](https://raw.githubusercontent.com/pipegalera/ml_diabetes/main/images/dinh_2019_Table1_3.png)
