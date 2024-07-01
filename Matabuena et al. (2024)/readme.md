# Deep Learning Framework with Uncertainty Quantification for Survey Data: Assessing and Predicting Diabetes Mellitus Risk in the American Population (2024)

## Brief Summary


## Data

## Preprocessing and data decisions

## Target

Diabetes defined as:

i) Glucose ≥126 mg/dL or;
ii) HbA1c ≥ 6.5% or;
iii) Diagnosed Diabetes.



## 🚧 Preliminary notes:

- **The paper uploaded to arxiv is an old pre-print, the points below will be probably addressed in the final journal submission**

- Race was mentioned but not added in any model.
- It would be good to see overfitting the training data to make sure the problem is solvable in the first place by the network
- Not specified:

  - Benchmark.
  - Why NNs and not other non-parametric methods that [are usually better for tabular data](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9998482).
  - Why they choose the specific NN architecture (rather generic?)
  - Number of epochs. Needs a Train/Val viz at least?
  ![](https://i.sstatic.net/qBhX6.png)
  - Why they choose that CV strategy.
  - Why only using data from 2011 to 2014.
  - How they've created the 5011 cohort used for the analysis.

- Cross-Entropy [is not a metric](https://sebastianraschka.com/faq/docs/proper-metric-cross-entropy.html). Metrics are functions that measures quality of the model prediction. E.g. It can be the case that the loss change but identify the same number of true labels.
