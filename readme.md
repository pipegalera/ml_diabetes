
🚧🚧🚧 --- Work In Progress --- 🚧🚧🚧

# Deep Neural Nets & Gradient Boosted Trees for Diabetes Prediction

The purpose of this experiment is:

1) Replicating the scarce literature on the topic.

2) Compare the ability of non-parametric models predicting Diabetes based on survey and lab public medical data from different sources:

- [NHANES](https://www.cdc.gov/nchs/index.htm). US Population.
- [DRYAD](https://doi.org/10.5061/dryad.ft8750v). China Population.

## Literature on the topic

- Dinh et al. (2019): [A data-driven approach to predicting diabetes and cardiovascular disease with machine learning](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-019-0918-5)
- Cahn et al. (2020): [Prediction of progression from pre-diabetes to diabetes: Development and validation of a machine learning model](https://onlinelibrary.wiley.com/doi/10.1002/dmrr.3252)
- Wu et al. (2021): [Machine Learning for Predicting the 3-Year Risk of Incident Diabetes in Chinese Adults](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2021.626331/full)
- Matabuena et al. (2024): [Deep Learning Framework with Uncertainty Quantification for Survey Data: Assessing and Predicting Diabetes Mellitus Risk in the American Population
](https://arxiv.org/abs/2403.19752)

🚧 Preliminary notes:

- What blood pressure readings are taking (1rst, 2nd, 3rd) ?

## 1. Replication

### Data

To download `NHANES` data, I've created `nhanes_data_downloader.py`.:

![nhanes_data_downloader.py](https://raw.githubusercontent.com/pipegalera/ml_diabetes/main/images/NHANES_downloader.png)

The selection of the variables and time-frames used to predict diabetes depend of the paper (see more in each paper folder).

### Diabestes definition

Even that the definition of the label "Diabetes" slightly change in the papers, they is consistent with the
[American Diabetes Association (see Table 3)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2797383/table/T3/).

## 2. Model Comparison

🚧 Preliminary notes:

- Need a basic standarized methodology to ensure that Deep Learning and Machine Learning methods are on a somewhat level playing field - E.g.NNs performace bastly depend on number of layers and epochs trained on.

It cannot be compared a deep NN trained for days with a 5-min trainned default XGB. It sounds obvious but it can be the case as the papers do not specify how train the models or for how long. The seeds are a mystery. Maybe the reviwers at the time were not familiar with Neural Networks.

### 2.1 Summary of Models to compare

*TBD - Incomplete*

Bechmarks:

1. Logistic Regression (simple para).
2. 32-16-8 Neural Network (simple non-para).

Tree Models:

3. CatBoost
4. XGBoost

Deep Learning Models:

5. TabPFN
6. TabNet

Ensemble Learning:

7. Ensemble

### 2.2 Tree-based Models

#### CatBoost

- Paper: [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)
- Code: https://github.com/catboost/catboost

#### XGBoost

- Paper: [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- Code: https://xgboost.readthedocs.io/en/stable/

### 2.2 Deep Learning Models

#### TabPFN

- Paper: [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848)
- Code: https://github.com/automl/TabPFN

"it doesn’t require training, hyperparameter tuning, or cross-validation – it only requires a single forward pass on a new training set. The caveat is that it requires synthetic datasets for the prior." (https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html)

#### TabNet

- Paper: [TabNet: Attentive Interpretable Tabular Learning
](https://arxiv.org/abs/1908.07442)
- Code: https://github.com/dreamquark-ai/tabnet
- Example: https://github.com/google-research/google-research/tree/master/tabnet
