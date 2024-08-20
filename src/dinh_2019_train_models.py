from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.kernel_approximation import Nystroem

import pandas as pd
import numpy as np
import pickle
import time
from scipy.stats import uniform, loguniform, randint
import xgboost as xgb
from utils import ConvertToCategory, MissingValueCategoryAs999, WeightedEnsemble, find_model_name_from_pipeline
import sys
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

SEED = int(os.getenv("SEED"))
PROC_DATA_PATH = os.getenv("PROC_DATA_PATH")
MODEL_RESULTS_PATH = os.getenv("MODEL_RESULTS_PATH")

df = pd.read_csv(PROC_DATA_PATH + "Dinh_2019_clean_data.csv")

## Downsample
df['strata'] = (df['Diabetes_Case_I'].astype(str) +
          '_' + df['Diabetes_Case_II'].astype(str) +
          '_' + df['CVD'].astype(str))
index_to_drop = df[df['strata'] == '0_0_0'].sample(frac=0.75).index
df = df.drop(index_to_drop).reset_index(drop=True)
df = df.drop('strata', axis=1)


def stratified_split(df, target:str):

    X = df.drop(columns= [
        'Diabetes_Case_I',
        'Diabetes_Case_II',
        'CVD',
        'SEQN',
        'Survey_year'])

    y = df[target]


    strata = (df['Diabetes_Case_I'].astype(str) +
              '_' + df['Diabetes_Case_II'].astype(str) +
              '_' + df['CVD'].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=SEED,
                                                        stratify=strata)

    train_target_pcr = round(y_train.reset_index()[target].sum()/len(y_train),3)
    test_target_pcr = round(y_test.reset_index()[target].sum()/len(y_test),3)

    if train_target_pcr != test_target_pcr:
        raise ValueError("Test and Train set have different target proportions")
    else:
        return X_train, X_test, y_train, y_test


logistic_regression = {
        'estimator': [LogisticRegression(random_state=SEED, max_iter=10_000)],
        'estimator__C': uniform(0.1, 10),
        'estimator__penalty': ['l1', 'l2'],
        'estimator__solver': ['liblinear', 'saga'],
    }

support_vector_machine = {
    'estimator': [Pipeline([
            ('nystroem', Nystroem(random_state=SEED)),
            ('svm', SVC(probability=True, random_state=SEED))
        ])],
    'estimator__nystroem__n_components': randint(10, 500),
    'estimator__nystroem__gamma': loguniform(1e-3, 1),
    'estimator__svm__C': loguniform(1e-3, 1e3)
}

random_forest = {
    'estimator': [RandomForestClassifier(random_state=SEED)],
    'estimator__n_estimators': randint(50, 200),
    'estimator__max_features': ['sqrt', 'log2'],
    'estimator__max_depth': randint(1, 10),
    'estimator__criterion': ['gini', 'entropy']
}

xgb = {
    'estimator': [xgb.XGBClassifier(random_state=SEED)],
    'estimator__n_estimators': randint(50, 200),
    'estimator__learning_rate': [0.01,0.05,0.1],
    'estimator__max_depth': randint(1, 10),
    'estimator__gamma': [0, 0.5, 1],
    'estimator__reg_alpha': [0, 0.5, 1],
    'estimator__reg_lambda': [0.5, 1, 5],
    'estimator__base_score': [0.2, 0.5, 1]
}

def model_pipeline(X_train, y_train, model):
    # Categorical variables
    categorical_vars = [ 'Race_ethnicity', 'General_health', 'Health_status', 'Told_High_Cholesterol', 'Household_income', 'Relative_Had_Diabetes']

    # Numerical variables
    numerical_vars = [ 'Age', 'Alcohol_consumption', 'Arm_circumference', 'Arm_length', 'Osmolality', 'Blood_urea_nitrogen', 'Body_mass_index', 'Chloride', 'Sodium', 'Gamma_glutamyl_transferase', 'Height', 'LDL_cholesterol', 'Leg_length', 'Lymphocytes', 'Mean_cell_volume', 'Pulse', 'Self_reported_greatest_weight', 'Total_cholesterol', 'Triglycerides', 'Waist_circumference', 'Weight', 'White_blood_cell_count', 'Aspartate_aminotransferase_AST', 'Alcohol_Intake', 'Caffeine_Intake', 'Calcium_Intake', 'Carbohydrate_Intake', 'Fiber_Intake', 'Kcal_Intake', 'Sodium_Intake', 'HDL_Cholesterol', 'Diastolic_Blood_Pressure', 'Systolic_Blood_Pressure']

    categorical_pipeline = Pipeline([
        ('convert_to_cat', ConvertToCategory(categorical_vars)),
        ('add_unknown_cat', MissingValueCategoryAs999(categorical_vars)),
    ])

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ('categorical', categorical_pipeline, categorical_vars),
        ('numerical', numerical_pipeline, numerical_vars)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('estimator', model)
    ])

    grid = RandomizedSearchCV(
        pipeline,
        param_distributions=model,
        n_iter=50,
        scoring='roc_auc',
        random_state=SEED,
        n_jobs=-1,
        cv=10,
    )
    return grid


def calculate_metrics(y_true, y_pred, y_pred_proba):
    return {
        'AUC': [roc_auc_score(y_true, y_pred_proba).round(3)],
        'Precision': [precision_score(y_true, y_pred).round(3)],
        'Recall': [recall_score(y_true, y_pred).round(3)],
        'F1': [f1_score(y_true, y_pred).round(3)],
    }


def run_models(X_train, X_test,
              y_train, y_test,
              model_pipelines):

    auc_scores = []
    fitted_models = []
    all_metrics = []

    case  = y_train.reset_index().columns[1]

    # Fit models
    for pipeline in model_pipelines:

        model_name = find_model_name_from_pipeline(pipeline.param_distributions)

        print(f"Training model: {model_name} predicting {case}...")
        start_time = time.time()

        pipeline.fit(X_train, y_train)

        # Save models
        filename = f'dinh_model_{case}_{model_name}.pkl'
        with open(MODEL_RESULTS_PATH + filename, 'wb') as file:
            pickle.dump(pipeline, file)

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Model {model_name} training time: {training_time:.2f} seconds")

        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        auc_scores.append(metrics['AUC'][0])
        fitted_models.append(pipeline)

        all_metrics.append({
            'Case': case,
            'Model': model_name,
            'Training Time (seconds)': f'{training_time:.2f}',
            **metrics
        })

    print(f"Creating AUC Weighted Ensemble...")
    # Normalize AUC scores to use as weights
    weights = np.array(auc_scores) / sum(auc_scores)

    # Create auc weighted ensemble
    ensemble = WeightedEnsemble(fitted_models, weights)

    # Stack the ensemble model prediction to the table
    y_pred = ensemble.predict(X_test)
    y_pred_proba = ensemble.predict_proba(X_test)

    ensemble_metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    all_metrics.append({
        'Case': case,
        'Model': 'AUC Weighted Ensemble',
        'Training Time (seconds)': 'Training not needed',
        **ensemble_metrics
    })

    # Create a table with all metrics
    return pd.DataFrame(all_metrics)


def main():
    targets = ['Diabetes_Case_I', 'Diabetes_Case_II', 'CVD']
    table_metrics = pd.DataFrame()

    for target in targets:
        X_train, X_test, y_train, y_test = stratified_split(df, target)
        pipeline_logistic_regression = model_pipeline(X_train, y_train, logistic_regression)
        #pipeline_svm = model_pipeline(X_train, y_train, support_vector_machine)
        pipeline_random_forest = model_pipeline(X_train, y_train, random_forest)
        pipeline_xgb = model_pipeline(X_train, y_train, xgb)

        model_pipelines = [
                           pipeline_logistic_regression,
                           #pipeline_svm,
                           pipeline_random_forest,
                           pipeline_xgb]

        metrics = run_models(X_train, X_test,
                             y_train, y_test,
                             model_pipelines)

        table_metrics = pd.concat([table_metrics, metrics], ignore_index=True)

    table_metrics.to_csv(MODEL_RESULTS_PATH + "dinh_2019_results.csv", index=False)
    print(table_metrics)


if __name__ == "__main__":
    main()
