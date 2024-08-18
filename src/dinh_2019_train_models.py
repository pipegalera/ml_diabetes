from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

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

import pandas as pd
from scipy.stats import uniform, loguniform, randint
import xgboost as xgb
from utils import ConvertToCategory, MissingValueCategoryAs999
import sys
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

SEED = int(os.getenv("SEED"))
PROC_DATA_PATH = os.getenv("PROC_DATA_PATH")


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
    'estimator': [SVC(random_state=SEED)],
    'estimator__C': uniform(0.1, 5),
    'estimator__gamma': loguniform(1e-3, 1),
    'estimator__kernel': ['linear']
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


def run_model(X_train, X_test,
              y_train, y_test,
              model_pipeline):

    # Fit model
    model_pipeline.fit(X_train, y_train)

    # Predict
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba).round(3)
    precision = precision_score(y_test, y_pred).round(3)
    recall = recall_score(y_test, y_pred).round(3)
    f1 = f1_score(y_test, y_pred).round(3)

    # create a table
    metrics = {
        'Case': y_train.reset_index().columns[1],
        'Model': model_pipeline.param_distributions['estimator'],
        'AUC': [auc],
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1]
    }

    return pd.DataFrame(metrics)


def main(data):
    table_metrics = pd.DataFrame()
    models = [logistic_regression, random_forest]
    targets = ['Diabetes_Case_I', 'Diabetes_Case_II', 'CVD']

    for target in targets:
        for model in models:
            X_train, X_test, y_train, y_test = stratified_split(df, target)
            pipeline = model_pipeline(X_train, y_train, model)
            metrics = run_model(X_train, X_test, y_train, y_test, pipeline)
            table_metrics = pd.concat([table_metrics, metrics], ignore_index=True)

    print(table_metrics)


if __name__ == "__main__":
    main(df)
