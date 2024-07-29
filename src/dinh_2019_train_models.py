import pandas as pd
from scipy.stats import uniform, loguniform, randint

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

from utils import ConvertToCategory, MissingValueCategoryAs999
from utils import PROC_DATA_PATH,SEED

import mlflow
import mlflow.sklearn

# Uncomment when developing in local
#PROC_DATA_PATH = '/Users/pipegalera/dev/ml_diabetes/data/processed/NHANES/'

df = pd.read_csv(PROC_DATA_PATH + "Dinh_2019_clean_data.csv")

def split(df, target:str):

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

    train_target_pcr = round(y_train.reset_index()[target].sum()/len(y_train),5)
    test_target_pcr = round(y_test.reset_index()[target].sum()/len(y_test),5)

    if train_target_pcr == test_target_pcr:
        return X_train, X_test, y_train, y_test
    else:
        print("Test and Train set have different target proportions")


logistic_regression_params = {
        'estimator': [LogisticRegression(random_state=SEED)],
        'estimator__C': uniform(0.1, 10),
        'estimator__penalty': ['l1', 'l2'],
        'estimator__solver': ['liblinear', 'saga'],
    }

param_dist_svc = {
    'estimator': [SVC(random_state=SEED)],
    'estimator__C': uniform(0.1, 5),
    'estimator__gamma': loguniform(1e-3, 1),
    'estimator__kernel': ['rbf', 'linear']
}

def model_pipeline(X_train, y_train, model_params):
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
        ('estimator', model_params)
    ])

    grid = RandomizedSearchCV(
        pipeline,
        param_distributions=model_params,
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

    # Start MLflow run
    mlflow.set_experiment("Dinh et al. 2019")
    with mlflow.start_run():
        # Fit model
        model_pipeline.fit(X_train, y_train)

        # Predict
        y_pred = model_pipeline.predict(X_test)
        y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

        # Accuracy Metrics
        best_score = model_pipeline.best_score_
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log model parameters
        mlflow.log_params(model_pipeline.best_params_)
        mlflow.sklearn.log_model(model_pipeline.best_estimator_, "model")
        mlflow.set_tag("Target", target)
        mlflow.log_metric("Train ROC-AUC", best_score)
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1-score", f1)

if __name__ == "__main__":
    target = 'Diabetes_Case_I'
    model = param_dist_svc
    X_train, X_test, y_train, y_test = split(df, target)
    pipeline = model_pipeline(X_train, y_train, model)
    run_model(X_train, X_test, y_train, y_test, pipeline)
