from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.stats import uniform, randint
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from utils import ConvertToCategory, MissingValueCategoryAs999,
from utils import PROC_DATA_PATH,SEED

df = pd.read_csv(PROC_DATA_PATH + "Dinh_2019_clean_data.csv")

target = 'Diabetes_Case_I'

y = df[target]
X = df.drop(columns= ['Diabetes_Case_I',
                      'Diabetes_Case_II',
                      'CVD',
                      'SEQN', 'Survey_year'])

strata = (df['Diabetes_Case_I'].astype(str) +
          '_' + df['Diabetes_Case_II'].astype(str)
          + '_' + df['CVD'].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=SEED,
                                                    stratify=strata)

del strata



# Categorical variables
categorical_vars = [ 'Race_ethnicity', 'General_health', 'Health_status', 'Told_High_Cholesterol', 'Household_income', 'Relative_Had_Diabetes']

# Numerical variables
numerical_vars = [ 'Age', 'Alcohol_consumption', 'Arm_circumference', 'Arm_length', 'Osmolality', 'Blood_urea_nitrogen', 'Body_mass_index', 'Chloride', 'Sodium', 'Gamma_glutamyl_transferase', 'Height', 'LDL_cholesterol', 'Leg_length', 'Lymphocytes', 'Mean_cell_volume', 'Pulse', 'Self_reported_greatest_weight', 'Total_cholesterol', 'Triglycerides', 'Waist_circumference', 'Weight', 'White_blood_cell_count', 'Aspartate_aminotransferase_AST', 'Alcohol_Intake', 'Caffeine_Intake', 'Calcium_Intake', 'Carbohydrate_Intake', 'Fiber_Intake', 'Kcal_Intake', 'Sodium_Intake', 'HDL_Cholesterol', 'Diastolic_Blood_Pressure', 'Systolic_Blood_Pressure']


def run_ml_pipeline(data):


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

param_dist_logistic = {
    'estimator': [LogisticRegression(random_state=SEED)],
    'estimator__C': uniform(0.1, 10),
    'estimator__penalty': ['l1', 'l2'],
    'estimator__solver': ['liblinear', 'saga']
}

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('estimator', param_dist_logistic)
])

CV_pipeline = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist_logistic,
    n_iter=50,
    scoring='roc_auc',
    random_state=SEED,
    n_jobs=-1
)

if __name__ == "__main__":