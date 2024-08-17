#python python "/root/dev/ml_diabetes/paper_replication/Dinh et al. (2019)/dinh_2019.py"

from utils import create_intake_new_column, compile_data
import pandas as pd
import numpy as np
from glob import glob
from dotenv import load_dotenv

load_dotenv()

PROC_DATA_PATH = os.getenv("PROC_DATA_PATH")


def preprocessing_nhanes(data):
    df = data.copy()

    # These are numerical variables that have coded categorical answers "Don't know" and "Refused" as numerical
    df['ALQ130'] = df['ALQ130'].replace([77, 99, 777, 999], np.nan)
    df['WHD140'] = df['WHD140'].replace([7777, 77777, 9999, 99999], np.nan)

    # Create new columns
    df['Alcohol_Intake'] = create_intake_new_column(df, 'DRXTALCO', 'DR1TALCO', 'DR2TALCO')
    df['Caffeine_Intake'] = create_intake_new_column(df, 'DRXTCAFF', 'DR1TCAFF', 'DR2TCAFF')
    df['Calcium_Intake'] = create_intake_new_column(df, 'DRXTCALC', 'DR1TCALC', 'DR2TCALC')
    df['Carbohydrate_Intake'] = create_intake_new_column(df, 'DRXTCARB', 'DR1TCARB', 'DR2TCARB')
    df['Fiber_Intake'] = create_intake_new_column(df, 'DRXTFIBE', 'DR1TFIBE', 'DR2TFIBE')
    df['Kcal_Intake'] = create_intake_new_column(df, 'DRXTKCAL', 'DR1TKCAL', 'DR2TKCAL')
    df['Sodium_Intake'] = create_intake_new_column(df, 'DRDTSODI', 'DR1TSODI', 'DR2TSODI')

    # Combine same variables
    df['Relative_Had_Diabetes'] = df['MCQ250A'].combine_first(df['MCQ300C']).combine_first(df['MCQ300c'])
    df['Told_CHF'] = df['MCQ160B'].combine_first(df['MCQ160b'])
    df['Told_CHD'] = df['MCQ160C'].combine_first(df['MCQ160c'])
    df['Told_HA'] = df['MCQ160E'].combine_first(df['MCQ160e'])
    df['Told_stroke'] = df['MCQ160F'].combine_first(df['MCQ160f'])
    df['Pregnant'] = df['SEQ060'].combine_first(df['RHQ141']).combine_first(df['RHD143'])
    df['HDL_Cholesterol'] = df['LBDHDLSI'].combine_first(df['LBDHDDSI'])
    df['Glucose'] = df['LBXGLUSI'].combine_first(df['LBDGLUSI'])
    # Choosing the last blood reading
    df['Diastolic_Blood_Pressure'] = df['BPXDI4'].combine_first(df['BPXDI3']).combine_first(df['BPXDI2']).combine_first(df['BPXDI1'])
    df['Systolic_Blood_Pressure'] = df['BPXSY4'].combine_first(df['BPXSY3']).combine_first(df['BPXSY2']).combine_first(df['BPXSY1'])

    # Filter data
    cond_1 = (df['Pregnant'].isna()) | (df['Pregnant'] != 1)
    cond_2 =(df['RIDAGEYR'] >= 20)
    df = df[cond_1 & cond_2]

    # Delete old columns that are not needed
    columns_to_drop = ['DRXTALCO', 'DR1TALCO', 'DR2TALCO', 'DRXTCAFF', 'DR1TCAFF', 'DR2TCAFF',
                    'DRXTCALC', 'DR1TCALC', 'DR2TCALC', 'DRXTCARB', 'DR1TCARB', 'DR2TCARB',
                    'DRXTFIBE', 'DR1TFIBE', 'DR2TFIBE', 'DRXTKCAL', 'DR1TKCAL', 'DR2TKCAL',
                    'DRDTSODI', 'DR1TSODI', 'DR2TSODI', 'MCQ250A', 'MCQ300C', 'MCQ300c',
                    'MCQ160B', 'MCQ160b', 'MCQ160C', 'MCQ160c', 'MCQ160E', 'MCQ160e',
                    'MCQ160F', 'MCQ160f', 'SEQ060', 'RHQ141', 'RHD143',
                    'LBDHDLSI', 'LBDHDDSI', 'LBXGLUSI', 'LBDGLUSI',
                    'BPXDI4', 'BPXDI3', 'BPXDI2', 'BPXDI1',
                    'BPXSY4', 'BPXSY3', 'BPXSY2', 'BPXSY1', 'Pregnant']

    df = df.drop(columns=columns_to_drop)

    return df


def create_targets(data):
    df = data.copy()

    df['Diabetes_Case_I'] = np.where(
      (df['Glucose'] > 7.0) | (df['DIQ010'] == 1), 1, 0)

    df['Diabetes_Case_II'] = np.where(
      (df['Diabetes_Case_I'] == 0) & (df['Glucose'] >= 5.6) & (df['Glucose'] < 7.0), 1, 0)

    df['CVD'] = np.where(
        (df['Told_CHF'] == 1) | (df['Told_CHD'] == 1) | (df['Told_HA'] == 1) | (df['Told_stroke'] == 1), 1, 0)

    # Drop the unnecessary columns
    df = df.drop(columns=['Told_CHF', 'Told_CHD', 'Told_HA', 'Told_stroke', 'Glucose', 'DIQ010'])

    return df

def rename_columns(data):

    df = data.copy()

    df = df.rename(columns={ 'ALQ130': 'Alcohol_consumption', 'BMXARMC': 'Arm_circumference', 'BMXARML': 'Arm_length', 'BMXBMI': 'Body_mass_index', 'BMXHT': 'Height', 'BMXLEG': 'Leg_length', 'BMXWAIST': 'Waist_circumference', 'BMXWT': 'Weight', 'BPQ080': 'Told_High_Cholesterol', 'BPXPLS': 'Pulse', 'HSD010': 'General_health', 'HUQ010': 'Health_status', 'INDHHIN2': 'Household_income', 'LBXSCLSI': 'Chloride', 'LBXSNASI': 'Sodium', 'LBDLDLSI': 'LDL_cholesterol', 'LBDLYMNO': 'Lymphocytes', 'LBDSBUSI': 'Blood_urea_nitrogen', 'LBDSTRSI': 'Triglycerides', 'LBDTCSI': 'Total_cholesterol', 'LBXMCVSI': 'Mean_cell_volume', 'LBXSASSI': 'Aspartate_aminotransferase_AST', 'LBXSGTSI': 'Gamma_glutamyl_transferase', 'LBXSOSSI': 'Osmolality', 'LBXWBCSI': 'White_blood_cell_count', 'RIDAGEYR': 'Age', 'RIDRETH1': 'Race_ethnicity', 'WHD140': 'Self_reported_greatest_weight', 'YEAR': 'Survey_year'
    })

    return df


def main():
    # Get the name of the variables we need from manual file
    dinh_2019_variables = pd.read_excel(PROC_DATA_PATH + "dinh_2019_variables_doc.xlsx")["NHANES Name"].unique()

    # Compile raw data into a unique raw dataframe
    df = compile_data(variable_list=dinh_2019_variables, print_statemets=False)
    # Clean raw data
    df = (df
             .pipe(preprocessing_nhanes)
             .pipe(create_targets)
             .pipe(rename_columns)
         )
    # Save file
    df.to_csv(PROC_DATA_PATH + "Dinh_2019_clean_data.csv", index=False)
    print(f"--> Clean data saved as {PROC_DATA_PATH + "Dinh_2019_clean_data.csv"}")

if __name__ == "__main__":
    main()
