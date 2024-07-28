import pandas as pd
import numpy as np
import os
import re
from glob import glob
from sklearn.base import BaseEstimator, TransformerMixin

RAW_DATA_PATH = "/root/dev/ml_diabetes/data/raw_data/NHANES/"
PROC_DATA_PATH = "/root/dev/ml_diabetes/data/processed/NHANES/"
SEED = 4208


def compile_data(variable_list,
                RAW_DATA_PATH=RAW_DATA_PATH,
                PROC_DATA_PATH=PROC_DATA_PATH,
                save_file_as=False,
                print_statemets=True):
    """
    for var in variable_list that you want:
        - Look in the docs what parquet files contain that variable
        - Make a list called parquet_files that contains the path of the files.
        for file in parquet_files:
            - Read that file specific column variable + SEQN
            - Concat to the "master" dataframe
            - Save the file if choosen

    """
    docs_df = pd.read_csv(PROC_DATA_PATH + "documentation_variables.csv")
    docs_df = docs_df[docs_df["Use Constraints"] != "RDC Only"]

    # Initial dataset just with all the individual indexes and its year
    master_df = pd.DataFrame()

    file_path = sorted(glob(RAW_DATA_PATH + "**" + "/*DEMO*.parquet", recursive=True))
    for file in file_path:
        df = pd.read_parquet(file, columns=["SEQN"])
        df["YEAR"] = re.search(r"\d{4}-\d{4}", file).group()
        master_df = pd.concat([master_df, df], ignore_index=True)
    master_df.sort_values(by=["SEQN", "YEAR"], inplace=True)


    for var in variable_list:
        if print_statemets:
            print(f"Searching for variable {var} ...")
        parquet_files = sorted(docs_df[docs_df["Variable Name"] == var]['Data File Name'].unique())

        variable_concat_df = pd.DataFrame()
        for file in parquet_files:
            pattern = os.path.join(RAW_DATA_PATH, "**", f"{file.upper()}.parquet")
            file_path = glob(pattern, recursive=True)
            file_path = ''.join(file_path)

            if file_path:
                df = pd.read_parquet(file_path, columns=["SEQN", var.upper()])
                df = df.rename({df.columns[1]: var}, axis=1) # Because bad formatting of raw NHANES data for MCQ300c
                variable_concat_df = pd.concat([variable_concat_df, df], ignore_index=True)
                if print_statemets:
                    print(f"--> Successfully added: {var} from {file_path}")

        master_df = master_df.merge(variable_concat_df, on = ["SEQN"], how="left")

    if save_file_as:
        master_df.to_parquet(RAW_DATA_PATH + f"{save_file_as}.parquet", index=False)
        print("File saved in the following folder: ", RAW_DATA_PATH)


    return master_df

def create_intake_new_column(df, day0_col, day1_col, day2_col):
    return np.where(df[day0_col].isna(),
                    df[[day1_col, day2_col]].mean(axis=1, skipna=True),
                    df[day0_col])

class ConvertToCategory(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].astype('category')
        return X

class MissingValueCategoryAs999(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].cat.add_categories(999).fillna(999)
        return X
