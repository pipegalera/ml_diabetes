import pandas as pd
import numpy as np

def get_raw_nhanes(year, DATA_PATH = "data/NHANES/raw_data/"):

  FILES = ["BMX", "BPX", "DEMO", "GHB", "GLU", "TRIGLY", "DIQ"]

  year_code_dict = {'2011-2012':'G',
                    '2013-2014':'H'}

  # Appending all the data
  df = pd.DataFrame(columns=['SEQN'])
  for file in FILES:
        df_part = pd.read_sas(DATA_PATH + f"{year}/{file}_{year_code_dict[year]}.XPT")
        df = df.merge(df_part, how="outer", on="SEQN")
        df['Year'] = year

  return df

def clean_nhanes(nhanes_raw_data):

  df = nhanes_raw_data.copy()

  # New columns with last reading available
  df["BPXDIX"] = np.where(df["BPXDI4"].isna(), df["BPXDI3"], df["BPXDI4"])
  df["BPXSYX"] = np.where(df["BPXSY4"].isna(), df["BPXSY3"], df["BPXSY4"])

  # Filter data
  relevant_columns = ["Year", "SEQN","RIDAGEYR","RIAGENDR",
                      "BMXWT","BMXHT","BMXBMI","BMXWAIST",
                      "BPXDIX","BPXSYX",
                      "BPXPLS", "LBDTRSI","LBDLDL",
                      "LBXGH", "LBDGLUSI", "DIQ010",]

  df = df[relevant_columns]

  # Replace
  df['DIQ010'] = df['DIQ010'].replace({1:"Yes",
                                       2:"No",
                                       3:"Borderline",
                                       7:"Refused",
                                       9:"Don't know",
                                      })

  df['RIAGENDR'] = df['RIAGENDR'].replace({1:"Male",
                                           2:"Female",
                                          })

  rename_cols_dict = {
      "SEQN":"Person ID",
      "RIDAGEYR":"Age",
      "RIAGENDR":"Gender",
      "BMXWT":"Weight",
      "BMXHT":"Height",
      "BMXBMI":"BMI",
      "BMXWAIST":"Waist",
      "BPXDIX":"Diastolic Blood pressure",
      "BPXSYX":"Systolic Blood pressure",
      "BPXPLS":"Pulse",
      "LBDTRSI":"Triglyceride",
      "LBDLDL":"LDL-Cholesterol",
      "LBXGH":"Glycohemoglobin",
      "LBDGLUSI":"Glucose",
      "DIQ010":"Diagnosed Diabetes",
  }

  df = df.rename(rename_cols_dict, axis=1)

  return df
