from tqdm import tqdm
import requests
import os

year_code_dict = {'1999-2000': '',
                  '2001-2002': '_B',
                  '2003-2004': '_C',
                  '2005-2006': '_D',
                  '2007-2008': '_E',
                  '2009-2010': '_F',
                  '2011-2012': '_G',
                  '2013-2014': '_H',
                  '2015-2016': '_I',
                  '2017-2018': '_J',
                  '2019-2020': '_K'
                 }

PATH = "/Users/pipegalera/dev/ml_diabetes/data/NHANES/raw_data"
FILES = ["BMX", "BPX", "DEMO", "GHB", "GLU", "TRIGLY", "DIQ", "CDQ"]

for file in FILES:
    for key,value in year_code_dict.items():
        url = f"https://wwwn.cdc.gov/Nchs/Nhanes/{key}/{file}{value}.XPT"
        resp = requests.get(url, stream=True)
        file_name = url.split('/')[-2] + '_' + url.split('/')[-1]
        save_path = f"{PATH}/{file_name}"
        if os.path.exists(save_path):
            print(f"{file_name} file already in the destination folder")
        else:
            with open(save_path, "wb") as f_out:
                for data in tqdm(resp.iter_content(),
                                desc=f"{key} -> {file}",
                                postfix=f"saved in {save_path}",
                                total=int(resp.headers["Content-Length"])):
                    f_out.write(data)
