# Dinh et al. (2019)

## A data-driven approach to predicting diabetes and cardiovascular disease with machine learning

URL: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-019-0918-5


## Brief Summary

Dinh et al. (2019) uses different ML models (logistic regression, support vector machines, random forest, and gradient boosting) on NHANES dataset to predict i) Diabetes and ii) Cardiovascular disease ("CVD").

**Goal**: Identification mechanism for patients at risk of diabetes and cardiovascular diseases and key contributors to diabetes .

**Results**:

Best scores:

- CVB prediction based on 131 NHANES variables achieved an AU-ROC score of 83.9% .
- Diabetes prediction based on 123 NHANES variables achieved an AU-ROC score of 95.7% .
- Pre-diabetic prediction based on 123 NHANES variables achieved an AU-ROC score of 84.4% .
- Top 5 predictors in diabetes patients were 1) `waist size`, 2) `age`, 3) `self-reported weight`, 4) `leg length`, 5) `sodium intake`.



This notebook replicates the results of the paper. The structure follows the following steps: 

1. NHANES data 
2. Pre-processing of the data
3. Transformation of the data
4. Train/Test Split 
5. CV 10-fold
6. Training monitoring using MLflow
7. Get metric results (AUC)


The structure of the analysis emulates the Figure 1 from the paper: 

![Fig 1 from Dinh et al. 2019](https://raw.githubusercontent.com/pipegalera/ml_diabetes/main/images/dinh_2019_Fig1.png)



```R
library(arrow)
library(dplyr)
library(readxl)
```

## 1. HNANES data

### Covariates and Targets 

- Source: https://www.cdc.gov/nchs/index.htm
- Downloaded raw data via: `notebooksnhanes_data_backfill`


The paper did not mention what variables they use from NHANES. I emailed the author in the correspondence section of the paper to try to get the list of variables they used, but no answer from him yet.

Please notice that NHANES have more than 3900 variables, therefore without the list of the specific variables used it is impossible to fully replicate the paper.

For now, I will consider the variables taken from [Figure 5](https://raw.githubusercontent.com/pipegalera/ml_diabetes/main/images/dinh_2019_Fig5.png) and [Figure 6](https://raw.githubusercontent.com/pipegalera/ml_diabetes/main/images/dinh_2019_Fig6.png) of the paper. I compiled them by hand in an Excel file using NHANES search tool for variables (see: `processed/NHANES/dinh_2019_variables_doc.xlsx`).


- `Case I: Diabetes`

    - Glucose >= 126 mg/dL. OR;
    - "Yes" to the question "Have you ever been told by a doctor that you have diabetes?"

- `Case II: Undiagnosed Diabetes`

    - Glucose >= 126 mg/dL. AND;
    - "No" to the question "Have you ever been told by a doctor that you have diabetes?"

- `Cardiovascular disease`

    - "Yes" to any of the the questions "Have you ever been told by a doctor that you had congestive heart failure, coronary heart disease, a heart attack, or a stroke?"

- `Pre diabetes`

    - Glucose 125 >= 100 mg/dL


```R
DINH_DOCS_PATH <- "/Users/pipegalera/dev/ml_diabetes/data/processed/NHANES/"
dinh_2019_vars <- read_excel(paste0(DINH_DOCS_PATH, "dinh_2019_variables_doc.xlsx"))

head(dinh_2019_vars[c("Variable Name", "NHANES Name")], n=15)

```


<table class="dataframe">
<caption>A tibble: 15 x 2</caption>
<thead>
	<tr><th scope=col>Variable Name</th><th scope=col>NHANES Name</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Age                          </td><td>RIDAGEYR</td></tr>
	<tr><td>Alcohol consumption          </td><td>ALQ130  </td></tr>
	<tr><td>Alcohol intake               </td><td>DRXTALCO</td></tr>
	<tr><td>Alcohol intake, First Day    </td><td>DR1TALCO</td></tr>
	<tr><td>Alcohol intake, Second Day   </td><td>DR2TALCO</td></tr>
	<tr><td>Arm circumference            </td><td>BMXARMC </td></tr>
	<tr><td>Arm length                   </td><td>BMXARML </td></tr>
	<tr><td>Blood osmolality             </td><td>LBXSOSSI</td></tr>
	<tr><td>Blood relatives have diabetes</td><td>MCQ250A </td></tr>
	<tr><td>Blood urea nitrogen          </td><td>LBDSBUSI</td></tr>
	<tr><td>BMI                          </td><td>BMXBMI  </td></tr>
	<tr><td>Caffeine intake              </td><td>DRXTCAFF</td></tr>
	<tr><td>Caffeine intake, First Day   </td><td>DR1TCAFF</td></tr>
	<tr><td>Caffeine intake, Second Day  </td><td>DR2TCAFF</td></tr>
	<tr><td>Calcium intake, First Day    </td><td>DR1TCALC</td></tr>
</tbody>
</table>



For the complete list of variables, check the file `dinh_2019_variables_doc.xlsx` under NHANES data folder.

NHANES data is made by multiple files (see `NHANES` unde data folder) that have to be compiled together. The data was downloaded automatically via script, all the files converted from SAS to parquet, and the files were stacked and merged based on the individual index ("SEQN"). For more details please check the `nhanes_data_backfill` notebook. 

Plese notice that no transformation are made to the covariates, the files were only arranged and stacked together. 


```R
DATA_PATH  <- "/Users/pipegalera/dev/ml_diabetes/data/raw_data/NHANES/"
df <- read_parquet(paste0(DATA_PATH, "dinh_raw_data.parquet"))
```


```R
head(df)
```


<table class="dataframe">
<caption>A tibble: 6 x 77</caption>
<thead>
	<tr><th scope=col>SEQN</th><th scope=col>YEAR</th><th scope=col>RIDAGEYR</th><th scope=col>ALQ130</th><th scope=col>DRXTALCO</th><th scope=col>DR1TALCO</th><th scope=col>DR2TALCO</th><th scope=col>BMXARMC</th><th scope=col>BMXARML</th><th scope=col>LBXSOSSI</th><th scope=col>...</th><th scope=col>SEQ060</th><th scope=col>DIQ010</th><th scope=col>MCQ160B</th><th scope=col>MCQ160b</th><th scope=col>MCQ160C</th><th scope=col>MCQ160c</th><th scope=col>MCQ160E</th><th scope=col>MCQ160e</th><th scope=col>MCQ160F</th><th scope=col>MCQ160f</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>...</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>1999-2000</td><td> 2</td><td>NA</td><td> 0.00</td><td>NA</td><td>NA</td><td>15.2</td><td>18.6</td><td> NA</td><td>...</td><td>NA</td><td>2</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr>
	<tr><td>2</td><td>1999-2000</td><td>77</td><td> 1</td><td> 0.00</td><td>NA</td><td>NA</td><td>29.8</td><td>38.2</td><td>288</td><td>...</td><td>NA</td><td>2</td><td> 2</td><td>NA</td><td> 2</td><td>NA</td><td> 2</td><td>NA</td><td> 2</td><td>NA</td></tr>
	<tr><td>3</td><td>1999-2000</td><td>10</td><td>NA</td><td> 0.00</td><td>NA</td><td>NA</td><td>19.7</td><td>25.5</td><td> NA</td><td>...</td><td>NA</td><td>2</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr>
	<tr><td>4</td><td>1999-2000</td><td> 1</td><td>NA</td><td> 0.00</td><td>NA</td><td>NA</td><td>16.4</td><td>20.4</td><td> NA</td><td>...</td><td>NA</td><td>2</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr>
	<tr><td>5</td><td>1999-2000</td><td>49</td><td> 3</td><td>34.56</td><td>NA</td><td>NA</td><td>35.8</td><td>39.7</td><td>276</td><td>...</td><td>NA</td><td>2</td><td> 2</td><td>NA</td><td> 2</td><td>NA</td><td> 2</td><td>NA</td><td> 2</td><td>NA</td></tr>
	<tr><td>6</td><td>1999-2000</td><td>19</td><td>NA</td><td> 0.00</td><td>NA</td><td>NA</td><td>26.0</td><td>34.5</td><td>277</td><td>...</td><td> 2</td><td>2</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr>
</tbody>
</table>




```R
tail(df)
```


<table class="dataframe">
<caption>A tibble: 6 x 77</caption>
<thead>
	<tr><th scope=col>SEQN</th><th scope=col>YEAR</th><th scope=col>RIDAGEYR</th><th scope=col>ALQ130</th><th scope=col>DRXTALCO</th><th scope=col>DR1TALCO</th><th scope=col>DR2TALCO</th><th scope=col>BMXARMC</th><th scope=col>BMXARML</th><th scope=col>LBXSOSSI</th><th scope=col>...</th><th scope=col>SEQ060</th><th scope=col>DIQ010</th><th scope=col>MCQ160B</th><th scope=col>MCQ160b</th><th scope=col>MCQ160C</th><th scope=col>MCQ160c</th><th scope=col>MCQ160E</th><th scope=col>MCQ160e</th><th scope=col>MCQ160F</th><th scope=col>MCQ160f</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>...</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>83726</td><td>2013-2014</td><td>40</td><td>NA</td><td>NA</td><td>NA</td><td>  NA</td><td>31.0</td><td>39.0</td><td> NA</td><td>...</td><td>NA</td><td>2</td><td>NA</td><td> 2</td><td>NA</td><td> 2</td><td>NA</td><td> 2</td><td>NA</td><td> 2</td></tr>
	<tr><td>83727</td><td>2013-2014</td><td>26</td><td> 3</td><td>NA</td><td>14</td><td>19.9</td><td>29.9</td><td>35.2</td><td>285</td><td>...</td><td>NA</td><td>2</td><td>NA</td><td> 2</td><td>NA</td><td> 2</td><td>NA</td><td> 2</td><td>NA</td><td> 2</td></tr>
	<tr><td>83728</td><td>2013-2014</td><td> 2</td><td>NA</td><td>NA</td><td> 0</td><td> 0.0</td><td>14.7</td><td>16.5</td><td> NA</td><td>...</td><td>NA</td><td>2</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr>
	<tr><td>83729</td><td>2013-2014</td><td>42</td><td>NA</td><td>NA</td><td> 0</td><td> 0.0</td><td>37.0</td><td>37.6</td><td>277</td><td>...</td><td>NA</td><td>2</td><td>NA</td><td> 2</td><td>NA</td><td> 2</td><td>NA</td><td> 2</td><td>NA</td><td> 2</td></tr>
	<tr><td>83730</td><td>2013-2014</td><td> 7</td><td>NA</td><td>NA</td><td>NA</td><td>  NA</td><td>19.0</td><td>26.0</td><td> NA</td><td>...</td><td>NA</td><td>2</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr>
	<tr><td>83731</td><td>2013-2014</td><td>11</td><td>NA</td><td>NA</td><td> 0</td><td> 0.0</td><td>25.0</td><td>31.7</td><td> NA</td><td>...</td><td>NA</td><td>2</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr>
</tbody>
</table>




```R
nrow(df)
```


82091



```R
colnames(df)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'SEQN'</li><li>'YEAR'</li><li>'RIDAGEYR'</li><li>'ALQ130'</li><li>'DRXTALCO'</li><li>'DR1TALCO'</li><li>'DR2TALCO'</li><li>'BMXARMC'</li><li>'BMXARML'</li><li>'LBXSOSSI'</li><li>'MCQ250A'</li><li>'LBDSBUSI'</li><li>'BMXBMI'</li><li>'DRXTCAFF'</li><li>'DR1TCAFF'</li><li>'DR2TCAFF'</li><li>'DR1TCALC'</li><li>'DR2TCALC'</li><li>'DRXTCALC'</li><li>'DR1TCARB'</li><li>'DR2TCARB'</li><li>'DRXTCARB'</li><li>'LB2SCLSI'</li><li>'MCQ300c'</li><li>'MCQ300C'</li><li>'BPXDI1'</li><li>'BPXDI4'</li><li>'BPXDI2'</li><li>'BPXDI3'</li><li>'RIDRETH1'</li><li>'DR1TFIBE'</li><li>'DR2TFIBE'</li><li>'DRXTFIBE'</li><li>'LBXSGTSI'</li><li>'HSD010'</li><li>'HUQ010'</li><li>'LBDHDLSI'</li><li>'LBDHDDSI'</li><li>'BMXHT'</li><li>'BPQ080'</li><li>'INDHHIN2'</li><li>'DRXTKCAL'</li><li>'DR1TKCAL'</li><li>'DR2TKCAL'</li><li>'LBDLDLSI'</li><li>'BMXLEG'</li><li>'LBDLYMNO'</li><li>'LBXMCVSI'</li><li>'BPXPLS'</li><li>'WHD140'</li><li>'DR1TSODI'</li><li>'DR2TSODI'</li><li>'DRDTSODI'</li><li>'BPXSY1'</li><li>'BPXSY4'</li><li>'BPXSY2'</li><li>'BPXSY3'</li><li>'LBDTCSI'</li><li>'LBDSTRSI'</li><li>'BMXWAIST'</li><li>'BMXWT'</li><li>'LBXWBCSI'</li><li>'LBXSASSI'</li><li>'LBXGLUSI'</li><li>'LBDGLUSI'</li><li>'RHQ141'</li><li>'RHD143'</li><li>'SEQ060'</li><li>'DIQ010'</li><li>'MCQ160B'</li><li>'MCQ160b'</li><li>'MCQ160C'</li><li>'MCQ160c'</li><li>'MCQ160E'</li><li>'MCQ160e'</li><li>'MCQ160F'</li><li>'MCQ160f'</li></ol>



## 2. Pre-processing and Data modeling


### 2.1 Extreme values and replacing Missing/Don't know answers

> The preprocessing stage also converted any undecipherable values (errors in datatypes and standard formatting) from the database to null representations.

For this, I've checked the variables according to their possible values in the NHANES documentation (https://wwwn.cdc.gov/nchs/nhanes/search/default.aspx). I did not found any any extreme value out of the possible ranges. However, the data is reviwed and updated after the survey, so it might be that the NCHS applied some fixes after they saw them. 


I have replaced "Don't know" and "Refused" for NA values and converted the intial encoding of the categorical variables to the real values in the survey - given that the encoding is not consistent accross years. For the model, I will encode the variables myself so I don't have to jungle NHANES encoding. 

All the variables can by found at  https://wwwn.cdc.gov/nchs/nhanes/search/default.aspx


```R
# Refused or Don"t know for NA
df_formatted <- df %>%
  mutate(
    ALQ130 = case_when(
      ALQ130 == 77 ~ NA,
      ALQ130 == 99 ~ NA, 
      ALQ130 == 777 ~ NA,
      ALQ130 == 999 ~ NA,
      TRUE ~ ALQ130
    ),
    WHD140 = case_when(
      WHD140 == 7777 ~ NA,
      WHD140 == 77777 ~ NA,
      WHD140 == 9999 ~ NA,
      WHD140 == 99999 ~ NA,
      TRUE ~ WHD140
    ),
    DIQ010 = case_when(
      DIQ010 == 1 ~ "Yes",
      DIQ010 == 2 ~ "No",
      DIQ010 == 3 ~ "Borderline",
      DIQ010 == 7 ~ NA,
      DIQ010 == 9 ~ NA,
      TRUE ~ as.character(DIQ010)
    ),
    SEQ060 = case_when(
      SEQ060 == 1 ~ "Yes",
      SEQ060 == 2 ~ "No",
      SEQ060 == 7 ~ NA,
      SEQ060 == 9 ~ NA,
      TRUE ~ as.character(SEQ060)
    ),
    RHQ141 = case_when(
      RHQ141 == 1 ~ "Yes",
      RHQ141 == 2 ~ "No",
      RHQ141 == 7 ~ NA,
      RHQ141 == 9 ~ NA,
      TRUE ~ as.character(RHQ141)
    ),
    RHD143 = case_when(
      RHD143 == 1 ~ "Yes",
      RHD143 == 2 ~ "No",
      RHD143 == 7 ~ NA,
      RHD143 == 9 ~ NA,
      TRUE ~ as.character(RHD143)
    ),
    MCQ250A = case_when(
      MCQ250A == 1 ~ "Yes",
      MCQ250A == 2 ~ "No",
      MCQ250A == 7 ~ NA,
      MCQ250A == 9 ~ NA,
      TRUE ~ as.character(MCQ250A)
    ),
    MCQ300C = case_when(
      MCQ300C == 1 ~ "Yes",
      MCQ300C == 2 ~ "No",
      MCQ300C == 7 ~ NA,
      MCQ300C == 9 ~ NA,
      TRUE ~ as.character(MCQ300C)
    ),
    MCQ300c = case_when(
      MCQ300c == 1 ~ "Yes",
      MCQ300c == 2 ~ "No",
      MCQ300c == 7 ~ NA,
      MCQ300c == 9 ~ NA,
      TRUE ~ as.character(MCQ300c)
    ),
    MCQ160B = case_when(
      MCQ160B == 1 ~ "Yes",
      MCQ160B == 2 ~ "No",
      MCQ160B == 7 ~ NA,
      MCQ160B == 9 ~ NA,
      TRUE ~ as.character(MCQ160B)
    ),
    MCQ160b = case_when(
      MCQ160b == 1 ~ "Yes",
      MCQ160b == 2 ~ "No",
      MCQ160b == 7 ~ NA,
      MCQ160b == 9 ~ NA,
      TRUE ~ as.character(MCQ160b)
    ),
    MCQ160C = case_when(
      MCQ160C == 1 ~ "Yes",
      MCQ160C == 2 ~ "No",
      MCQ160C == 7 ~ NA,
      MCQ160C == 9 ~ NA,
      TRUE ~ as.character(MCQ160C)
    ),
    MCQ160c = case_when(
      MCQ160c == 1 ~ "Yes",
      MCQ160c == 2 ~ "No",
      MCQ160c == 7 ~ NA,
      MCQ160c == 9 ~ NA,
      TRUE ~ as.character(MCQ160c)
    ),
    MCQ160E = case_when(
      MCQ160E == 1 ~ "Yes",
      MCQ160E == 2 ~ "No",
      MCQ160E == 7 ~ NA,
      MCQ160E == 9 ~ NA,
      TRUE ~ as.character(MCQ160E)
    ),
    MCQ160e = case_when(
      MCQ160e == 1 ~ "Yes",
      MCQ160e == 2 ~ "No",
      MCQ160e == 7 ~ NA,
      MCQ160e == 9 ~ NA,
      TRUE ~ as.character(MCQ160e)
    ),
    MCQ160F = case_when(
      MCQ160F == 1 ~ "Yes",
      MCQ160F == 2 ~ "No",
      MCQ160F == 7 ~ NA,
      MCQ160F == 9 ~ NA,
      TRUE ~ as.character(MCQ160F)
    ),
    MCQ160f = case_when(
      MCQ160f == 1 ~ "Yes",
      MCQ160f == 2 ~ "No",
      MCQ160f == 7 ~ NA,
      MCQ160f == 9 ~ NA,
      TRUE ~ as.character(MCQ160f)
    ),
    BPQ080 = case_when(
      BPQ080 == 1 ~ "Yes",
      BPQ080 == 2 ~ "No",
      BPQ080 == 7 ~ NA,
      BPQ080 == 9 ~ NA,
      TRUE ~ as.character(BPQ080)
    ),
    HUQ010 = case_when(
      HUQ010 == 1 ~ "Excellent",
      HUQ010 == 2 ~ "Very Good",
      HUQ010 == 3 ~ "Good",
      HUQ010 == 4 ~ "Fair",
      HUQ010 == 5 ~ "Poor",
      HUQ010 == 7 ~ NA,
      HUQ010 == 9 ~ NA,
      TRUE ~ as.character(HUQ010)
    ),
    HSD010 = case_when(
      HSD010 == 1 ~ "Excellent",
      HSD010 == 2 ~ "Very Good",
      HSD010 == 3 ~ "Good",
      HSD010 == 4 ~ "Fair",
      HSD010 == 5 ~ "Poor",
      HSD010 == 7 ~ NA,
      HSD010 == 9 ~ NA,
      TRUE ~ as.character(HSD010)
    ),
    INDHHIN2 = case_when(
      INDHHIN2 == 1 ~	"$0 to $ 4,999",
      INDHHIN2 == 2 ~	"$5,000 to $ 9,999",
      INDHHIN2 == 3 ~	"$10,000 to $14,999",
      INDHHIN2 == 4 ~	"$15,000 to $19,999",
      INDHHIN2 == 5 ~	"$20,000 to $24,999",
      INDHHIN2 == 6 ~	"$25,000 to $34,999",
      INDHHIN2 == 7 ~	"$35,000 to $44,999",
      INDHHIN2 == 8 ~	"$45,000 to $54,999",
      INDHHIN2 == 9 ~	"$55,000 to $64,999",
      INDHHIN2 == 10 ~ "$65,000 to $74,999",
      INDHHIN2 == 12 ~ "Over $20,000",
      INDHHIN2 == 13 ~ "Under $20,000",
      INDHHIN2 == 14 ~ "$75,000 to $99,999",
      INDHHIN2 == 15 ~ "$100,000 and Over",
      INDHHIN2 == 77 ~ NA,
      INDHHIN2 == 99 ~ NA,
      TRUE ~ as.character(INDHHIN2)
    ),
    RIDRETH1 = case_when(
      RIDRETH1 == 1 ~ "Mexican American",
      RIDRETH1 == 2 ~ "Other Hispanic",
      RIDRETH1 == 3 ~ "Non-Hispanic White",
      RIDRETH1 == 4 ~ "Non-Hispanic Blac",
      RIDRETH1 == 5 ~ "Other Race - Including Multi-Racial",
      TRUE ~ as.character(RIDRETH1)
    ),

  )
```


### 2.2 Homogenize variables that are the same but are called diffrent in different NHANES years

Intake variables went from 1 day in 1999 to 2001 to 2 days from 2003 on, therefore the variable has to be homogenized. Dinh et al. (2019) do not specify which examination records the authors, but my best guess is that they problably took the average of both days that the examination was performed. 

This situation happends with:

- Alcohol intake (`DRXTALCO`, `DR1TALCO`, `DR2TALCO`)
- Caffeine intake (`DRXTCAFF`, `DR1TCAFF`, `DR2TCAFF`)
- Calcium intake (`DRXTCALC`, `DR1TCALC`, `DR2TCALC`)
- Carbohydrate intake (`DRXTCARB`, `DR1TCARB`, `DR2TCARB`)
- Fiber intake (`DRXTFIBE`, `DR1TFIBE`, `DR2TFIBE`)
- Kcal intake (`DRXTKCAL`, `DR1TKCAL`, `DR2TKCAL`)
- Sodium intake (`DRDTSODI`, `DR1TSODI`, `DR2TSODI`)


Also, small changes in same quesion format are registered with different codes. Examples: 

- `MCQ250A`, `MCQ300C` and `MCQ300c`
- `LBDHDDSI` and `LBDHDLSI`
- `LBXGLUSI` and `LBDGLUSI`
- `SEQ060`,`RHQ141`, and `RHD143`

And same questions are coded differnetly as well:

- `MCQ160B` and `MCQ160b`
- `MCQ160C` and `MCQ160c`
- `MCQ160E` and `MCQ160e`
- `MCQ160F` and `MCQ160F`


It can be seen here:


```R
# Similar questions (or the same) with different NHANES variable codes
var_docs <- read_excel(paste0(DINH_DOCS_PATH, "dinh_2019_variables_doc.xlsx"))
var_docs |> 
  filter(`NHANES Name` %in% c('MCQ250A', 'MCQ300C', 'MCQ300c', 'LBDHDDSI', 'LBDHDLSI', 'LBXGLUSI', 'LBDGLUSI'))
```


<table class="dataframe">
<caption>A tibble: 7 x 5</caption>
<thead>
	<tr><th scope=col>Variable Name</th><th scope=col>NHANES Name</th><th scope=col>NHANES File</th><th scope=col>NHANES Type of data</th><th scope=col>Variable Definition</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Blood relatives have diabetes</td><td>MCQ250A </td><td><span style=white-space:pre-wrap>MCQ              </span></td><td>Questionnaire</td><td>Including living and deceased, were any of {SP&amp;apos;s/ your} biological that is, blood relatives including grandparents, parents, brothers, sisters ever told by a health professional that they had . . .diabetes?</td></tr>
	<tr><td><span style=white-space:pre-wrap>Close relative had diabetes  </span></td><td>MCQ300c </td><td><span style=white-space:pre-wrap>MCQ              </span></td><td>Questionnaire</td><td><span style=white-space:pre-wrap>Including living and deceased, were any of {SP&amp;apos;s/your} close biological that is, blood relatives including father, mother, sisters or brothers, ever told by a health professional that they had diabetes?    </span></td></tr>
	<tr><td><span style=white-space:pre-wrap>Close relative had diabetes  </span></td><td>MCQ300C </td><td><span style=white-space:pre-wrap>MCQ              </span></td><td>Questionnaire</td><td><span style=white-space:pre-wrap>Including living and deceased, were any of {SP&amp;apos;s/your} close biological that is, blood relatives including father, mother, sisters or brothers, ever told by a health professional that they had diabetes?    </span></td></tr>
	<tr><td>HDL-cholesterol              </td><td>LBDHDLSI</td><td>Lab13, l13_b, HDL</td><td>Laboratory   </td><td>HDL-cholesterol (mmol/L)                                                                                                                                                                                           </td></tr>
	<tr><td>HDL-cholesterol              </td><td>LBDHDDSI</td><td>Lab13, l13_b, HDL</td><td>Laboratory   </td><td>HDL-cholesterol (mmol/L)                                                                                                                                                                                           </td></tr>
	<tr><td>Plasma Glucose               </td><td>LBXGLUSI</td><td>LAB10AM, L10AM_B </td><td>Laboratory   </td><td>Plasma glucose: SI(mmol/L)                                                                                                                                                                                         </td></tr>
	<tr><td>Plasma Glucose               </td><td>LBDGLUSI</td><td>GLU              </td><td>Laboratory   </td><td>Plasma glucose: SI(mmol/L)                                                                                                                                                                                         </td></tr>
</tbody>
</table>




```R
var_docs |> 
  filter(`NHANES Name` %in% c('MCQ160b', 'MCQ160B', 'MCQ160c', 'MCQ160C', 'MCQ160F', 'MCQ160f', 'MCQ160E', 'MCQ160e', 'SEQ060','RHQ141','RHD143'))
```


<table class="dataframe">
<caption>A tibble: 11 x 5</caption>
<thead>
	<tr><th scope=col>Variable Name</th><th scope=col>NHANES Name</th><th scope=col>NHANES File</th><th scope=col>NHANES Type of data</th><th scope=col>Variable Definition</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Pregnant               </td><td>RHQ141 </td><td>RHQ</td><td>Questionnaire</td><td>{Do you/Does SP} think {you are/he/she is} pregnant now?                                                                                                                </td></tr>
	<tr><td>Pregnant               </td><td>RHD143 </td><td>RHQ</td><td>Questionnaire</td><td>{Are you/Is SP} pregnant now?                                                                                                                                           </td></tr>
	<tr><td>Pregnant               </td><td>SEQ060 </td><td>RHQ</td><td>Questionnaire</td><td>Are you currently pregnant?                                                                                                                                             </td></tr>
	<tr><td>Told CHF by a Doctor   </td><td>MCQ160B</td><td>MCQ</td><td>Questionnaire</td><td>Has a doctor or other health professional ever told {you/SP} that {you/s/he} . . .had congestive heart failure?                                                         </td></tr>
	<tr><td>Told CHF by a Doctor   </td><td>MCQ160b</td><td>MCQ</td><td>Questionnaire</td><td>Has a doctor or other health professional ever told {you/SP} that {you/s/he} . . .had congestive heart failure?                                                         </td></tr>
	<tr><td>Told CHD by a Doctor   </td><td>MCQ160C</td><td>MCQ</td><td>Questionnaire</td><td>Has a doctor or other health professional ever told {you/SP} that {you/s/he} . . .had coronary heart disease?                                                           </td></tr>
	<tr><td>Told CHD by a Doctor   </td><td>MCQ160c</td><td>MCQ</td><td>Questionnaire</td><td>Has a doctor or other health professional ever told {you/SP} that {you/s/he} . . .had coronary heart disease?                                                           </td></tr>
	<tr><td>Told HA by a Doctor    </td><td>MCQ160E</td><td>MCQ</td><td>Questionnaire</td><td>Has a doctor or other health professional ever told {you/SP} that {you/s/he} . . .had a heart attack (also called myocardial infarction (my-o-car-dee-al in-fark-shun))?</td></tr>
	<tr><td>Told HA by a Doctor    </td><td>MCQ160e</td><td>MCQ</td><td>Questionnaire</td><td>Has a doctor or other health professional ever told {you/SP} that {you/s/he} . . .had a heart attack (also called myocardial infarction (my-o-car-dee-al in-fark-shun))?</td></tr>
	<tr><td>Told stroke by a Doctor</td><td>MCQ160F</td><td>MCQ</td><td>Questionnaire</td><td>Has a doctor or other health professional ever told {you/SP} that {you/s/he} . . .had a stroke?                                                                         </td></tr>
	<tr><td>Told stroke by a Doctor</td><td>MCQ160f</td><td>MCQ</td><td>Questionnaire</td><td>Has a doctor or other health professional ever told {you/SP} that {you/s/he} . . .had a stroke?                                                                         </td></tr>
</tbody>
</table>




```R
# unique(df$YEAR[!is.na(df$MCQ250A)])
# unique(df$YEAR[!is.na(df$MCQ300C)])
```

To fix that, I will create a function that creates an average of the Intake variable of Day 1 and Day and average them, givin only one variable - for example "Alcohol_Intake" instead of having 'DRXTALCO', 'DR1TALCO', 'DR2TALCO'.


```R
create_intake_new_column <- function(df, day0_col, day1_col, day2_col) {
    ifelse(is.na(df[[day0_col]]), 
           rowMeans(df[, c(day1_col, day2_col)], na.rm = TRUE), 
           df[[day0_col]])
}

df_formatted <- df_formatted |>
# Create new columns
  mutate(
    # Alcohol intake
    Alcohol_Intake = create_intake_new_column(df,'DRXTALCO', 'DR1TALCO', 'DR2TALCO'),
    # Caffeine intake
    Caffeine_Intake = create_intake_new_column(df,'DRXTCAFF', 'DR1TCAFF', 'DR2TCAFF'),
    # Calcium intake
    Calcium_Intake = create_intake_new_column(df,'DRXTCALC', 'DR1TCALC', 'DR2TCALC'),
    # Carbohydrate intake
    Carbohydrate_Intake = create_intake_new_column(df,'DRXTCARB', 'DR1TCARB', 'DR2TCARB'),
    # Fiber intake
    Fiber_Intake = create_intake_new_column(df,'DRXTFIBE', 'DR1TFIBE', 'DR2TFIBE'),
    # Kcal intake
    Kcal_Intake = create_intake_new_column(df,'DRXTKCAL', 'DR1TKCAL', 'DR2TKCAL'),
    # Sodium intake
    Sodium_Intake = create_intake_new_column(df,'DRDTSODI', 'DR1TSODI', 'DR2TSODI'),
    # Relative_Had_Diabetes
    Relative_Had_Diabetes = coalesce(MCQ250A, MCQ300C, MCQ300c),
    # Heart conditions
    Told_CHF = coalesce(MCQ160B, MCQ160b),
    Told_CHD = coalesce(MCQ160C, MCQ160c),
    Told_HA = coalesce(MCQ160E, MCQ160e),
    Told_stroke = coalesce(MCQ160F, MCQ160f),
    # Pregnancy
    Pregnant = coalesce(SEQ060,RHQ141,RHD143),
    # HDL-cholesterol
    HDL_Cholesterol = coalesce(LBDHDLSI, LBDHDDSI),
    # Glucose
    Glucose = coalesce(LBXGLUSI, LBDGLUSI)
   ) |>
# Delete old columns that are not needed
  select(-c(DRXTALCO, DR1TALCO, DR2TALCO, DRXTCAFF, DR1TCAFF, DR2TCAFF,
            DRXTCALC, DR1TCALC, DR2TCALC, DRXTCARB, DR1TCARB, DR2TCARB,
            DRXTFIBE, DR1TFIBE, DR2TFIBE, DRXTKCAL, DR1TKCAL, DR2TKCAL,
            DRDTSODI, DR1TSODI, DR2TSODI, MCQ250A, MCQ300C, MCQ300c,
            MCQ160B, MCQ160b, MCQ160C, MCQ160c, MCQ160E, MCQ160e,
            MCQ160F, MCQ160f, SEQ060, RHQ141, RHD143,
            LBDHDLSI, LBDHDDSI, LBXGLUSI, LBDGLUSI,)
            )
```


```R
#unique(df_formatted$YEAR[!is.na(df_formatted$Relative_Had_Diabetes)])
```

### 2.3 Choosing between different readings in Blood analysis 

[From NHANES](https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BPX_H.htm): 

> After resting quietly in a seated position for 5 minutes and once the participants maximum inflation level (MIL) has been determined, three consecutive blood pressure readings are obtained. If a blood pressure measurement is interrupted or incomplete, a fourth attempt may be made. All BP determinations (systolic and diastolic) are taken in the mobile examination center (MEC). 

In Dinh et al. (2019) the authors do not say which readings are taking, but I'm assuming they take the last one to avoid the [white coat syndrom](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5352963/) and for data consistency.


```R
df_formatted <- df_formatted |>
# Create new columns
  mutate(
    Diastolic_Blood_Pressure = coalesce(BPXDI4, BPXDI3, BPXDI2, BPXDI1),
    Systolic_Blood_Pressure = coalesce(BPXSY4, BPXSY3, BPXSY2, BPXSY1),
  ) |>
# Delete old columns that are not needed
  select(-c(BPXDI4, BPXDI3, BPXDI2, BPXDI1,
            BPXSY4, BPXSY3, BPXSY2, BPXSY1)
  )
```

### 2.4 Discretional trimming of the data according to the authors

> In our study, all datasets were limited to non-pregnant subjects and adults of at least twenty years of age.


```R
df_formatted |> 
  filter(Pregnant != "Yes") |> 
  filter(RIDAGEYR >= 20)  |> 
  select(-c(RIDAGEYR, Pregnant))
```


<table class="dataframe">
<caption>A tibble: 6012 x 45</caption>
<thead>
	<tr><th scope=col>SEQN</th><th scope=col>YEAR</th><th scope=col>ALQ130</th><th scope=col>BMXARMC</th><th scope=col>BMXARML</th><th scope=col>LBXSOSSI</th><th scope=col>LBDSBUSI</th><th scope=col>BMXBMI</th><th scope=col>LB2SCLSI</th><th scope=col>RIDRETH1</th><th scope=col>...</th><th scope=col>Sodium_Intake</th><th scope=col>Relative_Had_Diabetes</th><th scope=col>Told_CHF</th><th scope=col>Told_CHD</th><th scope=col>Told_HA</th><th scope=col>Told_stroke</th><th scope=col>HDL_Cholesterol</th><th scope=col>Glucose</th><th scope=col>Diastolic_Blood_Pressure</th><th scope=col>Systolic_Blood_Pressure</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>...</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>  7</td><td>1999-2000</td><td>NA</td><td>31.7</td><td>38.1</td><td>283</td><td>3.6</td><td>29.39</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td>3808.53</td><td>Yes</td><td>No</td><td>No</td><td>No </td><td>No</td><td>2.73</td><td>4.756</td><td> 82</td><td>124</td></tr>
	<tr><td> 15</td><td>1999-2000</td><td> 2</td><td>32.5</td><td>37.5</td><td>271</td><td>4.3</td><td>26.68</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>3832.49</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.49</td><td>5.484</td><td> 70</td><td>106</td></tr>
	<tr><td> 24</td><td>1999-2000</td><td> 4</td><td>30.2</td><td>36.0</td><td>272</td><td>3.6</td><td>25.93</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>2811.52</td><td>Yes</td><td>No</td><td>No</td><td>Yes</td><td>No</td><td>2.72</td><td>   NA</td><td> 72</td><td>112</td></tr>
	<tr><td> 25</td><td>1999-2000</td><td> 2</td><td>42.6</td><td>38.0</td><td>273</td><td>5.0</td><td>37.60</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>2251.98</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.42</td><td>5.700</td><td> 84</td><td>120</td></tr>
	<tr><td> 34</td><td>1999-2000</td><td> 4</td><td>30.5</td><td>37.5</td><td>284</td><td>3.6</td><td>25.62</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td>2135.20</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.06</td><td>5.339</td><td> 74</td><td>114</td></tr>
	<tr><td> 45</td><td>1999-2000</td><td>NA</td><td>33.4</td><td>36.1</td><td>282</td><td>3.6</td><td>27.47</td><td>NA</td><td>Mexican American                   </td><td>...</td><td>2473.31</td><td>Yes</td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.41</td><td>   NA</td><td> 80</td><td>114</td></tr>
	<tr><td> 96</td><td>1999-2000</td><td> 1</td><td>33.0</td><td>35.6</td><td> NA</td><td> NA</td><td>27.54</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td> 977.22</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>  NA</td><td>   NA</td><td> 82</td><td>126</td></tr>
	<tr><td>102</td><td>1999-2000</td><td> 1</td><td>34.0</td><td>33.6</td><td>275</td><td>4.6</td><td>26.32</td><td>NA</td><td>Other Race - Including Multi-Racial</td><td>...</td><td>3005.35</td><td>Yes</td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.18</td><td>4.572</td><td> 70</td><td>100</td></tr>
	<tr><td>107</td><td>1999-2000</td><td> 2</td><td>30.7</td><td>32.9</td><td>278</td><td>3.9</td><td>26.97</td><td>NA</td><td>Mexican American                   </td><td>...</td><td>4313.80</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.34</td><td>   NA</td><td> 54</td><td>106</td></tr>
	<tr><td>115</td><td>1999-2000</td><td> 1</td><td>27.8</td><td>33.8</td><td>259</td><td>2.9</td><td>20.89</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>    NaN</td><td>No </td><td>No</td><td>NA</td><td>No </td><td>No</td><td>1.31</td><td>   NA</td><td> 74</td><td>116</td></tr>
	<tr><td>132</td><td>1999-2000</td><td>NA</td><td>38.2</td><td>40.8</td><td>275</td><td>3.2</td><td>41.93</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>1060.47</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.02</td><td>5.928</td><td> 68</td><td>110</td></tr>
	<tr><td>141</td><td>1999-2000</td><td> 3</td><td>30.5</td><td>37.3</td><td>269</td><td>4.3</td><td>26.19</td><td>NA</td><td>Other Hispanic                     </td><td>...</td><td>3652.88</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.28</td><td>5.222</td><td> 76</td><td>148</td></tr>
	<tr><td>149</td><td>1999-2000</td><td>NA</td><td>31.0</td><td>34.2</td><td>273</td><td>2.9</td><td>27.06</td><td>NA</td><td>Other Hispanic                     </td><td>...</td><td>4686.03</td><td>Yes</td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.17</td><td>4.972</td><td> 58</td><td> 94</td></tr>
	<tr><td>177</td><td>1999-2000</td><td> 2</td><td>37.6</td><td>37.7</td><td>278</td><td>7.1</td><td>35.56</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>7829.90</td><td>Yes</td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.16</td><td>5.850</td><td> 64</td><td>114</td></tr>
	<tr><td>184</td><td>1999-2000</td><td>NA</td><td>30.1</td><td>35.5</td><td>266</td><td>3.9</td><td>24.69</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td>1128.19</td><td>Yes</td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.37</td><td>   NA</td><td> 68</td><td>120</td></tr>
	<tr><td>188</td><td>1999-2000</td><td> 1</td><td>32.1</td><td>40.1</td><td>279</td><td>3.6</td><td>28.63</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>2346.64</td><td>Yes</td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.22</td><td>   NA</td><td>104</td><td>160</td></tr>
	<tr><td>192</td><td>1999-2000</td><td>NA</td><td>30.0</td><td>37.0</td><td>272</td><td>4.3</td><td>26.48</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td>4682.73</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.65</td><td>4.519</td><td> 60</td><td> 98</td></tr>
	<tr><td>193</td><td>1999-2000</td><td>NA</td><td>25.2</td><td>36.0</td><td>284</td><td>3.6</td><td>19.54</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td>4156.90</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.39</td><td>   NA</td><td> 88</td><td>118</td></tr>
	<tr><td>194</td><td>1999-2000</td><td>NA</td><td>35.0</td><td>36.0</td><td>278</td><td>3.6</td><td>31.54</td><td>NA</td><td>Mexican American                   </td><td>...</td><td>3175.70</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.39</td><td>5.389</td><td> 74</td><td>116</td></tr>
	<tr><td>198</td><td>1999-2000</td><td> 1</td><td>31.1</td><td>35.8</td><td>282</td><td>5.7</td><td>27.74</td><td>NA</td><td>Other Hispanic                     </td><td>...</td><td>3540.22</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.71</td><td>   NA</td><td> 76</td><td>116</td></tr>
	<tr><td>199</td><td>1999-2000</td><td>NA</td><td>28.0</td><td>33.0</td><td>282</td><td>3.2</td><td>25.66</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>1337.21</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.31</td><td>   NA</td><td> 74</td><td>120</td></tr>
	<tr><td>206</td><td>1999-2000</td><td>NA</td><td>30.4</td><td>33.0</td><td>281</td><td>5.7</td><td>25.79</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td>3261.32</td><td>Yes</td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.23</td><td>   NA</td><td> 92</td><td>132</td></tr>
	<tr><td>232</td><td>1999-2000</td><td> 1</td><td>31.7</td><td>34.9</td><td>276</td><td>3.9</td><td>25.34</td><td>NA</td><td>Mexican American                   </td><td>...</td><td>1106.77</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>2.17</td><td>4.817</td><td> 62</td><td>110</td></tr>
	<tr><td>236</td><td>1999-2000</td><td> 3</td><td>28.0</td><td>31.0</td><td>273</td><td>3.9</td><td>27.04</td><td>NA</td><td>Mexican American                   </td><td>...</td><td>3637.11</td><td>Yes</td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.28</td><td>6.189</td><td> 66</td><td>108</td></tr>
	<tr><td>250</td><td>1999-2000</td><td> 1</td><td>29.7</td><td>34.7</td><td>276</td><td>4.6</td><td>26.90</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>3113.84</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.84</td><td>4.839</td><td> 74</td><td>116</td></tr>
	<tr><td>265</td><td>1999-2000</td><td>10</td><td>40.9</td><td>33.5</td><td>282</td><td>5.4</td><td>41.84</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>4267.00</td><td>Yes</td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.21</td><td>   NA</td><td> 74</td><td>120</td></tr>
	<tr><td>300</td><td>1999-2000</td><td> 4</td><td>34.8</td><td>35.7</td><td>278</td><td>5.4</td><td>30.26</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>1249.04</td><td>Yes</td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.22</td><td>4.550</td><td> 84</td><td>110</td></tr>
	<tr><td>301</td><td>1999-2000</td><td> 2</td><td>30.6</td><td>35.4</td><td>276</td><td>3.2</td><td>24.80</td><td>NA</td><td>Mexican American                   </td><td>...</td><td>2460.69</td><td>No </td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.21</td><td>5.039</td><td> 74</td><td>104</td></tr>
	<tr><td>314</td><td>1999-2000</td><td> 1</td><td>27.7</td><td>37.3</td><td>273</td><td>3.2</td><td>24.18</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td>3030.99</td><td>Yes</td><td>No</td><td>No</td><td>No </td><td>No</td><td>1.36</td><td>   NA</td><td> 70</td><td>114</td></tr>
	<tr><td>334</td><td>1999-2000</td><td> 1</td><td>28.0</td><td>29.6</td><td>282</td><td>3.9</td><td>22.82</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>1562.63</td><td>Yes</td><td>No</td><td>No</td><td>No </td><td>No</td><td>2.53</td><td>4.917</td><td> 60</td><td>154</td></tr>
	<tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td></td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>
	<tr><td>83214</td><td>2013-2014</td><td> 2</td><td>31.4</td><td>32.1</td><td>271</td><td>2.50</td><td>30.5</td><td>NA</td><td>Mexican American                   </td><td>...</td><td>2288.0</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.55</td><td>   NA</td><td>68</td><td> 96</td></tr>
	<tr><td>83252</td><td>2013-2014</td><td> 2</td><td>38.2</td><td>39.5</td><td>274</td><td>2.14</td><td>28.8</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td>3293.5</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.03</td><td>4.885</td><td>68</td><td>112</td></tr>
	<tr><td>83260</td><td>2013-2014</td><td>NA</td><td>27.0</td><td>33.4</td><td>274</td><td>3.57</td><td>21.4</td><td>NA</td><td>Other Race - Including Multi-Racial</td><td>...</td><td>2085.5</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.32</td><td>4.996</td><td>72</td><td>102</td></tr>
	<tr><td>83301</td><td>2013-2014</td><td>NA</td><td>38.0</td><td>34.0</td><td>283</td><td>3.57</td><td>42.0</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>2993.5</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>0.93</td><td>   NA</td><td>80</td><td>118</td></tr>
	<tr><td>83329</td><td>2013-2014</td><td> 2</td><td>24.8</td><td>32.7</td><td>272</td><td>1.79</td><td>19.9</td><td>NA</td><td>Other Hispanic                     </td><td>...</td><td>4063.0</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.58</td><td>   NA</td><td>52</td><td>108</td></tr>
	<tr><td>83336</td><td>2013-2014</td><td>NA</td><td>30.4</td><td>34.3</td><td>277</td><td>1.79</td><td>27.8</td><td>NA</td><td>Mexican American                   </td><td>...</td><td>1650.5</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>0.70</td><td>   NA</td><td>84</td><td>136</td></tr>
	<tr><td>83356</td><td>2013-2014</td><td> 2</td><td>29.2</td><td>34.3</td><td>280</td><td>1.79</td><td>25.9</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>6006.5</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>0.78</td><td>   NA</td><td>76</td><td>108</td></tr>
	<tr><td>83362</td><td>2013-2014</td><td> 1</td><td>24.8</td><td>33.3</td><td>278</td><td>5.36</td><td>20.6</td><td>NA</td><td>Other Race - Including Multi-Racial</td><td>...</td><td>4329.5</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.14</td><td>5.384</td><td>68</td><td>104</td></tr>
	<tr><td>83368</td><td>2013-2014</td><td> 1</td><td>24.2</td><td>32.7</td><td>276</td><td>3.57</td><td>20.3</td><td>NA</td><td>Other Race - Including Multi-Racial</td><td>...</td><td>2986.5</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.60</td><td>   NA</td><td>74</td><td> 96</td></tr>
	<tr><td>83372</td><td>2013-2014</td><td> 4</td><td>30.4</td><td>33.5</td><td>282</td><td>4.28</td><td>25.5</td><td>NA</td><td>Other Hispanic                     </td><td>...</td><td>2734.0</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.11</td><td>5.273</td><td>46</td><td> 92</td></tr>
	<tr><td>83411</td><td>2013-2014</td><td> 2</td><td>28.0</td><td>32.0</td><td>278</td><td>2.86</td><td>22.2</td><td>NA</td><td>Mexican American                   </td><td>...</td><td>1678.0</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.99</td><td>   NA</td><td>86</td><td>122</td></tr>
	<tr><td>83412</td><td>2013-2014</td><td> 1</td><td>28.5</td><td>33.9</td><td> NA</td><td>  NA</td><td>22.5</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td>3212.5</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>  NA</td><td>   NA</td><td>84</td><td>132</td></tr>
	<tr><td>83415</td><td>2013-2014</td><td> 1</td><td>24.8</td><td>33.8</td><td>273</td><td>5.00</td><td>19.9</td><td>NA</td><td>Other Race - Including Multi-Racial</td><td>...</td><td>5419.5</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.58</td><td>4.385</td><td>72</td><td> 98</td></tr>
	<tr><td>83427</td><td>2013-2014</td><td> 3</td><td>32.0</td><td>36.5</td><td>281</td><td>3.57</td><td>28.8</td><td>NA</td><td>Mexican American                   </td><td>...</td><td>5459.0</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.06</td><td>5.162</td><td>60</td><td> 92</td></tr>
	<tr><td>83433</td><td>2013-2014</td><td> 1</td><td>29.5</td><td>32.5</td><td>280</td><td>3.93</td><td>22.8</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>4499.0</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.73</td><td>5.218</td><td>46</td><td> 90</td></tr>
	<tr><td>83448</td><td>2013-2014</td><td> 3</td><td>34.1</td><td>33.7</td><td>279</td><td>4.28</td><td>32.8</td><td>NA</td><td>Other Race - Including Multi-Racial</td><td>...</td><td>3830.0</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.66</td><td>7.050</td><td>76</td><td>126</td></tr>
	<tr><td>83459</td><td>2013-2014</td><td> 1</td><td>43.5</td><td>36.0</td><td> NA</td><td>  NA</td><td>45.1</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>2465.0</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>  NA</td><td>   NA</td><td>56</td><td>104</td></tr>
	<tr><td>83461</td><td>2013-2014</td><td> 3</td><td>26.1</td><td>34.6</td><td>271</td><td>1.79</td><td>23.4</td><td>NA</td><td>Other Hispanic                     </td><td>...</td><td>2266.0</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>2.25</td><td>   NA</td><td>78</td><td>136</td></tr>
	<tr><td>83491</td><td>2013-2014</td><td> 3</td><td>41.0</td><td>43.0</td><td>281</td><td>5.00</td><td>40.0</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td>3033.0</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.47</td><td>5.384</td><td>80</td><td>118</td></tr>
	<tr><td>83497</td><td>2013-2014</td><td> 1</td><td>28.8</td><td>32.5</td><td>279</td><td>5.00</td><td>23.2</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>   NaN</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.53</td><td>5.162</td><td>66</td><td>100</td></tr>
	<tr><td>83518</td><td>2013-2014</td><td> 2</td><td>32.6</td><td>36.5</td><td>278</td><td>3.93</td><td>30.2</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td>3498.0</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.24</td><td>   NA</td><td>66</td><td>126</td></tr>
	<tr><td>83558</td><td>2013-2014</td><td>NA</td><td>31.4</td><td>36.7</td><td> NA</td><td>  NA</td><td>28.0</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td> 743.5</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>  NA</td><td>   NA</td><td>64</td><td>112</td></tr>
	<tr><td>83643</td><td>2013-2014</td><td> 6</td><td>26.6</td><td>33.3</td><td>275</td><td>3.21</td><td>22.3</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>3276.0</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.24</td><td>   NA</td><td>64</td><td>100</td></tr>
	<tr><td>83678</td><td>2013-2014</td><td>10</td><td>23.7</td><td>34.7</td><td>275</td><td>2.50</td><td>18.3</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>5585.5</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>3.03</td><td>5.384</td><td>78</td><td>122</td></tr>
	<tr><td>83683</td><td>2013-2014</td><td>NA</td><td>34.4</td><td>36.2</td><td>277</td><td>2.14</td><td>30.0</td><td>NA</td><td>Mexican American                   </td><td>...</td><td>3933.5</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.73</td><td>5.607</td><td>68</td><td>102</td></tr>
	<tr><td>83688</td><td>2013-2014</td><td> 3</td><td>26.5</td><td>34.6</td><td>279</td><td>3.21</td><td>22.5</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>3783.0</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.11</td><td>   NA</td><td>72</td><td>104</td></tr>
	<tr><td>83689</td><td>2013-2014</td><td> 3</td><td>27.0</td><td>36.0</td><td>285</td><td>5.00</td><td>23.6</td><td>NA</td><td>Non-Hispanic Blac                  </td><td>...</td><td>3646.0</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.19</td><td>   NA</td><td>76</td><td>112</td></tr>
	<tr><td>83694</td><td>2013-2014</td><td> 1</td><td>31.9</td><td>34.8</td><td>276</td><td>3.21</td><td>25.3</td><td>NA</td><td>Other Race - Including Multi-Racial</td><td>...</td><td>3449.0</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>0.96</td><td>5.495</td><td>60</td><td>116</td></tr>
	<tr><td>83699</td><td>2013-2014</td><td> 1</td><td>25.2</td><td>34.0</td><td>276</td><td>2.50</td><td>20.8</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>5362.0</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.68</td><td>4.607</td><td>70</td><td>114</td></tr>
	<tr><td>83711</td><td>2013-2014</td><td>NA</td><td>32.1</td><td>33.8</td><td>282</td><td>2.14</td><td>33.5</td><td>NA</td><td>Non-Hispanic White                 </td><td>...</td><td>1680.0</td><td>No </td><td>No</td><td>No</td><td>No</td><td>No</td><td>1.11</td><td>5.551</td><td>78</td><td>112</td></tr>
</tbody>
</table>



### 2.5 Creating the  Target Variables

Tables 1 & 3 from Dinh et al. 2019:

From [Tables 1 & 3 from Dinh et al. 2019](https://raw.githubusercontent.com/pipegalera/ml_diabetes/main/images/dinh_2019_Table1_3.png):

`Diabetes = 1` if

- Glucose >= 126 mg/dL. OR;
- "Yes" to the question "Have you ever been told by a doctor that you have diabetes?"

`undiagnosed diabetes = 1` if

- Glucose >= 126 mg/dL. AND;
- "No" to the question "Have you ever been told by a doctor that you have diabetes?" and had a blood glucose level greater than or equal

`pre diabetes = 1` if

- Glucose 125 >= 100 mg/dL

`CVD = 1` if

- "Yes" to any of the the questions "Have you ever been told by a doctor that you had congestive heart failure, coronary heart disease, a heart attack, or a stroke?"




```R
colnames(df_formatted)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'SEQN'</li><li>'YEAR'</li><li>'RIDAGEYR'</li><li>'ALQ130'</li><li>'BMXARMC'</li><li>'BMXARML'</li><li>'LBXSOSSI'</li><li>'LBDSBUSI'</li><li>'BMXBMI'</li><li>'LB2SCLSI'</li><li>'RIDRETH1'</li><li>'LBXSGTSI'</li><li>'HSD010'</li><li>'HUQ010'</li><li>'BMXHT'</li><li>'BPQ080'</li><li>'INDHHIN2'</li><li>'LBDLDLSI'</li><li>'BMXLEG'</li><li>'LBDLYMNO'</li><li>'LBXMCVSI'</li><li>'BPXPLS'</li><li>'WHD140'</li><li>'LBDTCSI'</li><li>'LBDSTRSI'</li><li>'BMXWAIST'</li><li>'BMXWT'</li><li>'LBXWBCSI'</li><li>'LBXSASSI'</li><li>'DIQ010'</li><li>'Alcohol_Intake'</li><li>'Caffeine_Intake'</li><li>'Calcium_Intake'</li><li>'Carbohydrate_Intake'</li><li>'Fiber_Intake'</li><li>'Kcal_Intake'</li><li>'Sodium_Intake'</li><li>'Relative_Had_Diabetes'</li><li>'Told_CHF'</li><li>'Told_CHD'</li><li>'Told_HA'</li><li>'Told_stroke'</li><li>'Pregnant'</li><li>'HDL_Cholesterol'</li><li>'Glucose'</li><li>'Diastolic_Blood_Pressure'</li><li>'Systolic_Blood_Pressure'</li></ol>




```R
df_formatted <- df_formatted %>%
  mutate(
    # Diabetic or not diabetic
    Diabetes_Case_I = case_when(
      (Glucose > 7.0 | DIQ010 == "Yes") ~ 1,
      TRUE ~ 0),
    Diabetes_Case_II = case_when(
      # Undiagnosed Diabetic 
      (Diabetes_Case_I == 0 & Glucose > 7.0 & DIQ010 == "No") ~ 1,
      # Prediabetic
      (Diabetes_Case_I == 0 & Glucose >= 5.6 & Glucose < 7.0) ~ 1,
      TRUE ~  0),
    # Cardiovascular Disease
    CVD = case_when(
      (Told_CHF == "Yes" | Told_CHD == "Yes" | Told_HA == "Yes" | Told_stroke == "Yes") ~ 1,
      TRUE ~  0)
  )  |>
  select(-c(Told_CHF, Told_CHD, Told_HA, Told_stroke, Glucose, DIQ010))
```

### 2.6 Column name formatting


```R
df_formatted <- df_formatted %>% 
  rename(
    Alcohol_consumption = ALQ130,
    Arm_circumference = BMXARMC,
    Arm_length = BMXARML,
    Body_mass_index = BMXBMI,
    Height = BMXHT,
    Leg_length = BMXLEG,
    Waist_circumference = BMXWAIST,
    Weight = BMXWT,
    Told_High_Cholesterol = BPQ080,
    Pulse = BPXPLS,
    General_health = HSD010,
    Health_status = HUQ010,
    Household_income = INDHHIN2,
    Chloride = LB2SCLSI,
    LDL_cholesterol = LBDLDLSI,
    Lymphocytes = LBDLYMNO,
    Blood_urea_nitrogen = LBDSBUSI,
    Triglycerides = LBDSTRSI,
    Total_cholesterol = LBDTCSI,
    Mean_cell_volume = LBXMCVSI,
    Aspartate_aminotransferase_AST = LBXSASSI,
    Gamma_glutamyl_transferase = LBXSGTSI,
    Osmolality = LBXSOSSI,
    White_blood_cell_count = LBXWBCSI,
    Age = RIDAGEYR,
    Race_ethnicity = RIDRETH1,
    `Self-reported_greatest_weight` = WHD140,
    Survey_year = YEAR,
  )
```


```R
colnames(df_formatted)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'SEQN'</li><li>'Survey_year'</li><li>'Age'</li><li>'Alcohol_consumption'</li><li>'Arm_circumference'</li><li>'Arm_length'</li><li>'Osmolality'</li><li>'Blood_urea_nitrogen'</li><li>'Body_mass_index'</li><li>'Chloride'</li><li>'Race_ethnicity'</li><li>'Gamma_glutamyl_transferase'</li><li>'General_health'</li><li>'Health_status'</li><li>'Height'</li><li>'Told_High_Cholesterol'</li><li>'Household_income'</li><li>'LDL_cholesterol'</li><li>'Leg_length'</li><li>'Lymphocytes'</li><li>'Mean_cell_volume'</li><li>'Pulse'</li><li>'Self-reported_greatest_weight'</li><li>'Total_cholesterol'</li><li>'Triglycerides'</li><li>'Waist_circumference'</li><li>'Weight'</li><li>'White_blood_cell_count'</li><li>'Aspartate_aminotransferase_AST'</li><li>'Alcohol_Intake'</li><li>'Caffeine_Intake'</li><li>'Calcium_Intake'</li><li>'Carbohydrate_Intake'</li><li>'Fiber_Intake'</li><li>'Kcal_Intake'</li><li>'Sodium_Intake'</li><li>'Relative_Had_Diabetes'</li><li>'Pregnant'</li><li>'HDL_Cholesterol'</li><li>'Diastolic_Blood_Pressure'</li><li>'Systolic_Blood_Pressure'</li><li>'Diabetes_Case_I'</li><li>'Diabetes_Case_II'</li><li>'CVD'</li></ol>



### 2.7 Normalization and Categorical Encoding.


> Normalization was performed on the data using the following standardization model: x' = x−x^/σ 

Before we apply `scale`, we need to: 

1. Classify all the columns between categorical and numerical.
2. Only apply the standarization to the numerical ones. 



```R
# Categorical variables
categorical_vars <- c(
  'SEQN',
  'Survey_year',
  'Race_ethnicity',
  'General_health',
  'Health_status',
  'Told_High_Cholesterol',
  'Household_income',
  'Relative_Had_Diabetes'
)

# Numerical variables
numerical_vars <- c(
  'Age',
  'Alcohol_consumption',
  'Arm_circumference',
  'Arm_length',
  'Osmolality',
  'Blood_urea_nitrogen',
  'Body_mass_index',
  'Chloride',
  'Gamma_glutamyl_transferase',
  'Height',
  'LDL_cholesterol',
  'Leg_length',
  'Lymphocytes',
  'Mean_cell_volume',
  'Pulse',
  'Self-reported_greatest_weight',
  'Total_cholesterol',
  'Triglycerides',
  'Waist_circumference',
  'Weight',
  'White_blood_cell_count',
  'Aspartate_aminotransferase_AST',
  'Alcohol_Intake',
  'Caffeine_Intake',
  'Calcium_Intake',
  'Carbohydrate_Intake',
  'Fiber_Intake',
  'Kcal_Intake',
  'Sodium_Intake',
  'HDL_Cholesterol',
  'Diastolic_Blood_Pressure',
  'Systolic_Blood_Pressure'
)

df_formatted <- df_formatted |> 
    mutate(
        across(all_of(numerical_vars), scale)
    )

```


```R
df_formatted
```


<table class="dataframe">
<caption>A tibble: 82091 x 44</caption>
<thead>
	<tr><th scope=col>SEQN</th><th scope=col>Survey_year</th><th scope=col>Age</th><th scope=col>Alcohol_consumption</th><th scope=col>Arm_circumference</th><th scope=col>Arm_length</th><th scope=col>Osmolality</th><th scope=col>Blood_urea_nitrogen</th><th scope=col>Body_mass_index</th><th scope=col>Chloride</th><th scope=col>...</th><th scope=col>Kcal_Intake</th><th scope=col>Sodium_Intake</th><th scope=col>Relative_Had_Diabetes</th><th scope=col>Pregnant</th><th scope=col>HDL_Cholesterol</th><th scope=col>Diastolic_Blood_Pressure</th><th scope=col>Systolic_Blood_Pressure</th><th scope=col>Diabetes_Case_I</th><th scope=col>Diabetes_Case_II</th><th scope=col>CVD</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>...</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;dbl[,1]&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td> 1</td><td>1999-2000</td><td>-1.1528298</td><td>         NA</td><td>-1.6636987</td><td>-1.8937916</td><td>         NA</td><td>         NA</td><td>-1.38902690</td><td>NA</td><td>...</td><td>-0.6751562</td><td>-0.89336922</td><td>NA </td><td>NA </td><td>         NA</td><td>         NA</td><td>        NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td> 2</td><td>1999-2000</td><td> 1.8626044</td><td>-0.65078964</td><td> 0.2069443</td><td> 0.6926871</td><td> 1.96698244</td><td> 1.12256662</td><td>-0.05442850</td><td>NA</td><td>...</td><td> 0.5726372</td><td> 1.63753100</td><td>No </td><td>NA </td><td> 0.06515962</td><td>-0.62537804</td><td>-1.0586164</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td> 3</td><td>1999-2000</td><td>-0.8311835</td><td>         NA</td><td>-1.0871307</td><td>-0.9832455</td><td>         NA</td><td>         NA</td><td>-1.02468154</td><td>NA</td><td>...</td><td>-0.4956811</td><td>-0.85922508</td><td>NA </td><td>NA </td><td>-1.51960665</td><td>-0.23674586</td><td>-0.3205259</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td> 4</td><td>1999-2000</td><td>-1.1930356</td><td>         NA</td><td>-1.5099472</td><td>-1.6562579</td><td>         NA</td><td>         NA</td><td>         NA</td><td>NA</td><td>...</td><td>-0.5440052</td><td>-1.10633059</td><td>NA </td><td>NA </td><td>         NA</td><td>         NA</td><td>        NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td> 5</td><td>1999-2000</td><td> 0.7368423</td><td> 0.04596624</td><td> 0.9757016</td><td> 0.8906320</td><td>-0.34171723</td><td> 0.58303298</td><td> 0.50610283</td><td>NA</td><td>...</td><td> 0.7931698</td><td> 0.42820578</td><td>No </td><td>NA </td><td>-0.74021340</td><td> 1.05869472</td><td> 0.2066816</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td> 6</td><td>1999-2000</td><td>-0.4693314</td><td>         NA</td><td>-0.2799354</td><td> 0.2044233</td><td>-0.14932559</td><td>-0.64317984</td><td>-0.36672453</td><td>NA</td><td>...</td><td>-0.9522854</td><td>-1.30923318</td><td>NA </td><td>No </td><td> 0.53279557</td><td> 0.92915066</td><td>-0.3205259</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td> 7</td><td>1999-2000</td><td> 1.1389002</td><td>         NA</td><td> 0.4503841</td><td> 0.6794908</td><td> 1.00502424</td><td>-0.44698579</td><td> 0.54480618</td><td>NA</td><td>...</td><td> 0.1325214</td><td> 0.46049910</td><td>Yes</td><td>No </td><td> 3.54644946</td><td> 1.05869472</td><td> 0.3121231</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td> 8</td><td>1999-2000</td><td>-0.7105661</td><td>         NA</td><td>-1.0358802</td><td> 0.4947423</td><td>-0.72650051</td><td> 0.92637256</td><td>-1.30761640</td><td>NA</td><td>...</td><td> 4.6304409</td><td> 4.53106721</td><td>NA </td><td>NA </td><td> 0.97445174</td><td>-0.62537804</td><td>-1.1640579</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td> 9</td><td>1999-2000</td><td>-0.7909777</td><td>         NA</td><td>-0.7796277</td><td>-0.1254847</td><td>         NA</td><td>         NA</td><td>-0.91124067</td><td>NA</td><td>...</td><td>-0.4810347</td><td>-0.72717962</td><td>NA </td><td>NA </td><td> 0.32495737</td><td>-1.14355427</td><td>-0.4259674</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>10</td><td>1999-2000</td><td> 0.4956076</td><td>-0.65078964</td><td> 1.2063289</td><td> 1.3261105</td><td> 0.62024097</td><td> 0.04349934</td><td> 0.75166894</td><td>NA</td><td>...</td><td> 1.7667610</td><td> 0.19345554</td><td>NA </td><td>NA </td><td>-0.14267858</td><td> 1.96550312</td><td> 1.2610966</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>11</td><td>1999-2000</td><td>-0.6301545</td><td>         NA</td><td> 0.1300685</td><td> 0.5607239</td><td> 0.04306605</td><td> 0.23969339</td><td>-0.44146204</td><td>NA</td><td>...</td><td> 4.2676934</td><td> 2.66495635</td><td>NA </td><td>NA </td><td>-0.89609205</td><td>-1.01401021</td><td>-0.7422919</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>12</td><td>1999-2000</td><td> 0.2543728</td><td> 0.04596624</td><td> 1.1550784</td><td> 0.9302209</td><td> 1.00502424</td><td> 1.26971215</td><td> 0.70896179</td><td>NA</td><td>...</td><td> 1.5731480</td><td> 2.75244606</td><td>Yes</td><td>NA </td><td>-1.00001115</td><td> 2.22459124</td><td> 3.0536022</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>13</td><td>1999-2000</td><td> 1.5811639</td><td>-0.30241170</td><td>-0.1518092</td><td> 0.0724601</td><td> 1.96698244</td><td> 1.12256662</td><td> 0.03498959</td><td>NA</td><td>...</td><td>-0.5889842</td><td>-1.23668618</td><td>Yes</td><td>NA </td><td>-0.22061790</td><td> 0.28143037</td><td> 0.6284476</td><td>1</td><td>0</td><td>0</td></tr>
	<tr><td>14</td><td>1999-2000</td><td> 2.0234276</td><td>-0.65078964</td><td> 0.6681987</td><td> 0.4947423</td><td> 1.19741588</td><td> 0.23969339</td><td> 0.26987891</td><td>NA</td><td>...</td><td>       NaN</td><td>        NaN</td><td>No </td><td>NA </td><td>-0.84413250</td><td>-0.10720180</td><td> 1.0502136</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>15</td><td>1999-2000</td><td> 0.2945786</td><td>-0.30241170</td><td> 0.5528851</td><td> 0.6003129</td><td>-1.30367543</td><td>-0.10364620</td><td> 0.18313002</td><td>NA</td><td>...</td><td> 0.6971092</td><td> 0.47533039</td><td>No </td><td>No </td><td> 0.32495737</td><td> 0.28143037</td><td>-0.6368504</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>1999-2000</td><td> 2.1842507</td><td>         NA</td><td>-0.5874384</td><td> 0.2440123</td><td> 2.35176572</td><td> 2.69211902</td><td>-0.71372011</td><td>NA</td><td>...</td><td>-1.1955787</td><td>-0.57855110</td><td>No </td><td>NA </td><td> 0.11711917</td><td>-0.23674586</td><td> 0.9447721</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>17</td><td>1999-2000</td><td>-1.1528298</td><td>         NA</td><td>-1.5996356</td><td>-2.0785401</td><td>         NA</td><td>         NA</td><td>-1.43974164</td><td>NA</td><td>...</td><td> 1.2350486</td><td> 1.02301698</td><td>NA </td><td>NA </td><td>         NA</td><td>         NA</td><td>        NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>18</td><td>1999-2000</td><td>-1.1930356</td><td>         NA</td><td>-1.3946336</td><td>-2.4348407</td><td>         NA</td><td>         NA</td><td>         NA</td><td>NA</td><td>...</td><td>-0.7626053</td><td>-0.70649258</td><td>NA </td><td>NA </td><td>         NA</td><td>         NA</td><td>        NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>19</td><td>1999-2000</td><td>-1.2332413</td><td>         NA</td><td>-1.4715094</td><td>-1.9729695</td><td>         NA</td><td>         NA</td><td>         NA</td><td>NA</td><td>...</td><td>-1.3240626</td><td>-1.24700494</td><td>NA </td><td>NA </td><td>         NA</td><td>         NA</td><td>        NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>20</td><td>1999-2000</td><td>-0.3085082</td><td> 0.04596624</td><td>-0.2671228</td><td> 0.2044233</td><td>-1.30367543</td><td>-0.44698579</td><td>-0.21724950</td><td>NA</td><td>...</td><td>-0.3249305</td><td>-0.19694269</td><td>Yes</td><td>Yes</td><td>-0.68825385</td><td>-0.23674586</td><td>-0.7422919</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>21</td><td>1999-2000</td><td>-0.5095371</td><td>         NA</td><td> 0.6810113</td><td> 0.4023681</td><td>-0.91889215</td><td>-0.10364620</td><td> 1.92878473</td><td>NA</td><td>...</td><td>-0.8869529</td><td>-1.05926788</td><td>NA </td><td>NA </td><td>-1.23382913</td><td> 0.67006254</td><td>-0.1096429</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>22</td><td>1999-2000</td><td>-0.7105661</td><td>         NA</td><td>-0.6899394</td><td> 0.5739203</td><td>-0.14932559</td><td>-0.10364620</td><td>-0.79513061</td><td>NA</td><td>...</td><td>-1.0252463</td><td>-0.76197371</td><td>NA </td><td>No </td><td> 0.66269444</td><td> 0.41097443</td><td>-0.6368504</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>23</td><td>1999-2000</td><td>-0.7507719</td><td>         NA</td><td> 0.5913230</td><td> 0.7718651</td><td> 0.23545769</td><td>-0.44698579</td><td> 0.12307309</td><td>NA</td><td>...</td><td>-0.1890330</td><td>-0.02995458</td><td>NA </td><td>No </td><td>-1.38970777</td><td>-0.49583398</td><td>-1.1640579</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>24</td><td>1999-2000</td><td> 0.8976655</td><td> 0.39434418</td><td> 0.2581948</td><td> 0.4023681</td><td>-1.11128379</td><td>-0.44698579</td><td> 0.08303514</td><td>NA</td><td>...</td><td>-0.3149741</td><td>-0.15665186</td><td>Yes</td><td>No </td><td> 3.52046969</td><td> 0.41097443</td><td>-0.3205259</td><td>0</td><td>0</td><td>1</td></tr>
	<tr><td>25</td><td>1999-2000</td><td> 0.4554018</td><td>-0.30241170</td><td> 1.8469600</td><td> 0.6662945</td><td>-0.91889215</td><td> 0.23969339</td><td> 1.64051147</td><td>NA</td><td>...</td><td>-0.1462465</td><td>-0.50300812</td><td>No </td><td>No </td><td> 0.14309895</td><td> 1.18823878</td><td> 0.1012401</td><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>26</td><td>1999-2000</td><td>-0.6703603</td><td>         NA</td><td> 0.9885143</td><td> 0.9566136</td><td> 1.38980752</td><td>-0.64317984</td><td> 0.91715914</td><td>NA</td><td>...</td><td>-1.7045530</td><td>-1.70810555</td><td>NA </td><td>No </td><td>-0.27257745</td><td>-0.36628992</td><td>-1.1640579</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>27</td><td>1999-2000</td><td>-0.5095371</td><td>         NA</td><td> 0.2069443</td><td> 0.7718651</td><td>         NA</td><td>         NA</td><td> 0.32593205</td><td>NA</td><td>...</td><td> 0.1742682</td><td>-0.78486439</td><td>NA </td><td>NA </td><td>         NA</td><td> 0.79960660</td><td>-0.1096429</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>28</td><td>1999-2000</td><td>-0.5095371</td><td>         NA</td><td>-0.2286849</td><td> 0.4947423</td><td> 0.23545769</td><td>-0.29984025</td><td>-0.78311923</td><td>NA</td><td>...</td><td> 1.1260368</td><td> 0.89798480</td><td>NA </td><td>NA </td><td> 0.74063377</td><td>-1.53218644</td><td>-1.5858239</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>29</td><td>1999-2000</td><td> 1.2595176</td><td>         NA</td><td> 1.0910153</td><td> 0.8246504</td><td> 2.73654900</td><td> 3.57499225</td><td> 1.55242798</td><td>NA</td><td>...</td><td>-0.6951142</td><td> 0.17731817</td><td>Yes</td><td>NA </td><td>-0.22061790</td><td> 0.02234225</td><td> 0.4175646</td><td>1</td><td>0</td><td>1</td></tr>
	<tr><td>30</td><td>1999-2000</td><td>-0.9518008</td><td>         NA</td><td>-1.4330715</td><td>-0.7457118</td><td>         NA</td><td>         NA</td><td>-1.55585170</td><td>NA</td><td>...</td><td> 1.8516673</td><td> 0.56299342</td><td>NA </td><td>NA </td><td> 1.26022927</td><td>         NA</td><td>        NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td></td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>
	<tr><td>83702</td><td>2013-2014</td><td> 1.9832218</td><td>-0.65078964</td><td> 0.6681987</td><td> 0.45515338</td><td> 1.77459080</td><td> 0.94108712</td><td> 0.34595102</td><td>NA</td><td>...</td><td> 0.01661581</td><td>-0.10435859</td><td>No </td><td>NA</td><td>-0.06473925</td><td> 1.83595906</td><td> 1.682862654</td><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>83703</td><td>2013-2014</td><td>-0.3487140</td><td> 3.18136770</td><td> 1.2063289</td><td> 0.85104299</td><td>-0.14932559</td><td> 0.06311874</td><td> 1.32020786</td><td>NA</td><td>...</td><td>-0.95981204</td><td>-0.59770301</td><td>No </td><td>NA</td><td>-0.71423363</td><td>-0.10720180</td><td> 0.628447624</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83704</td><td>2013-2014</td><td>-0.6301545</td><td>         NA</td><td> 2.2185261</td><td> 1.12816572</td><td> 1.00502424</td><td>-0.11345590</td><td> 1.86739320</td><td>NA</td><td>...</td><td> 0.17822366</td><td> 0.53259424</td><td>NA </td><td>NA</td><td>-1.05197070</td><td>-4.25261166</td><td>-0.109642897</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83705</td><td>2013-2014</td><td> 0.1739613</td><td>         NA</td><td> 0.6041356</td><td> 0.66629451</td><td> 0.04306605</td><td> 0.58793783</td><td> 0.03899339</td><td>NA</td><td>...</td><td>-0.73717745</td><td>-0.56396742</td><td>Yes</td><td>NA</td><td> 0.55877535</td><td> 0.54051849</td><td>-0.847733419</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83706</td><td>2013-2014</td><td>-0.9920066</td><td>         NA</td><td>-1.5099472</td><td>-1.20758297</td><td>         NA</td><td>         NA</td><td>-1.45575682</td><td>NA</td><td>...</td><td>        NaN</td><td>        NaN</td><td>NA </td><td>NA</td><td> 1.07837084</td><td>         NA</td><td>          NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83707</td><td>2013-2014</td><td>-0.5095371</td><td> 4.22650152</td><td> 0.2581948</td><td> 0.16483434</td><td> 0.42784933</td><td>-0.28512570</td><td>-0.37473212</td><td>NA</td><td>...</td><td>-0.81515607</td><td>-0.97838911</td><td>NA </td><td>NA</td><td> 0.01320007</td><td>-0.75492209</td><td>-0.847733419</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83708</td><td>2013-2014</td><td> 1.3399291</td><td>-0.30241170</td><td> 2.5260290</td><td> 1.27332524</td><td> 1.19741588</td><td> 2.69211902</td><td> 3.21533759</td><td>NA</td><td>...</td><td> 0.41272457</td><td>-0.08888355</td><td>Yes</td><td>NA</td><td>-1.18186958</td><td> 0.15188631</td><td>-1.374940934</td><td>1</td><td>0</td><td>1</td></tr>
	<tr><td>83709</td><td>2013-2014</td><td>-0.2683024</td><td>-0.65078964</td><td> 0.3991336</td><td> 0.40236810</td><td> 0.62024097</td><td> 1.11275691</td><td>-0.13450440</td><td>NA</td><td>...</td><td> 0.48900799</td><td> 0.70189123</td><td>No </td><td>NA</td><td>-0.24659768</td><td> 0.02234225</td><td>-0.004201394</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83710</td><td>2013-2014</td><td>-1.1528298</td><td>         NA</td><td>-1.6124482</td><td>-2.10493274</td><td>         NA</td><td>         NA</td><td>-1.18883714</td><td>NA</td><td>...</td><td>-1.43672472</td><td>-1.16749416</td><td>NA </td><td>NA</td><td>         NA</td><td>         NA</td><td>          NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83711</td><td>2013-2014</td><td> 0.2945786</td><td>         NA</td><td> 0.5016346</td><td> 0.11204906</td><td> 0.81263261</td><td>-1.16309407</td><td> 1.09332613</td><td>NA</td><td>...</td><td>-1.12820064</td><td>-0.85706476</td><td>No </td><td>No</td><td>-0.66227408</td><td> 0.79960660</td><td>-0.320525903</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83712</td><td>2013-2014</td><td> 1.2193118</td><td>-0.65078964</td><td> 0.7066366</td><td> 1.15455836</td><td>-0.53410887</td><td> 0.06311874</td><td> 0.62621669</td><td>NA</td><td>...</td><td> 1.55132534</td><td>-0.07588451</td><td>No </td><td>NA</td><td>-1.38970777</td><td> 0.54051849</td><td> 0.523006121</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83713</td><td>2013-2014</td><td> 0.1337555</td><td>         NA</td><td>        NA</td><td>         NA</td><td>-0.91889215</td><td>-0.11345590</td><td>-0.21458031</td><td>NA</td><td>...</td><td> 0.04486893</td><td> 1.01974865</td><td>No </td><td>NA</td><td>-0.37649655</td><td> 0.54051849</td><td>-0.109642897</td><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>83714</td><td>2013-2014</td><td>-0.4693314</td><td>         NA</td><td> 0.4888220</td><td> 0.09885274</td><td>-0.14932559</td><td>-0.63827499</td><td> 0.37264299</td><td>NA</td><td>...</td><td>-0.84340919</td><td>-0.49463922</td><td>NA </td><td>NA</td><td>-1.38970777</td><td> 0.02234225</td><td>-0.425967406</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83715</td><td>2013-2014</td><td> 1.0986944</td><td>         NA</td><td> 0.7706997</td><td> 0.66629451</td><td> 0.62024097</td><td>-0.11345590</td><td> 0.13241528</td><td>NA</td><td>...</td><td> 0.81222369</td><td> 0.73748383</td><td>No </td><td>NA</td><td>-0.11669880</td><td> 0.79960660</td><td>-0.004201394</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83716</td><td>2013-2014</td><td>-0.5497429</td><td>         NA</td><td> 0.5144472</td><td> 0.69268715</td><td>-0.14932559</td><td>-0.11345590</td><td>-0.16119637</td><td>NA</td><td>...</td><td> 0.82126469</td><td> 0.31161061</td><td>NA </td><td>NA</td><td> 0.14309895</td><td>-0.49583398</td><td>-1.269499431</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83717</td><td>2013-2014</td><td> 1.9832218</td><td>         NA</td><td>-0.5361879</td><td> 0.04606746</td><td> 0.81263261</td><td> 1.28933156</td><td>-0.53488392</td><td>NA</td><td>...</td><td>-1.14458745</td><td>-0.84777974</td><td>No </td><td>NA</td><td> 0.14309895</td><td>         NA</td><td>          NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83718</td><td>2013-2014</td><td> 1.1791060</td><td>         NA</td><td> 0.3991336</td><td> 0.56072395</td><td> 0.42784933</td><td>-0.28512570</td><td> 0.27922110</td><td>NA</td><td>...</td><td>        NaN</td><td>        NaN</td><td>Yes</td><td>NA</td><td> 1.28620904</td><td>-0.10720180</td><td>-0.109642897</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83719</td><td>2013-2014</td><td>-1.1126240</td><td>         NA</td><td>        NA</td><td>         NA</td><td>         NA</td><td>         NA</td><td>-1.42906485</td><td>NA</td><td>...</td><td>        NaN</td><td>        NaN</td><td>NA </td><td>NA</td><td>         NA</td><td>         NA</td><td>          NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83720</td><td>2013-2014</td><td> 0.2141671</td><td>-0.65078964</td><td> 0.6169482</td><td> 1.10177308</td><td>-0.91889215</td><td>-0.46170034</td><td>-0.18788834</td><td>NA</td><td>...</td><td>        NaN</td><td>        NaN</td><td>No </td><td>NA</td><td> 1.41610792</td><td> 1.31778283</td><td> 0.312123115</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83721</td><td>2013-2014</td><td> 0.8574597</td><td>-0.65078964</td><td> 0.3606957</td><td> 0.74547243</td><td> 0.42784933</td><td>-0.11345590</td><td> 0.03899339</td><td>NA</td><td>...</td><td>-0.78916319</td><td>-0.74750145</td><td>No </td><td>NA</td><td> 0.14309895</td><td> 0.67006254</td><td>-0.531408909</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83722</td><td>2013-2014</td><td>-1.2332413</td><td>         NA</td><td>        NA</td><td>         NA</td><td>         NA</td><td>         NA</td><td>         NA</td><td>NA</td><td>...</td><td>-1.28528800</td><td>-1.73233328</td><td>NA </td><td>NA</td><td>         NA</td><td>         NA</td><td>          NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83723</td><td>2013-2014</td><td> 1.2193118</td><td>-0.30241170</td><td> 0.9757016</td><td> 0.93022091</td><td> 0.42784933</td><td> 0.76451247</td><td> 1.03994219</td><td>NA</td><td>...</td><td>-0.04158562</td><td> 0.62915852</td><td>No </td><td>NA</td><td>-0.24659768</td><td> 0.15188631</td><td> 1.050213636</td><td>1</td><td>0</td><td>0</td></tr>
	<tr><td>83724</td><td>2013-2014</td><td> 1.9832218</td><td>         NA</td><td> 0.3606957</td><td> 0.40236810</td><td> 1.77459080</td><td> 2.33896973</td><td>-0.05442850</td><td>NA</td><td>...</td><td>-0.09300630</td><td> 0.48957362</td><td>No </td><td>NA</td><td>-0.11669880</td><td> 0.02234225</td><td> 2.631836181</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83725</td><td>2013-2014</td><td>-0.9518008</td><td>         NA</td><td>-1.2793200</td><td>-0.94365656</td><td>         NA</td><td>         NA</td><td>-1.21552911</td><td>NA</td><td>...</td><td>        NaN</td><td>        NaN</td><td>NA </td><td>NA</td><td>-0.37649655</td><td>         NA</td><td>          NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83726</td><td>2013-2014</td><td> 0.3749902</td><td>         NA</td><td> 0.3606957</td><td> 0.79825771</td><td>         NA</td><td>         NA</td><td> 0.19914520</td><td>NA</td><td>...</td><td>        NaN</td><td>        NaN</td><td>No </td><td>NA</td><td>         NA</td><td>         NA</td><td>          NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83727</td><td>2013-2014</td><td>-0.1878908</td><td> 0.04596624</td><td> 0.2197569</td><td> 0.29679754</td><td> 1.38980752</td><td> 0.06311874</td><td>-0.10781244</td><td>NA</td><td>...</td><td> 2.93064270</td><td> 2.76100066</td><td>No </td><td>NA</td><td> 0.14309895</td><td> 0.67006254</td><td>-0.320525903</td><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>83728</td><td>2013-2014</td><td>-1.1528298</td><td>         NA</td><td>-1.7277618</td><td>-2.17091434</td><td>         NA</td><td>         NA</td><td>-1.25556706</td><td>NA</td><td>...</td><td>-1.00275679</td><td>-0.85985027</td><td>NA </td><td>NA</td><td>         NA</td><td>         NA</td><td>          NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83729</td><td>2013-2014</td><td> 0.4554018</td><td>         NA</td><td> 1.1294531</td><td> 0.61350923</td><td>-0.14932559</td><td>-0.46170034</td><td> 1.16005605</td><td>NA</td><td>...</td><td>-0.43260881</td><td> 0.26611397</td><td>No </td><td>NA</td><td>-0.32453700</td><td> 0.92915066</td><td> 1.050213636</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83730</td><td>2013-2014</td><td>-0.9518008</td><td>         NA</td><td>-1.1768190</td><td>-0.91726392</td><td>         NA</td><td>         NA</td><td>-1.22887509</td><td>NA</td><td>...</td><td>        NaN</td><td>        NaN</td><td>NA </td><td>NA</td><td>-0.11669880</td><td>         NA</td><td>          NA</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>83731</td><td>2013-2014</td><td>-0.7909777</td><td>         NA</td><td>-0.4080616</td><td>-0.16507367</td><td>         NA</td><td>         NA</td><td>-0.80180360</td><td>NA</td><td>...</td><td> 0.03243756</td><td>-0.01150833</td><td>NA </td><td>NA</td><td>         NA</td><td>-0.23674586</td><td>-1.480382437</td><td>0</td><td>0</td><td>0</td></tr>
</tbody>
</table>




```R
#jupyter nbconvert --to markdown R_replicate_Dinh.ipynb --output README.md
```

## 3. Model Development

### 3.1 Train/Test split

The paper do a 80/20 split to train the model, trying to keep the target class proportions of the NHANES population in train and test sets:

> Downsampling was used to produce a balanced 80/20 train/test split.


```R
print("% of cases of Diabetes:")
print(sum(df_formatted$Diabetes_Case_I) * 100 / nrow(df_formatted))

print("% of cases of Undiagnosed Diabetes:")
print(sum(df_formatted$Diabetes_Case_II) * 100 / nrow(df_formatted))

print("% of cases of CVD:")
print(sum(df_formatted$CVD) * 100 / nrow(df_formatted))
```

    [1] "% of cases of Diabetes:"
    [1] 7.150601
    [1] "% of cases of Undiagnosed Diabetes:"
    [1] 7.801099
    [1] "% of cases of CVD:"
    [1] 5.637646



```R
library(caret)

```


    Error in library(caret): there is no package called 'caret'
    Traceback:


    1. library(caret)


Models used in the paper:

- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosted Trees
- Ensemble model of the 5 models.   


```R

```


```R

```


```R

```
