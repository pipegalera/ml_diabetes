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
library(caret)
library(glue)
library(recipes)
library(zeallot)
library(pROC)


SEED <- 4208
DATA_PATH  <- "/Users/pipegalera/dev/ml_diabetes/data/raw_data/NHANES/"
DINH_DOCS_PATH <- "/Users/pipegalera/dev/ml_diabetes/data/processed/NHANES/"
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
df <- read_parquet(paste0(DATA_PATH, "dinh_raw_data.parquet"))
```


```R
head(df)
```


<table class="dataframe">
<caption>A tibble: 6 x 78</caption>
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
<ol class=list-inline><li>'SEQN'</li><li>'YEAR'</li><li>'RIDAGEYR'</li><li>'ALQ130'</li><li>'DRXTALCO'</li><li>'DR1TALCO'</li><li>'DR2TALCO'</li><li>'BMXARMC'</li><li>'BMXARML'</li><li>'LBXSOSSI'</li><li>'MCQ250A'</li><li>'LBDSBUSI'</li><li>'BMXBMI'</li><li>'DRXTCAFF'</li><li>'DR1TCAFF'</li><li>'DR2TCAFF'</li><li>'DR1TCALC'</li><li>'DR2TCALC'</li><li>'DRXTCALC'</li><li>'DR1TCARB'</li><li>'DR2TCARB'</li><li>'DRXTCARB'</li><li>'LBXSNASI'</li><li>'LBXSCLSI'</li><li>'MCQ300c'</li><li>'MCQ300C'</li><li>'BPXDI1'</li><li>'BPXDI4'</li><li>'BPXDI2'</li><li>'BPXDI3'</li><li>'RIDRETH1'</li><li>'DR1TFIBE'</li><li>'DR2TFIBE'</li><li>'DRXTFIBE'</li><li>'LBXSGTSI'</li><li>'HSD010'</li><li>'HUQ010'</li><li>'LBDHDLSI'</li><li>'LBDHDDSI'</li><li>'BMXHT'</li><li>'BPQ080'</li><li>'INDHHIN2'</li><li>'DRXTKCAL'</li><li>'DR1TKCAL'</li><li>'DR2TKCAL'</li><li>'LBDLDLSI'</li><li>'BMXLEG'</li><li>'LBDLYMNO'</li><li>'LBXMCVSI'</li><li>'BPXPLS'</li><li>'WHD140'</li><li>'DR1TSODI'</li><li>'DR2TSODI'</li><li>'DRDTSODI'</li><li>'BPXSY1'</li><li>'BPXSY4'</li><li>'BPXSY2'</li><li>'BPXSY3'</li><li>'LBDTCSI'</li><li>'LBDSTRSI'</li><li>'BMXWAIST'</li><li>'BMXWT'</li><li>'LBXWBCSI'</li><li>'LBXSASSI'</li><li>'LBXGLUSI'</li><li>'LBDGLUSI'</li><li>'RHQ141'</li><li>'RHD143'</li><li>'SEQ060'</li><li>'DIQ010'</li><li>'MCQ160B'</li><li>'MCQ160b'</li><li>'MCQ160C'</li><li>'MCQ160c'</li><li>'MCQ160E'</li><li>'MCQ160e'</li><li>'MCQ160F'</li><li>'MCQ160f'</li></ol>



## 2. Pre-processing and Data modeling


### 2.1 Extreme values and replacing Missing/Don't know answers

> The preprocessing stage also converted any undecipherable values (errors in datatypes and standard formatting) from the database to null representations.

For this, I've checked the variables according to their possible values in the NHANES documentation (https://wwwn.cdc.gov/nchs/nhanes/search/default.aspx). I did not found any any extreme value out of the possible ranges. However, the data is reviwed and updated after the survey, so it might be that the NCHS applied some fixes after they saw them. 


I have replaced "Don't know" and "Refused" for NA values and converted the intial encoding of the categorical variables to the real values in the survey - given that the encoding is not consistent accross years. For the model, I will encode the variables myself so I don't have to jungle NHANES encoding. 

All the variables can by found at  https://wwwn.cdc.gov/nchs/nhanes/search/default.aspx


```R
# Refused or Don"t know for NA
df <- df |>
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

df <- df |>
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
#unique(df$YEAR[!is.na(df$Relative_Had_Diabetes)])
```

### 2.3 Choosing between different readings in Blood analysis 

[From NHANES](https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BPX_H.htm): 

> After resting quietly in a seated position for 5 minutes and once the participants maximum inflation level (MIL) has been determined, three consecutive blood pressure readings are obtained. If a blood pressure measurement is interrupted or incomplete, a fourth attempt may be made. All BP determinations (systolic and diastolic) are taken in the mobile examination center (MEC). 

In Dinh et al. (2019) the authors do not say which readings are taking, but I'm assuming they take the last one to avoid the [white coat syndrom](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5352963/) and for data consistency.


```R
df <- df |>
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
df <- df |> 
  filter(is.na(Pregnant) | Pregnant != "Yes") |> 
  filter(RIDAGEYR >= 20)  |> 
  select(-c(Pregnant))
```


```R
nrow(df)
```


42655


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
df <- df |>
  mutate(
    # Diabetic or not diabetic
    Diabetes_Case_I = case_when(
      (Glucose > 7.0 | DIQ010 == "Yes") ~ "Yes",
      TRUE ~ "No"),
    Diabetes_Case_II = case_when(
      # Undiagnosed Diabetic 
      (Diabetes_Case_I == "No" & Glucose > 7.0 & DIQ010 == "No") ~ "Yes",
      # Prediabetic
      (Diabetes_Case_I == "No" & Glucose >= 5.6 & Glucose < 7.0) ~ "Yes",
      TRUE ~  "No"),
    # Cardiovascular Disease
    CVD = case_when(
      (Told_CHF == "Yes" | Told_CHD == "Yes" | Told_HA == "Yes" | Told_stroke == "Yes") ~ "Yes",
      TRUE ~  "No")
  )  |>
  select(-c(Told_CHF, Told_CHD, Told_HA, Told_stroke, Glucose, DIQ010))
```

### 2.6 Column name formatting


```R
df <- df |>
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
    Chloride = LBXSCLSI,
    Sodium = LBXSNASI,
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
    Self-reported_greatest_weight = WHD140,
    Survey_year = YEAR,
  )
```


```R
colnames(df)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'SEQN'</li><li>'Survey_year'</li><li>'Age'</li><li>'Alcohol_consumption'</li><li>'Arm_circumference'</li><li>'Arm_length'</li><li>'Osmolality'</li><li>'Blood_urea_nitrogen'</li><li>'Body_mass_index'</li><li>'Sodium'</li><li>'Chloride'</li><li>'Race_ethnicity'</li><li>'Gamma_glutamyl_transferase'</li><li>'General_health'</li><li>'Health_status'</li><li>'Height'</li><li>'Told_High_Cholesterol'</li><li>'Household_income'</li><li>'LDL_cholesterol'</li><li>'Leg_length'</li><li>'Lymphocytes'</li><li>'Mean_cell_volume'</li><li>'Pulse'</li><li>'Self-reported_greatest_weight'</li><li>'Total_cholesterol'</li><li>'Triglycerides'</li><li>'Waist_circumference'</li><li>'Weight'</li><li>'White_blood_cell_count'</li><li>'Aspartate_aminotransferase_AST'</li><li>'Alcohol_Intake'</li><li>'Caffeine_Intake'</li><li>'Calcium_Intake'</li><li>'Carbohydrate_Intake'</li><li>'Fiber_Intake'</li><li>'Kcal_Intake'</li><li>'Sodium_Intake'</li><li>'Relative_Had_Diabetes'</li><li>'HDL_Cholesterol'</li><li>'Diastolic_Blood_Pressure'</li><li>'Systolic_Blood_Pressure'</li><li>'Diabetes_Case_I'</li><li>'Diabetes_Case_II'</li><li>'CVD'</li></ol>



## 3. Model Development

The training procedure is not very clear from the paper, but guessing from the context what it would make sense is:

1. Split data into train (80%) and test (20%) sets.
2. Hyper-tunning using 10-fold CV random grid search on the training set for every model (e.g. via `RandomizedSearchCV(cv=10)`).
3. Train the final model on the entire training set using the best hyperparameters found.
4. Evaluate the final model on the test set.

### 3.1 Train/Test split

The paper do a 80/20 split to train the model, trying to keep the target class proportions of the NHANES population in train and test sets:

> Downsampling was used to produce a balanced 80/20 train/test split.

First, I created a split based on the interaction of all the targets to keep the balance of the target labels. Afterwards, I created a quick R function to check it:


```R
set.seed(SEED)

df$strata <- interaction(df$Diabetes_Case_I, df$Diabetes_Case_II, df$CVD)
split <- createDataPartition(df$strata, p = 0.8, list = FALSE)

train_data <- df[split, ]
test_data <- df[-split, ]

df$strata <- NULL
train_data$strata <- NULL
test_data$strata <- NULL
```

    Warning message in createDataPartition(df$strata, p = 0.8, list = FALSE):
    "Some classes have no records ( Yes.Yes.No, Yes.Yes.Yes ) and these will be ignored"



```R
percent_target <- function(data, target) {
    df_name <- deparse(substitute(data))
    glue::glue("In {df_name}, percentage of cases of {target} is {round(sum(data[[target]]== 'Yes') / nrow(data) * 100, 2)}%")
}

percent_target(df, "Diabetes_Case_I")
percent_target(train_data, "Diabetes_Case_I")
percent_target(test_data, "Diabetes_Case_I")

percent_target(df, "Diabetes_Case_II")
percent_target(train_data, "Diabetes_Case_II")
percent_target(test_data, "Diabetes_Case_II")

percent_target(df, "CVD")
percent_target(train_data, "CVD")
percent_target(test_data, "CVD")

```


'In df, percentage of cases of Diabetes_Case_I is 13.43%'



'In train_data, percentage of cases of Diabetes_Case_I is 13.44%'



'In test_data, percentage of cases of Diabetes_Case_I is 13.42%'



'In df, percentage of cases of Diabetes_Case_II is 13.08%'



'In train_data, percentage of cases of Diabetes_Case_II is 13.08%'



'In test_data, percentage of cases of Diabetes_Case_II is 13.07%'



'In df, percentage of cases of CVD is 10.84%'



'In train_data, percentage of cases of CVD is 10.85%'



'In test_data, percentage of cases of CVD is 10.83%'


### 3.2 RandomSearchCV in R

> For each model, a grid-search approach with parallelized performance evaluation for model parameter tuning was used to generate the best model parameters. Next, each of the models underwent a 10-fold cross-validation (10 folds of training and testing with randomized data-split)

It doesn't make so much sense to do grid-search on the entire training set before a cross-validation. The reason is that you would basically train the data before the validation set that you are using to check if the model generalize well. And then the whole purpose of the cross-validation is to check model fit outside of the training set - it would lose all purpose. 

My best guess is that they used `GridSearchCV/RandomSearchCV` from sklearn and they assumed that the grid search go first. 


The preprocessing also includes standarization, made also by the authors: 

> Normalization was performed on the data using the following standardization model: x' = x−x^/σ 



```R
# Categorical variables
categorical_vars <- c(
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
  'Sodium',
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
```


```R
#target_vars <- c("Diabetes_Case_I", "Diabetes_Case_II", "CVD")

data_ml_pipeline <- function(data,
                            numerical_vars, 
                            categorical_vars) {
    # Target 
    y <-  as.factor(data[[target]])

    # Standarization and encoding of categorical variables.
    X <- data |> select(all_of(c(numerical_vars, categorical_vars)))

    rec <- recipe(~ ., data = X) |>
      step_unknown(all_of(categorical_vars)) |> # assign a missing value in a factor level to "unknown".
      step_mutate_at(all_of(categorical_vars), fn = as.factor) |>
      step_integer(all_of(categorical_vars)) |>
      step_normalize(all_of(numerical_vars))

    prep_rec <- prep(rec)
    X_processed <- bake(prep_rec, new_data = X)
    X_processed <- as.data.frame(X_processed)

    # 10F CV with Random Search (caret: https://topepo.github.io/caret/random-hyperparameter-search.html)
    control_cv <- trainControl(
        method = "cv",
        number = 10,
        search = "random",
        classProbs = TRUE,
        summaryFunction = twoClassSummary
      )
    return(list(X, y, control_cv))
}
```


```R
c(X_train, y_train, control_cv) %<-% data_ml_pipeline(data=train_data, 
                                          numerical_vars=numerical_vars, 
                                          categorical_vars=categorical_vars)

c(X_test, y_test, control_cv) %<-% data_ml_pipeline(data=test_data, 
                                          numerical_vars=numerical_vars, 
                                          categorical_vars=categorical_vars)
```

### 3.3 Predictive Models

Models used in the paper:

- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosted Trees
- Ensemble model of the 5 models.  


```R
set.seed(SEED)

# Model 1 : Logistic Regression
lr <- train(x = X_train,
            y = y_train,
            method="glm",
            family="binomial",
            metric="AUC",
            trControl=control_cv,
            tuneLength=1
                      )


```

    Warning message in train.default(x = X_train, y = y_train, method = "glm", family = "binomial", :
    "The metric "AUC" was not in the result set. ROC will be used instead."
    Warning message:
    "Setting row names on a tibble is deprecated."
    Warning message:
    "Setting row names on a tibble is deprecated."
    Warning message:
    "glm.fit: fitted probabilities numerically 0 or 1 occurred"
    Warning message:
    "Setting row names on a tibble is deprecated."
    Warning message:
    "glm.fit: fitted probabilities numerically 0 or 1 occurred"
    Warning message:
    "Setting row names on a tibble is deprecated."
    Warning message:
    "glm.fit: fitted probabilities numerically 0 or 1 occurred"
    Warning message:
    "Setting row names on a tibble is deprecated."
    Warning message:
    "glm.fit: fitted probabilities numerically 0 or 1 occurred"
    Warning message:
    "Setting row names on a tibble is deprecated."
    Warning message:
    "glm.fit: fitted probabilities numerically 0 or 1 occurred"
    Warning message:
    "Setting row names on a tibble is deprecated."
    Warning message:
    "glm.fit: fitted probabilities numerically 0 or 1 occurred"
    Warning message:
    "Setting row names on a tibble is deprecated."
    Warning message:
    "glm.fit: fitted probabilities numerically 0 or 1 occurred"
    Warning message:
    "Setting row names on a tibble is deprecated."
    Warning message:
    "glm.fit: fitted probabilities numerically 0 or 1 occurred"
    Warning message:
    "Setting row names on a tibble is deprecated."
    Warning message:
    "glm.fit: fitted probabilities numerically 0 or 1 occurred"
    Warning message:
    "Setting row names on a tibble is deprecated."
    Warning message:
    "glm.fit: fitted probabilities numerically 0 or 1 occurred"


## Model Comparison


```R
model_auc <- function(model, X_data, y_data) {

    # AUC 
    predictions <- predict(model, newdata = X_data, type = "prob")
    roc_score <- roc(y_data, predictions[, "Yes"], quiet = TRUE)
    auc_score <- round(auc(roc_score), 3)

    # Message 
    data_name <- deparse(substitute(X_data))
    glue("AUC score for Logistic Regression on {data_name}: {auc_score} ")
}

model_auc(lr, X_test, y_test)

```


'AUC score for Logistic Regression on X_test: 0.959 '



```R
#jupyter nbconvert --to markdown R_replicate_Dinh.ipynb --output README.md
```
