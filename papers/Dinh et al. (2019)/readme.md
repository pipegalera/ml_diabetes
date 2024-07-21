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

    Some features are not enabled in this build of Arrow. Run `arrow_info()` for more information.
    
    The repository you retrieved Arrow from did not include all of Arrow's features.
    You can install a fully-featured version by running:
    `install.packages('arrow', repos = 'https://apache.r-universe.dev')`.
    
    
    Attaching package: 'arrow'
    
    
    The following object is masked from 'package:utils':
    
        timestamp
    
    
    
    Attaching package: 'dplyr'
    
    
    The following objects are masked from 'package:stats':
    
        filter, lag
    
    
    The following objects are masked from 'package:base':
    
        intersect, setdiff, setequal, union
    
    


## 1. HNANES data

URL: https://www.cdc.gov/nchs/index.htm


## Target

From the paper, the definitions are clear: 

![Dinh et al.(2019), Table 4](https://raw.githubusercontent.com/pipegalera/ml_diabetes/main/images/dinh_2019_Table4.png)

- Case I: Diabetes.

    - Glucose >= 126 mg/dL. OR;
    - "Yes" to the question "Have you ever been told by a doctor that you have diabetes?"

- Case II: Undiagnosed Diabetes. 

    - Glucose >= 126 mg/dL. AND;
    - "No" to the question "Have you ever been told by a doctor that you have diabetes?"

- Cardio: Cardiovascular disease.

    - "Yes" to any of the the questions "Have you ever been told by a doctor that you had congestive heart failure, coronary heart disease, a heart attack, or a stroke?"

The paper also defined and test for the target: 

- Pre diabetes

    - Glucose 125 >= 100 mg/dL

## Covariates

The paper did not say what variables they use from NHANES. I emailed the author in the correspondence section of the paper to try to get the list of variables they used, but no answer from him yet.

Given that NHANES have more than 3000 variables, I cannot just randomly take the variables I believe are important. 

For now, I will consider the variables taken from [Figure 5](https://raw.githubusercontent.com/pipegalera/ml_diabetes/main/images/dinh_2019_Fig5.png) and [Figure 6](https://raw.githubusercontent.com/pipegalera/ml_diabetes/main/images/dinh_2019_Fig6.png) of the paper. I compiled them by hand in an Excel file using NHANES search tool for variables:




```R
DATA_PATH <- "/Users/pipegalera/dev/ml_diabetes/data/NHANES/raw_data/"
dinh_2019_vars <- read_excel(paste0(DATA_PATH, "dinh_2019_variables_doc.xlsx"))

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



For the complete list (n=62), check the file `dinh_2019_variables_doc.xlsx` under NHANES data folder.

NHANES data is made by multiple files (see `NHANES` unde data folder) that have to be compiled together. The data was downloaded automatically via script, all the files converted from SAS to parquet, and the files were stacked and merged based on the individual index ("SEQN"). For more details please check the `nhanes_data_backfill` notebook. 

Plese notice that no transformation are made to the covariates, the files were only arranged and stacked together. 


```R
df <- read_parquet(paste0(DATA_PATH, "dinh_raw_data.parquet"))
```


```R
head(df)
```


<table class="dataframe">
<caption>A tibble: 6 x 64</caption>
<thead>
	<tr><th scope=col>SEQN</th><th scope=col>YEAR</th><th scope=col>RIDAGEYR</th><th scope=col>ALQ130</th><th scope=col>DRXTALCO</th><th scope=col>DR1TALCO</th><th scope=col>DR2TALCO</th><th scope=col>BMXARMC</th><th scope=col>BMXARML</th><th scope=col>LBXSOSSI</th><th scope=col>...</th><th scope=col>BPXSY4</th><th scope=col>BPXSY2</th><th scope=col>BPXSY3</th><th scope=col>LBDTCSI</th><th scope=col>LBDSTRSI</th><th scope=col>BMXWAIST</th><th scope=col>BMXWT</th><th scope=col>LBXWBCSI</th><th scope=col>LBXSASSI</th><th scope=col>RHD143</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>...</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>1999-2000</td><td> 2</td><td>NA</td><td> 0.00</td><td>NA</td><td>NA</td><td>15.2</td><td>18.6</td><td> NA</td><td>...</td><td>NA</td><td> NA</td><td> NA</td><td>  NA</td><td>   NA</td><td>45.7</td><td>12.5</td><td> NA</td><td>NA</td><td>NA</td></tr>
	<tr><td>2</td><td>1999-2000</td><td>77</td><td> 1</td><td> 0.00</td><td>NA</td><td>NA</td><td>29.8</td><td>38.2</td><td>288</td><td>...</td><td>NA</td><td> 98</td><td> 98</td><td>5.56</td><td>1.298</td><td>98.0</td><td>75.4</td><td>7.6</td><td>19</td><td>NA</td></tr>
	<tr><td>3</td><td>1999-2000</td><td>10</td><td>NA</td><td> 0.00</td><td>NA</td><td>NA</td><td>19.7</td><td>25.5</td><td> NA</td><td>...</td><td>NA</td><td>104</td><td>112</td><td>3.34</td><td>   NA</td><td>64.7</td><td>32.9</td><td>7.5</td><td>NA</td><td>NA</td></tr>
	<tr><td>4</td><td>1999-2000</td><td> 1</td><td>NA</td><td> 0.00</td><td>NA</td><td>NA</td><td>16.4</td><td>20.4</td><td> NA</td><td>...</td><td>NA</td><td> NA</td><td> NA</td><td>  NA</td><td>   NA</td><td>  NA</td><td>13.3</td><td>8.8</td><td>NA</td><td>NA</td></tr>
	<tr><td>5</td><td>1999-2000</td><td>49</td><td> 3</td><td>34.56</td><td>NA</td><td>NA</td><td>35.8</td><td>39.7</td><td>276</td><td>...</td><td>NA</td><td>122</td><td>122</td><td>7.21</td><td>3.850</td><td>99.9</td><td>92.5</td><td>5.9</td><td>22</td><td>NA</td></tr>
	<tr><td>6</td><td>1999-2000</td><td>19</td><td>NA</td><td> 0.00</td><td>NA</td><td>NA</td><td>26.0</td><td>34.5</td><td>277</td><td>...</td><td>NA</td><td>116</td><td>112</td><td>3.96</td><td>0.553</td><td>81.6</td><td>59.2</td><td>9.6</td><td>20</td><td>NA</td></tr>
</tbody>
</table>




```R
tail(df)
```


<table class="dataframe">
<caption>A tibble: 6 x 64</caption>
<thead>
	<tr><th scope=col>SEQN</th><th scope=col>YEAR</th><th scope=col>RIDAGEYR</th><th scope=col>ALQ130</th><th scope=col>DRXTALCO</th><th scope=col>DR1TALCO</th><th scope=col>DR2TALCO</th><th scope=col>BMXARMC</th><th scope=col>BMXARML</th><th scope=col>LBXSOSSI</th><th scope=col>...</th><th scope=col>BPXSY4</th><th scope=col>BPXSY2</th><th scope=col>BPXSY3</th><th scope=col>LBDTCSI</th><th scope=col>LBDSTRSI</th><th scope=col>BMXWAIST</th><th scope=col>BMXWT</th><th scope=col>LBXWBCSI</th><th scope=col>LBXSASSI</th><th scope=col>RHD143</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>...</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>83726</td><td>2013-2014</td><td>40</td><td>NA</td><td>NA</td><td>NA</td><td>  NA</td><td>31.0</td><td>39.0</td><td> NA</td><td>...</td><td>NA</td><td> NA</td><td> NA</td><td>  NA</td><td>   NA</td><td> 97.7</td><td>79.0</td><td> NA</td><td>NA</td><td>NA</td></tr>
	<tr><td>83727</td><td>2013-2014</td><td>26</td><td> 3</td><td>NA</td><td>14</td><td>19.9</td><td>29.9</td><td>35.2</td><td>285</td><td>...</td><td>NA</td><td>116</td><td>112</td><td>4.91</td><td>0.858</td><td> 87.1</td><td>71.8</td><td>5.1</td><td>27</td><td>NA</td></tr>
	<tr><td>83728</td><td>2013-2014</td><td> 2</td><td>NA</td><td>NA</td><td> 0</td><td> 0.0</td><td>14.7</td><td>16.5</td><td> NA</td><td>...</td><td>NA</td><td> NA</td><td> NA</td><td>  NA</td><td>   NA</td><td> 47.2</td><td>11.3</td><td>6.6</td><td>NA</td><td>NA</td></tr>
	<tr><td>83729</td><td>2013-2014</td><td>42</td><td>NA</td><td>NA</td><td> 0</td><td> 0.0</td><td>37.0</td><td>37.6</td><td>277</td><td>...</td><td>NA</td><td>130</td><td>138</td><td>3.93</td><td>1.197</td><td>102.7</td><td>89.6</td><td>6.4</td><td>26</td><td>NA</td></tr>
	<tr><td>83730</td><td>2013-2014</td><td> 7</td><td>NA</td><td>NA</td><td>NA</td><td>  NA</td><td>19.0</td><td>26.0</td><td> NA</td><td>...</td><td>NA</td><td> NA</td><td> NA</td><td>4.32</td><td>   NA</td><td> 53.0</td><td>22.8</td><td>9.9</td><td>NA</td><td>NA</td></tr>
	<tr><td>83731</td><td>2013-2014</td><td>11</td><td>NA</td><td>NA</td><td> 0</td><td> 0.0</td><td>25.0</td><td>31.7</td><td> NA</td><td>...</td><td>NA</td><td> 94</td><td> 90</td><td>  NA</td><td>   NA</td><td> 73.5</td><td>42.3</td><td> NA</td><td>NA</td><td>NA</td></tr>
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
<ol class=list-inline><li>'SEQN'</li><li>'YEAR'</li><li>'RIDAGEYR'</li><li>'ALQ130'</li><li>'DRXTALCO'</li><li>'DR1TALCO'</li><li>'DR2TALCO'</li><li>'BMXARMC'</li><li>'BMXARML'</li><li>'LBXSOSSI'</li><li>'MCQ250A'</li><li>'LBDSBUSI'</li><li>'BMXBMI'</li><li>'DRXTCAFF'</li><li>'DR1TCAFF'</li><li>'DR2TCAFF'</li><li>'DR1TCALC'</li><li>'DR2TCALC'</li><li>'DRXTCALC'</li><li>'DR1TCARB'</li><li>'DR2TCARB'</li><li>'DRXTCARB'</li><li>'LB2SCLSI'</li><li>'MCQ300c'</li><li>'MCQ300C'</li><li>'BPXDI1'</li><li>'BPXDI4'</li><li>'BPXDI2'</li><li>'BPXDI3'</li><li>'RIDRETH1'</li><li>'DR1TFIBE'</li><li>'DR2TFIBE'</li><li>'DRXTFIBE'</li><li>'LBXSGTSI'</li><li>'HSD010'</li><li>'HUQ010'</li><li>'LBDHDLSI'</li><li>'LBDHDDSI'</li><li>'BMXHT'</li><li>'BPQ080'</li><li>'INDHHIN2'</li><li>'DRXTKCAL'</li><li>'DR1TKCAL'</li><li>'DR2TKCAL'</li><li>'LBDLDLSI'</li><li>'BMXLEG'</li><li>'LBDLYMNO'</li><li>'LBXMCVSI'</li><li>'BPXPLS'</li><li>'WHD140'</li><li>'DR1TSODI'</li><li>'DR2TSODI'</li><li>'DRDTSODI'</li><li>'BPXSY1'</li><li>'BPXSY4'</li><li>'BPXSY2'</li><li>'BPXSY3'</li><li>'LBDTCSI'</li><li>'LBDSTRSI'</li><li>'BMXWAIST'</li><li>'BMXWT'</li><li>'LBXWBCSI'</li><li>'LBXSASSI'</li><li>'RHD143'</li></ol>



# 2. Pre-processing

There are some fixes before the data is ready for analysis. 


## 2.1 Homogenize variables that are the same but are called diffrent in different NHANES years

1. Intake variables went from 1 day in 1999 to 2001 to 2 days from 2003 on, therefore the variable has to be homogenized. Dinh et al. (2019) do not specify which examination records the authors, but my best guess is that they problably took the average of both days that the examination was performed. 

This situation happends with:

- Alcohol intake (`DRXTALCO`, `DR1TALCO`, `DR2TALCO`)
- Caffeine intake (`DRXTCAFF`, `DR1TCAFF`, `DR2TCAFF`)
- Calcium intake (`DRXTCALC`, `DR1TCALC`, `DR2TCALC`)
- Carbohydrate intake (`DRXTCARB`, `DR1TCARB`, `DR2TCARB`)
- Fiber intake (`DRXTFIBE`, `DR1TFIBE`, `DR2TFIBE`)
- Kcal intake (`DRXTKCAL`, `DR1TKCAL`, `DR2TKCAL`)
- Sodium intake (`DRDTSODI`, `DR1TSODI`, `DR2TSODI`)


2. Also, small changes in same quesion format are registered with different codes. Examples: 

    - `MCQ250A`, and `MCQ300C`
    - `LBDHDDSI` and `LBDHDLSI`.


```R
# DRXTALCO only in 1999-2002
unique(df$YEAR[!is.na(df$DRXTALCO)])

```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'1999-2000'</li><li>'2001-2002'</li></ol>




```R
# DRXTALCO replaced to DR1TALCO and DR2TALCO 2003 onwards due to new procedure.
unique(df$YEAR[!is.na(df$DR1TALCO)])

```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'2003-2004'</li><li>'2005-2006'</li><li>'2007-2008'</li><li>'2009-2010'</li><li>'2011-2012'</li><li>'2013-2014'</li></ol>




```R
# Similar questions (or the same) with different NHANES variable codes
var_docs <- read_excel(paste0(DATA_PATH, "dinh_2019_variables_doc.xlsx"))
var_docs |> 
  filter(`NHANES Name` %in% c('MCQ250A', 'MCQ300C', 'MCQ300c', 'LBDHDDSI', 'LBDHDLSI'))
```


<table class="dataframe">
<caption>A tibble: 5 x 5</caption>
<thead>
	<tr><th scope=col>Variable Name</th><th scope=col>NHANES Name</th><th scope=col>NHANES File</th><th scope=col>NHANES Type of data</th><th scope=col>Variable Definition</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Blood relatives have diabetes</td><td>MCQ250A </td><td>MCQ              </td><td>Questionnaire</td><td>Including living and deceased, were any of {SP's/ your} biological that is, blood relatives including grandparents, parents, brothers, sisters ever told by a health professional that they had . . .diabetes?</td></tr>
	<tr><td>Close relative had diabetes  </td><td>MCQ300c </td><td>MCQ              </td><td>Questionnaire</td><td>Including living and deceased, were any of {SP's/your} close biological that is, blood relatives including father, mother, sisters or brothers, ever told by a health professional that they had diabetes?    </td></tr>
	<tr><td>Close relative had diabetes  </td><td>MCQ300C </td><td>MCQ              </td><td>Questionnaire</td><td>Including living and deceased, were any of {SP's/your} close biological that is, blood relatives including father, mother, sisters or brothers, ever told by a health professional that they had diabetes?    </td></tr>
	<tr><td>HDL-cholesterol              </td><td>LBDHDLSI</td><td>Lab13, l13_b, HDL</td><td>Laboratory   </td><td>HDL-cholesterol (mmol/L)                                                                                                                                                                                      </td></tr>
	<tr><td>HDL-cholesterol              </td><td>LBDHDDSI</td><td>Lab13, l13_b, HDL</td><td>Laboratory   </td><td>HDL-cholesterol (mmol/L)                                                                                                                                                                                      </td></tr>
</tbody>
</table>




```R
unique(df$YEAR[!is.na(df$MCQ250A)])

```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'1999-2000'</li><li>'2001-2002'</li><li>'2003-2004'</li></ol>




```R
unique(df$YEAR[!is.na(df$MCQ300C)])

```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'2005-2006'</li><li>'2007-2008'</li><li>'2009-2010'</li></ol>




```R
unique(df$YEAR[!is.na(df$MCQ300c)])
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'2011-2012'</li><li>'2013-2014'</li></ol>




```R
unique(df$YEAR[!is.na(df$LBDHDLSI)])
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'1999-2000'</li><li>'2001-2002'</li></ol>




```R
unique(df$YEAR[!is.na(df$LBDHDDSI)])
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'2003-2004'</li><li>'2005-2006'</li><li>'2007-2008'</li><li>'2009-2010'</li><li>'2011-2012'</li><li>'2013-2014'</li></ol>




```R
create_intake_new_column <- function(df, day0_col, day1_col, day2_col) {
    ifelse(is.na(df[[day0_col]]), 
           rowMeans(df[, c(day1_col, day2_col)], na.rm = TRUE), 
           df[[day0_col]])
}

df_formated <- df |>
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
    # HDL-cholesterol
    HDL_cholesterol = coalesce(LBDHDLSI, LBDHDDSI)
   ) |>
# Delete old columns that are not needed
  select(-c(DRXTALCO, DR1TALCO, DR2TALCO, DRXTCAFF, DR1TCAFF, DR2TCAFF,
            DRXTCALC, DR1TCALC, DR2TCALC, DRXTCARB, DR1TCARB, DR2TCARB,
            DRXTFIBE, DR1TFIBE, DR2TFIBE, DRXTKCAL, DR1TKCAL, DR2TKCAL,
            DRDTSODI, DR1TSODI, DR2TSODI, MCQ250A, MCQ300C, MCQ300c, LBDHDLSI, LBDHDDSI)
            )
```


```R
unique(df_formated$YEAR[!is.na(df_formated$Relative_Had_Diabetes)])

```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'1999-2000'</li><li>'2001-2002'</li><li>'2003-2004'</li><li>'2005-2006'</li><li>'2007-2008'</li><li>'2009-2010'</li><li>'2011-2012'</li><li>'2013-2014'</li></ol>



## 2.2 Discretional trimming of the data according to the authors

> In our study, all datasets were limited to non-pregnant subjects and adults of at least twenty years of age.


```R
df_formated <- df_formated |> 
  filter(RHD143 == 2) |>  # Are you pregnant now? = "No"
  filter(RIDAGEYR >= 20) 
```

> The preprocessing stage also converted any undecipherable values (errors in datatypes and standard formatting) from the database to null representations.

For this, I've checked the variables according to their possible values in the NHANES documentation


```R
colnames(df_formated)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'SEQN'</li><li>'YEAR'</li><li>'RIDAGEYR'</li><li>'ALQ130'</li><li>'BMXARMC'</li><li>'BMXARML'</li><li>'LBXSOSSI'</li><li>'LBDSBUSI'</li><li>'BMXBMI'</li><li>'LB2SCLSI'</li><li>'BPXDI1'</li><li>'BPXDI4'</li><li>'BPXDI2'</li><li>'BPXDI3'</li><li>'RIDRETH1'</li><li>'LBXSGTSI'</li><li>'HSD010'</li><li>'HUQ010'</li><li>'BMXHT'</li><li>'BPQ080'</li><li>'INDHHIN2'</li><li>'LBDLDLSI'</li><li>'BMXLEG'</li><li>'LBDLYMNO'</li><li>'LBXMCVSI'</li><li>'BPXPLS'</li><li>'WHD140'</li><li>'BPXSY1'</li><li>'BPXSY4'</li><li>'BPXSY2'</li><li>'BPXSY3'</li><li>'LBDTCSI'</li><li>'LBDSTRSI'</li><li>'BMXWAIST'</li><li>'BMXWT'</li><li>'LBXWBCSI'</li><li>'LBXSASSI'</li><li>'RHD143'</li><li>'Alcohol_Intake'</li><li>'Caffeine_Intake'</li><li>'Calcium_Intake'</li><li>'Carbohydrate_Intake'</li><li>'Fiber_Intake'</li><li>'Kcal_Intake'</li><li>'Sodium_Intake'</li><li>'Relative_Had_Diabetes'</li><li>'HDL_cholesterol'</li></ol>




```R
unique(df_formated$ALQ130)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>4</li><li>2</li><li>1</li><li>3</li><li>10</li><li>&lt;NA&gt;</li><li>6</li><li>12</li><li>5</li><li>7</li><li>50</li><li>8</li><li>9</li><li>20</li><li>999</li><li>15</li><li>16</li><li>17</li><li>23</li><li>777</li><li>13</li><li>18</li></ol>




```R
df_formated %>%
  filter(RIDAGEYR < 20 & RIDAGEYR > 80) |> 
  


```

    Warning message in cbind(parts$left, chars$ellip_h, parts$right, deparse.level = 0L):
    "number of rows of result is not a multiple of vector length (arg 2)"
    Warning message in cbind(parts$left, chars$ellip_h, parts$right, deparse.level = 0L):
    "number of rows of result is not a multiple of vector length (arg 2)"
    Warning message in cbind(parts$left, chars$ellip_h, parts$right, deparse.level = 0L):
    "number of rows of result is not a multiple of vector length (arg 2)"
    Warning message in cbind(parts$left, chars$ellip_h, parts$right, deparse.level = 0L):
    "number of rows of result is not a multiple of vector length (arg 2)"



<table class="dataframe">
<caption>A tibble: 0 x 47</caption>
<thead>
	<tr><th scope=col>SEQN</th><th scope=col>YEAR</th><th scope=col>RIDAGEYR</th><th scope=col>ALQ130</th><th scope=col>BMXARMC</th><th scope=col>BMXARML</th><th scope=col>LBXSOSSI</th><th scope=col>LBDSBUSI</th><th scope=col>BMXBMI</th><th scope=col>LB2SCLSI</th><th scope=col>...</th><th scope=col>RHD143</th><th scope=col>Alcohol_Intake</th><th scope=col>Caffeine_Intake</th><th scope=col>Calcium_Intake</th><th scope=col>Carbohydrate_Intake</th><th scope=col>Fiber_Intake</th><th scope=col>Kcal_Intake</th><th scope=col>Sodium_Intake</th><th scope=col>Relative_Had_Diabetes</th><th scope=col>HDL_cholesterol</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>...</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
</tbody>
</table>




```R
sort(unique(df_formated$RIDAGEYR))
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>20</li><li>21</li><li>22</li><li>23</li><li>24</li><li>25</li><li>26</li><li>27</li><li>28</li><li>29</li><li>30</li><li>31</li><li>32</li><li>33</li><li>34</li><li>35</li><li>36</li><li>37</li><li>38</li><li>39</li><li>40</li><li>41</li><li>42</li><li>43</li><li>44</li><li>45</li><li>46</li><li>47</li><li>48</li><li>49</li><li>50</li><li>51</li><li>52</li><li>53</li><li>54</li><li>55</li><li>56</li><li>57</li><li>58</li><li>61</li><li>76</li><li>85</li></ol>




```R



seeing if there is any extreme value that might be due to bad input of the data. According to the paper:

> The preprocessing stage also converted any undecipherable values (errors in datatypes and standard formatting) from the database to null representations.
```


```R
boxplot(df[, c('col1', 'col2', 'colN')])

```


```R
normalize_columns <- function(master, columns) {
  for (column in columns) {
    master[, column] <- (master[, column] - min(master[, column])) / (max(master[, column]) - min(master[, column]))
  }
  return(master)
}

```
