# Homework :honeybee:

## Tool for differential gene expression analysis (v1)

Utility **diff_exp_tool_v1.py** provides differential gene expression analysis based on *confidence interval (CI) method* and *z-test*, i.e. for two cell types. As **input**, this script calls **two paths to .csv files** with gene expression data and **name for output .csv file** with results. Also this utility promotes multiple test correction using methods listed in `statsmodels.stats.multitest.multipletests` [documentation](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html) with `std.input` family-wise error rate (alpha). *Both input .csv files* should be organized in following way: for *each gene* should be *a column* with its expression *observations in rows*.

### Files

There are **three files** for *Differential gene expression tool v1*. Here some discriptions of them and guidelines on how to use it and how does it works.

- [README.md](./README.md): discriptions for files in this directory;

- [requirements.txt](./requirements.txt): .txt file with the dependencies for *diff_exp_tool_v1.py*; 

- [diff_exp_tool_v1.py](./diff_exp_tool_v1.py): Python script for differential expression analysis using *confidence interval (CI) method* and *z-test*

In `./data/` there are *two .csv.gz files* on which **diff_exp_tool_v1.py** was tested.


### System

Launch of utility **diff_exp_tool_v1.py** was tested on *Ubuntu 22.04.1 LTS* with *Python version 3.10.6*

## How to use

### Input

Utility **diff_exp_tool_v1.py** accepts **four** basic parameters and **two** additional parameters from `std.input`. This parameters listed bellow in calling order:

**1. Basic parameters:**

- **path to first .csv file** with DEG data (str);

- **path to second .csv file** with DEG data (str);

- **name for .csv file** with results (str);

- **multiple test correction** for z-test decision (**yes/no**) (str):

*Comment:* Enter **'yes'**, if you want to recive multiple test correction for z-test results. In other case, you will recieve results with z-test p-values without correction.

**2. Additional parameters:**

You need to enter following parameters, if you choose **to perform multiple test correction for z-test** (*'yes'*).

- **alpha value** for multiple test correction (in format 0.05);

- **method** for multiple test correction (str):

*Comment:* this utility uses `statsmodels.stats.multitest.multipletests` for multiple test correction using methods listed in `statsmodels.stats.multitest.multipletests` [documentation](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html).

Enter method used for testing and adjustment of p-values (listed bellow): <a name="add_params"></a>

- *bonferroni*: one-step correction;

- *sidak*: one-step correction;

- *holm-sidak*: step down method using Sidak adjustments;

- *holm*: step-down method using Bonferroni adjustments;

- *simes-hochberg*: step-up method (independent);

- *hommel*: closed method based on Simes tests (non-negative);

- *fdr_bh*: Benjamini/Hochberg (non-negative);

- *fdr_by*: Benjamini/Yekutieli (negative);

- *fdr_tsbh*: two stage fdr correction (non-negative);

- *fdr_tsbky*: two stage fdr correction (non-negative)


### Output .csv file


There are two kinds of **output {name}.csv file** *(with or without testing and adjustment of p-values)*, which contain statistics results for each gene DEG from two input .csv files. Here examples for *output .csv files*.

- **output {name}.csv file** *without* multiple test correction for z-test p-values}:

|       | gene name | ci_test_results | z_test_results | z_test_p_values | mean_diff |
| :---  |  :----:   |     :----:      |     :----:     |     :----:      |  :----:   |
| 0     | TMCC1     | TRUE            | FALSE          | 0.1794          | -3.45     |        
| 1     | RANBP3    | FALSE           | TRUE           | 0.0001          | -6.47     |
| 2     | GABRG3    | TRUE            | FALSE          | 0.7046          | 0.76      |
| ...   | ...       | ...             | ...            | ...             | ...       |

- **output {name}_{method for p-value_adj}.csv file** *with* multiple test correction for z-test p-values:

|       | gene name | ci_test_results | z_test_p_values | z_test_padj | z_test_results | mean_diff |
| :---  |  :----:   |     :----:      |     :----:      |    :----:   |     :----:     |  :----:   |
| 0     | TMCC1     | TRUE            | 0.1794          | 1.0         | FALSE          | -3.45     |        
| 1     | RANBP3    | FALSE           | 0.0001          | 1.0         | FALSE          | -6.47     |
| 2     | GABRG3    | TRUE            | 0.7046          | 1.0         | FALSE          | 0.76      |
| ...   | ...       | ...             | ...             | ...         | ...            | ...       |


## How does it work

After .csv files reading, for statistics calculations **diff_exp_tool_v1.py** using some **functions** listed bellow:

- `check_dge_with_ci()`: calculates gene expression *95% confidence intervals (CI)* using `scipy.stats.t.interval()`, calls second function `check_intervals_intersect()` for checking their intersection and puts *TRUE* into **ci_test_results column**, if they intersect. If CI intersects, there no difference in gene expression for given samples;

- `check_intervals_intersect()`: checks whether *confidence intervals (CI)* intersect;

- `check_dge_with_ztest()`: calculates z-test value and its p-value for gene expression using `statsmodels.stats.weightstats()` and puts *TRUE* into *z_test_results column*, if z-test p-value <= 0.05 (z-test p-values are in **z_test_p_values column**);

- `mean_exp()`: determines difference in mean expression between given samples and puts value into **mean_diff column**

*Attention! This function subtracts **the mean expression value of the second .csv file from the first .csv file.***

- `adjust_pvalue()`: tests results and performs z-test p-value correction for multiple tests using `statsmodels.stats.multitest.multipletests()` with *alpha value* and *method* from `std.input` (for methods check [here](#add_params) or in `statsmodels.stats.multitest.multipletests()` [documentation](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html)). Adjusted p-values for z-test are in **z_test_padj column**

## Launch

- create virtual environment

`conda create --name {env name} python=3.10.6`

- activate you virtual environment

`conda activate {env name}`

- install the specified packages using the configuration file **requirements.txt**

`pip install -r requirements.txt`

- download script **diff_exp_tool_v1.py**

`wget https://github.com/AnasZol/Stats_BI_2022/blob/diff_exp_tool/diff_exp_tool/diff_exp_tool_v1.py`

- launch **diff_exp_tool_v1.py**

`python3 diff_exp_tool_v1.py`
