# Homework1

## Differential expression in B and NK cells

There are **three files** for *Homework after Lecture 5*. Here some discription of them and guidelines for **hw_lecture5_task4.py** utility.

- **Google Colab Notebook** solutions for Tasks 1-3 based on differential expression data in B and NK cells;

- **hw_lecture5_task4.py** Python script for differential expression analysis using *confidence interval (CI) method* and *z-test*;

- **requirements.txt** for virtual environment creation

### System

Launch of utility **hw_lecture5_task4.py** was tested on *Ubuntu 22.04.1 LTS* with *Python version 3.10.6*

### Utility discription

The **hw_lecture5_task4.py** provides differential expression analysis based on *confidence interval (CI) method* and *z-test*, i.e. for two cell types. As `input`, this script calls **two paths to .csv files** with gene expression and **path for `output` .csv file** with results. *Both input .csv files* should be organized in following way: for *each gene* should be *a column* with its expression *observations in rows*.

**Output .csv file** contains statistics results for each gene expression in both cell types:

|       | gene name | ci_test_results | z_test_results | z_test_p_values | mean_diff |
| :---  |  :----:   |     :----:      |     :----:     |     :----:      |  :----:   |
| 0     | TMCC1     | TRUE            | FALSE          | 0.1794          | -3.45     |        
| 1     | RANBP3    | FALSE           | TRUE           | 0.0001          | -6.47     |
| 2     | GABRG3    | TRUE            | FALSE          | 0.7046          | 0.76      |
| ...   | ...       | ...             | ...            | ...             | ...       |

After .csv files reading, for statistics calculations **hw_lecture5_task4.py** using some **functions**, listed bellow:

- `check_dge_with_ci()`: calculates gene expression *95% confidence intervals (CI)* for both cell types using `statsmodels.t.interval()`, calls second function `check_intervals_intersect()` for checking their intersection and puts TRUE into *ci_test_results column*, if they are intersect. If CI intersects, there no difference in gene expression for this cell types;

- `check_intervals_intersect()`: checks whether *confidence intervals (CI)* intersect;

- `check_dge_with_ztest()`: calculates z-test value and its p-value for gene expression in both cell types using `scipy.stats.ztest()` and puts TRUE into *z_test_results column*, if z-test p-value < 0.05 (z-test p-values are in *z_test_p_values column*);

- `mean_exp()`: determine difference in mean expression between cell types and puts it into *mean_diff column*\
*Attention! This function subtracts **the mean expression value of the second .csv file from the first .csv file.***

### Launch

- create virtual environment

`conda create --name {env name} python=3.10.6`

- activate you virtual environment

`conda activate {env name}`

- install the specified packages using the configuration file **requirements.txt**

`pip install -r requirements.txt`

- launch **hw_lecture5_task4.py**

`python3 hw_lecture5_task4.py`
