#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as st
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.multitest import multipletests


def check_intervals_intersect(first_ci, second_ci):   
    first_ci = pd.Interval(*first_ci, closed='both') # transforms first_ci tuple to pd.Interval()
    second_ci = pd.Interval(*second_ci, closed='both') # transforms second_ci tuple to pd.Interval()
    are_intersect = first_ci.overlaps(second_ci)
    return are_intersect # True or False


def check_dge_with_ci(fst_table_gene, snd_table_gene):
    fst_table_gene_ci = st.t.interval(alpha=0.95, # 95% доверительный интервал
                            df=len(fst_table_gene) - 1, # число степеней свободы - 1
                            loc=np.mean(fst_table_gene), # Среднее
                            scale=st.sem(fst_table_gene)) # Стандартная ошибка среднего
    snd_table_gene_ci = st.t.interval(alpha=0.95, # 95% доверительный интервал
                            df=len(snd_table_gene) - 1, # число степеней свободы - 1
                            loc=np.mean(snd_table_gene), # Среднее
                            scale=st.sem(snd_table_gene)) # Стандартная ошибка среднего
    test_ci = check_intervals_intersect(fst_table_gene_ci, snd_table_gene_ci) 
    return test_ci


def check_dge_with_ztest(fst_table_gene, snd_table_gene):
    # dge - differential gene expression
    z_p_value = ztest(fst_table_gene, 
                         snd_table_gene)[1]
    return z_p_value


def adjust_pvalue(p_values, alpha, method):
    p_adj = multipletests(p_values, alpha, method)[1]
    return p_adj


def mean_exp(fst_table_gene, snd_table_gene):
    gene_mean_exp = fst_table_gene.mean() - snd_table_gene.mean()
    return round(gene_mean_exp, ndigits=2)


# input for .csv data files, name for .csv results
first_cell_type_expressions_path = input("Type path to .csv file with first cell type expression: ")
second_cell_type_expressions_path = input("Type path to .csv file with second cell type expression data: ")
save_results_table = input("Type name for output .csv file with results: ")

# input alpha and method for multiple test correction (if do_mtcor = yes)
do_mtcor = input("Multiple test correction for z-test (yes/no): ")
if do_mtcor == 'yes':
    mtcor_alpha = float(input("Type alpha for multiple test correction: "))
    mtcor_method = input("Type method for multiple test correction: ")

# reading from .csv
first_expression_data = pd.read_csv(first_cell_type_expressions_path, index_col=0)
second_expression_data = pd.read_csv(second_cell_type_expressions_path, index_col=0)

genes = [] # list for gene names
ci_test_results = [] # list for CI test results
z_test_p_values = [] # list for z-test p-values
mean_diff = [] # list for mean genes expression

for gene in first_expression_data.columns: # cycling through all columns in first .csv
    if gene in second_expression_data.columns: # finding column matches in second .csv
        # checking type of pd.Series()
        if first_expression_data[gene].dtype and second_expression_data[gene].dtype in (int, float):
            # CI calculations and checking their intersection 
            gene_ci = check_dge_with_ci(
                first_expression_data[gene], 
                second_expression_data[gene], 
            )
            genes.append(gene) # appending gene name to list
            ci_test_results.append(gene_ci) # appending CI test results to list
            
            # calling for z-test function
            z_p_value = check_dge_with_ztest(
                first_expression_data[gene], 
                second_expression_data[gene],
            )
            z_test_p_values.append(z_p_value) # appending z-test p-value to list

            # calling for gene mean_diff function
            gene_mean_exp = mean_exp(
                first_expression_data[gene], 
                second_expression_data[gene],
            )
            mean_diff.append(gene_mean_exp) # appending gene mean_diff function

# Python dict generation for results
results = {
    "gene": genes,
    "ci_test_results": ci_test_results,
    "z_test_p_values": z_test_p_values,
    "mean_diff": mean_diff
}

# calling function for multiple test correction (if do_mtcor = yes)
if do_mtcor == 'yes':
    z_test_padj = adjust_pvalue(z_test_p_values, mtcor_alpha, mtcor_method) 
    results.update({'z_test_padj': z_test_padj})

if 'z_test_padj' in results.keys():
    z_test_values = [(i <= mtcor_alpha) for i in results['z_test_padj']]
    results.update({'z_test_values': z_test_values})
    results = pd.DataFrame(results, columns=[
        'gene', 'ci_test_results', 'z_test_p_values', 
        'z_test_padj', 'z_test_values', 'mean_diff'
    ])
    results.to_csv(f'{save_results_table}_{mtcor_method}.csv')
else:
    z_test_values = [(i <= 0.05) for i in results['z_test_p_values']]
    results.update({'z_test_values': z_test_values})
    results = pd.DataFrame(results, columns=[
        'gene', 'ci_test_results','z_test_values', 
        'z_test_p_values', 'mean_diff'
    ])
    results.to_csv(f'{save_results_table}.csv')

