#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as st
from statsmodels.stats.weightstats import ztest


def check_intervals_intersect(first_ci, second_ci):   
    first_ci = pd.Interval(*first_ci, closed='both') # transforms first_ci tuple to pd.Interval()
    second_ci = pd.Interval(*second_ci, closed='both') # transforms second_ci tuple to pd.Interval()
    are_intersect = first_ci.overlaps(second_ci)
    return are_intersect # True or False


def check_dge_with_ci(first_table, second_table, gene):
    fst_table_gene_ci = st.t.interval(alpha=0.95, # 95% доверительный интервал
                            df=len(first_table[gene]) - 1, # число степеней свободы - 1
                            loc=np.mean(first_table[gene]), # Среднее
                            scale=st.sem(fibrst_table[gene])) # Стандартная ошибка среднего
    snd_table_gene_ci = st.t.interval(alpha=0.95, # 95% доверительный интервал
                            df=len(second_table[gene]) - 1, # число степеней свободы - 1
                            loc=np.mean(second_table[gene]), # Среднее
                            scale=st.sem(second_table[gene])) # Стандартная ошибка среднего
    test_ci = check_intervals_intersect(first_ci=fst_table_gene_ci, second_ci=snd_table_gene_ci) 
    return test_ci


def check_dge_with_ztest(first_table, second_table, gene):
    # dge - differential gene expression
    gene_z_value = ztest(first_table[gene], 
                         second_table[gene])
    z_value = gene_z_value[1] < 0.05
    z_p_value = round(gene_z_value[1], ndigits=4)
    return z_value, z_p_value


def mean_exp(first_table, second_table, gene):
    gene_mean_exp = first_table[gene].mean() - second_table[gene].mean()
    return round(gene_mean_exp, ndigits=2)

# input for .csv data files and .csv results
first_cell_type_expressions_path = input("Type path to .csv file with first cell type expression: ")
second_cell_type_expressions_path = input("Type path to .csv file with second cell type expression data: ")
save_results_table = input("Type path for .csv file with results: ")

# read from .csv
first_expression_data = pd.read_csv(first_cell_type_expressions_path, index_col=0)
second_expression_data = pd.read_csv(second_cell_type_expressions_path, index_col=0)

genes = [] # list for gene names
ci_test_results = [] # list for CI test results
z_test_results = [] # list for z-test values
z_test_p_values = [] # list for z-test p-values
mean_diff = [] # list for mean genes expression

for gene in first_expression_data.columns: # cycling through all columns in first .csv
    if gene in second_expression_data.columns: # finding column matches in second .csv
        # checking type of pd.Series()
        if first_expression_data[gene].dtype and second_expression_data[gene].dtype in (int, float):
            # CI calculations and checking their intersection 
            gene_ci = check_dge_with_ci(
                first_table=first_expression_data, 
                second_table=second_expression_data, 
                gene=gene
            )
            genes.append(gene) # appending gene name to list
            ci_test_results.append(gene_ci) # appending CI test results to list
            
            # calling for z-test function
            z_value, z_p_value = check_dge_with_ztest(
                first_table=first_expression_data, 
                second_table=second_expression_data,
                gene=gene
            )
            z_test_results.append(z_value) # appending z-value to list
            z_test_p_values.append(z_p_value) # appending z-test p-value to list

            # calling for gene mean_diff function
            gene_mean_exp = mean_exp(
                first_table=first_expression_data, 
                second_table=second_expression_data,
                gene=gene
            )
            mean_diff.append(gene_mean_exp) # appending gene mean_diff function

# Python dict generation for results
results = {
    "gene name": genes,
    "ci_test_results": ci_test_results,
    "z_test_results": z_test_results,
    "z_test_p_values": z_test_p_values,
    "mean_diff": mean_diff
}

# Results transformation to pd.DataFrame()
results = pd.DataFrame(results)

# writing Results to .csv
results.to_csv(save_results_table)

