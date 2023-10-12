#!/usr/bin/env python
# coding: utf-8

# In[2]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
import data_analysis_utilitites as da_utilities
import data_analysis_areas as da_area
from scipy.stats import shapiro, levene
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import numpy as np


# In[49]:


data = da_utilities.load_all_criticality_data_no_duplicate_files('/home/bellijjy/criticality_analysis', ['_con', '_dav', '_dud', '_Cor', '_cha'], area=None, state=None, day=None, epoch=None, time_chunk=None)


# In[86]:


def load_from_parquet(file_path):
    """
    Loads a DataFrame from a Parquet file.

    Parameters:
    -----------
    file_path : str
        The path to the Parquet file.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the data loaded from the Parquet file.
    """
    
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Failed to load data from {file_path}. Error: {e}")
        return None
    
complete_data = load_from_parquet('/home/bellijjy/completedataframe')
complete_data


# In[9]:


def add_distance_to_criticality_column(df, branching_factor_column='branching_factor'):
    """
    Add a new column to the DataFrame that calculates the distance to criticality.
    
    Parameters:
    - df: DataFrame, the input DataFrame
    - branching_factor_column: str, the column name containing branching factor values
    
    Returns:
    - DataFrame with an additional 'distance_to_criticality' column
    """
    df['distance_to_criticality'] = np.log(1 - df[branching_factor_column])
    return df

# Add the distance_to_criticality column to the DataFrame


# In[72]:


formula = "distance_to_criticality ~ area * linear_distance * linear_speed"
formulas_reduced = ["distance_to_criticality ~ linear_distance * linear_speed", "distance_to_criticality ~ linear_distance * area", "distance_to_criticality ~ area * linear_speed"]


# In[76]:


from scipy.stats import chi2

def compare_models(base_formula, reduced_formulas, data, groups):
    """
    Compare a base mixed-effects model to several reduced models using likelihood ratio tests. Use the ML instead of the REML
    as the ML partializes the fixed effects out -- and it is precisely those that are of interest.
    
    Parameters:
    - base_formula: str, formula for the base model
    - reduced_formulas: list of str, formulas for the reduced models
    - data: DataFrame, the data
    - groups_column: str, the column name to use for groups (random effects)
    
    Returns:
    - dict, Likelihood ratio statistics, p-values, degrees of freedom for each reduced model
    """
    
    # Fit the base model and obtain its log-likelihood
    base_model = smf.mixedlm(base_formula, data=data, groups=groups)
    base_fit_result = base_model.fit(reml = False)
    base_loglike = base_fit_result.llf
    
    # Initialize a dictionary to store comparison results
    comparison_results = {}
    
    # Loop through each reduced formula and compare to the base model
    for formula_i in reduced_formulas:
        # Fit the reduced model and obtain its log-likelihood
        model_i = smf.mixedlm(formula_i, data=data, groups=groups)
        fit_result_i = model_i.fit(reml = False)
        reduced_loglike = fit_result_i.llf
        
        # Manually calculate the likelihood ratio test statistic
        lr_stat = -2 * (reduced_loglike - base_loglike)
        
        # Calculate the p-value
        df_diff = base_fit_result.df_modelwc - fit_result_i.df_modelwc
        p_value = chi2.sf(lr_stat, df_diff)
        
        #Store the results along with AIC and BIC
        comparison_results[formula_i] = {
            'LR Statistic': lr_stat, 
            'p-value': p_value, 
            'df difference': df_diff   
        }
        
    return comparison_results

# Sample function call (Note: This is a placeholder; actual data would be used in practice)
results = compare_models(formula, formulas_reduced, filtered_df, groups= filtered_df['animal'])
results


# In[60]:


from scipy.stats import spearmanr, pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def check_linearity(fit_result, data, dependent_var, predictors):
    """
    Check the linearity assumption of a mixed-effects model by plotting residuals vs fitted values
    and residuals vs each predictor, and by calculating correlation coefficients.
    
    Parameters:
    - fit_result: mixedlm fit result object, the fitted mixed-effects model
    - data: DataFrame, the original data
    - dependent_var: str, the dependent variable
    - predictors: list of str, the predictor variables
    
    Returns:
    - None, but plots graphs for visual inspection and prints correlation coefficients
    """
    
    # Extract residuals and fitted values
    residuals = fit_result.resid
    fitted_vals = fit_result.fittedvalues
    
    # Check if the order of the indices matches
    if not data.index.unique().equals(residuals.index.unique()):
        raise IndexError('The index order is shuffled')
    
    # Add residuals and fitted values back to the original data
    data_with_residuals = data.copy()
    data_with_residuals['residuals'] = residuals.values
    data_with_residuals['fitted_vals'] = fitted_vals.values
    
    # Plot residuals vs fitted values
    plt.figure()
    sns.scatterplot(x='fitted_vals', y='residuals', data=data_with_residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Fitted Values")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.show()
    
    # Calculate Pearson and Spearman correlation for residuals vs fitted values
    pearson_corr, _ = pearsonr(data_with_residuals['fitted_vals'], data_with_residuals['residuals'])
    spearman_corr, _ = spearmanr(data_with_residuals['fitted_vals'], data_with_residuals['residuals'])
    print(f"Pearson correlation between residuals and fitted values: {pearson_corr}")
    print(f"Spearman correlation between residuals and fitted values: {spearman_corr}")
    
    # Plot residuals vs each predictor and calculate correlation coefficients
    for predictor in predictors:
        plt.figure()
        sns.scatterplot(x=predictor, y='residuals', data=data_with_residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f"Residuals vs {predictor}")
        plt.xlabel(predictor)
        plt.ylabel("Residuals")
        plt.show()

check_linearity(fit_result_ml, filtered_df, dependent_var='distance_to_criticality', predictors=['area', 'linear_distance', 'linear_speed'])



# In[90]:


formula = "distance_to_criticality ~ area * linear_speed * linear_distance"
df_with_distance = add_distance_to_criticality_column(complete_data, branching_factor_column='branching_factor')

filtered_df = df_with_distance[df_with_distance['linear_speed'].notna()]

model_with_ml = smf.mixedlm(formula, data=filtered_df, groups= filtered_df['animal'])

fit_result_ml = model_with_ml.fit(reml = False)
        
display(fit_result_ml.summary())


# In[71]:


from scipy.stats import shapiro, iqr
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import anderson, levene
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd
import seaborn as sns

def calculate_freedman_diaconis_bins(data):
    data_iqr = iqr(data)
    n = len(data)
    bin_width = 2 * (data_iqr / np.cbrt(n))
    if bin_width == 0:
        return 1
    num_bins = int((max(data) - min(data)) / bin_width)
    return num_bins

def check_mixed_model_assumptions(fit_result, data, dependent_var, predictors):
    residuals = fit_result.resid
    fitted_vals = fit_result.fittedvalues
    
    if not data.index.unique().equals(residuals.index.unique()):
        raise IndexError('The index order is shuffled')
    
    data_with_residuals = data.copy()
    data_with_residuals['residuals'] = residuals.values

    # Binning continuous variables using Freedman-Diaconis rule
    data_with_residuals['linear_distance_bin'] = pd.cut(data_with_residuals['linear_distance'], bins=calculate_freedman_diaconis_bins(data_with_residuals['linear_distance']))
    data_with_residuals['linear_speed_bin'] = pd.cut(data_with_residuals['linear_speed'], bins=calculate_freedman_diaconis_bins(data_with_residuals['linear_speed']))
    
    grouped = data_with_residuals.groupby(['animal', 'area', 'linear_distance_bin', 'linear_speed_bin'])
    residual_groups = []
    
    for name, group in grouped:
        if len(group['residuals']) >= 3:  # Shapiro-Wilk test requires at least 3 data points
            shapiro_test_stat, shapiro_p_value = shapiro(group['residuals'])
            print(f"Group: {name}, Shapiro-Wilk test for normality: Test Statistic = {shapiro_test_stat}, p-value = {shapiro_p_value}")

            # QQ-Plot for each group
            plt.figure()
            qqplot(group['residuals'], line='s')
            plt.title(f'Normal Q-Q for Group: {name}')
            
            residual_groups.append(group['residuals'])
    
    # Levene's test for homoscedasticity
    levene_test_stat, levene_p_value = levene(*residual_groups)
    print(f"Levene's test for homoscedasticity: Test Statistic = {levene_test_stat}, p-value = {levene_p_value}")

     
    X = data[predictors].copy()  # Use copy() to avoid SettingWithCopyWarning
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = list(set(X.columns) - set(numeric_cols))
    
    # One-hot encode only categorical variables
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Remove rows with missing or infinite values
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Convert Boolean columns to integers
    X = X * 1  # This will convert True to 1 and False to 0

    if X.empty:
        print("After cleaning, the data frame for VIF calculation is empty. Skipping VIF calculation.")
        return
    
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    
    print("Variance Inflation Factors:")
    print(vif_data)

check_mixed_model_assumptions(fit_result_ml, filtered_df, dependent_var='distance_to_criticality', predictors=['area', 'linear_distance', 'linear_speed'])

    


# In[ ]:




