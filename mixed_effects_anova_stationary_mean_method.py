#!/usr/bin/env python
# coding: utf-8

# In[1]:


import data_analysis_areas as da_area
import data_analysis_utilitites as da_utilities
import data_analysis_sleep_wake as da_wasl


# In[2]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
import data_analysis_utilitites as da_utilities
import data_analysis_areas as da_area
from scipy.stats import shapiro, levene
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[10]:


data = da_utilities.load_all_criticality_data_no_duplicate_files('/home/bellijjy/criticality_january_sm', ['_fra','_dud', '_dav', '_con', '_cha', '_Cor', '_egy', '_gov'], area=None, state=None, day=None, epoch=None, time_chunk=None)
data


# In[12]:


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
df_with_distance = add_distance_to_criticality_column(data)


# In[6]:


data_with_speed = data[data['speed'].notna()]

data_with_speed


# In[7]:


formula =  "distance_to_criticality ~ area * state * speed"
formulas_reduced = ["distance_to_criticality ~ state * day", "distance_to_criticality ~ area * day", "distance_to_criticality ~ area * state"]


# In[16]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

# Assuming 'formula' is defined and 'data' is your DataFrame
model_robust = smf.rlm(formula="distance_to_criticality ~ area * state", data=data)
fit_result_robust = model_robust.fit()

print(fit_result_robust.summary())


# In[14]:


import statsmodels.formula.api as smf
formula = "distance_to_criticality ~ area * state"
formulas_reduced = ["distance_to_criticality ~ state * day", "distance_to_criticality ~ area * day", "distance_to_criticality ~ area * state"]
     
# Assuming 'formula' is defined and 'data' is your DataFrame
model_ols = smf.ols(formula, data=data)
fit_result_ols = model_ols.fit()

print(fit_result_ols.summary())


# In[9]:


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

check_linearity(fit_result_robust, data, dependent_var='distance_to_criticality', predictors=['area', 'state'])



# In[17]:


from scipy.stats import f

formula = "distance_to_criticality ~ area * state * day"
formulas_reduced = ["distance_to_criticality ~ state * day", "distance_to_criticality ~ area * day", "distance_to_criticality ~ area * state"]

def compare_models(base_formula, reduced_formulas, data):
    base_model = smf.ols(base_formula, data=data)
    base_fit_result = base_model.fit()
    base_rss = base_fit_result.ssr
    base_df_model = base_fit_result.df_model

    comparison_results = {}

    for formula_i in reduced_formulas:
        model_i = smf.ols(formula_i, data=data)
        fit_result_i = model_i.fit()
        reduced_rss = fit_result_i.ssr
        reduced_df_resid = fit_result_i.df_resid
        reduced_df_model = fit_result_i.df_model

        num_parameters_difference = base_df_model - reduced_df_model
        numerator = (reduced_rss-base_rss) / num_parameters_difference
        denominator = reduced_rss / reduced_df_resid
        f_stat = numerator / denominator
        p_value = f.sf(f_stat, num_parameters_difference, reduced_df_resid)

        comparison_results[formula_i] = {
            'F Statistic': f_stat, 
            'p-value': p_value, 
            'df difference': base_df_model - reduced_df_model
        }

    return comparison_results

# Usage
results = compare_models(formula, formulas_reduced, data)
print(results)


# In[10]:


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
    #data_with_residuals['linear_distance_bin'] = pd.cut(data_with_residuals['linear_distance'], bins=calculate_freedman_diaconis_bins(data_with_residuals['linear_distance']))
    #data_with_residuals['linear_speed_bin'] = pd.cut(data_with_residuals['linear_speed'], bins=calculate_freedman_diaconis_bins(data_with_residuals['linear_speed']))
    
    grouped = data_with_residuals.groupby(['area', 'state'])
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

check_mixed_model_assumptions(fit_result_robust, data, dependent_var='distance_to_criticality', predictors=['area', 'state'])

    


# In[ ]:




