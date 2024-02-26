#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
complete_data = load_from_parquet('/home/bellijjy/completedataframeallrats')
complete_data


# In[3]:


def plot_tau_vs_branching_factor(df, x_col='branching_factor', y_col='tau'):
    """
    Generate a Seaborn scatter plot of tau vs branching_factor.
    
    Parameters:
        df (DataFrame): The DataFrame containing the data.
        x_col (str): The column name for the x-axis.
        y_col (str): The column name for the y-axis.
        
    Returns:
        None: The function will plot the data.
    """
    sns.set_theme(style="white")  # Set the theme to 'crest'
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    
    plt.title(f'{y_col} vs {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    
    plt.show()

# Test the function with the mock DataFrame
plot_tau_vs_branching_factor(complete_data)


# In[4]:


def plot_tau_distribution(df, state_filter):
    """
    Generate a Seaborn violin plot for the distribution of tau values
    for each day/epoch pair, filtered by the state (wake/sleep).
    
    Parameters:
        df (DataFrame): The DataFrame containing the data.
        state_filter (str): The state to filter by ('wake' or 'sleep').
        
    Returns:
        None: The function will plot the data.
    """
    # Filter the DataFrame by state
    df_filtered = df[df['state'] == state_filter]
    
    # Drop rows where any of the specified columns have NaN values
    df_filtered = df_filtered.dropna(subset=['day', 'epoch', 'tau'])
    
    # Create a new column to represent the day/epoch pair
    df_filtered['day_epoch'] = df_filtered['day'].astype(str) + "-" + df_filtered['epoch'].astype(str)
    
    # Sort by 'day' and 'epoch'
    df_filtered.sort_values(by=['day', 'epoch'], inplace=True)
    
    # Plotting
    sns.set_theme(style="whitegrid")  # Set the theme to 'whitegrid'
    plt.figure(figsize=(20, 6))
    sns.violinplot(data=df_filtered, x='day_epoch', y='tau')
    
    plt.title(f'Distribution of Tau values for each Day-Epoch pair during {state_filter.capitalize()}')
    plt.xlabel('Day-Epoch')
    plt.ylabel('Tau')
    
    plt.show()

# Create the mock DataFrame with some NaN values for testing
df_with_nan = complete_data.copy()
df_with_nan.loc[df_with_nan.sample(frac=0.1).index, 'tau'] = np.nan

# Generate the plots
plot_tau_distribution(df_with_nan, 'wake')
plot_tau_distribution(df_with_nan, 'sleep')


# In[5]:


def plot_tau_distribution_boxplot(df, state_filter):
    """
    Generate a Seaborn box plot for the distribution of tau values
    for each day/epoch pair, filtered by the state (wake/sleep).
    
    Parameters:
        df (DataFrame): The DataFrame containing the data.
        state_filter (str): The state to filter by ('wake' or 'sleep').
        
    Returns:
        None: The function will plot the data.
    """
    # Filter the DataFrame by state
    df_filtered = df[df['state'] == state_filter]
    
    # Drop rows where any of the specified columns have NaN values
    df_filtered = df_filtered.dropna(subset=['day', 'epoch', 'tau'])
    
    # Create a new column to represent the day/epoch pair
    df_filtered['day_epoch'] = df_filtered['day'].astype(str) + "-" + df_filtered['epoch'].astype(str)
    
    # Sort by 'day' and 'epoch'
    df_filtered.sort_values(by=['day', 'epoch'], inplace=True)
    
    # Plotting
    sns.set_theme(style="whitegrid")  # Set the theme to 'whitegrid'
    plt.figure(figsize=(20, 6))
    sns.boxplot(data=df_filtered, x='day_epoch', y='tau')
    
    plt.title(f'Distribution of Tau values for each Day-Epoch pair during {state_filter.capitalize()}')
    plt.xlabel('Day-Epoch')
    plt.ylabel('Tau')
    
    plt.show()

# Generate the box plots
plot_tau_distribution_boxplot(df_with_nan, 'wake')
plot_tau_distribution_boxplot(df_with_nan, 'sleep')


# In[6]:


def plot_tau_distribution_boxplot_complete(df, state_filter):
    """
    Generate a Seaborn box plot for the distribution of tau values
    for each day/epoch pair, filtered by the state (wake/sleep). 
    This version includes all day-epoch combinations that are not completely absent.
    
    Parameters:
        df (DataFrame): The DataFrame containing the data.
        state_filter (str): The state to filter by ('wake' or 'sleep').
        
    Returns:
        None: The function will plot the data.
    """
    # Filter the DataFrame by state
    df_filtered = df[df['state'] == state_filter]
    
    # Drop rows where any of the specified columns have NaN values
    df_filtered = df_filtered.dropna(subset=['day', 'epoch', 'tau'])
    
    # Get unique day-epoch pairs present in the DataFrame
    unique_day_epoch_pairs = df_filtered[['day', 'epoch']].drop_duplicates().sort_values(by=['day', 'epoch'])
    
    # Create a new column to represent the day/epoch pair
    unique_day_epoch_pairs['day_epoch'] = unique_day_epoch_pairs['day'].astype(str) + "-" + unique_day_epoch_pairs['epoch'].astype(str)
    
    # Create a new column to represent the day/epoch pair in the original DataFrame
    df_filtered['day_epoch'] = df_filtered['day'].astype(str) + "-" + df_filtered['epoch'].astype(str)
    
    # Plotting
    sns.set_theme(style="whitegrid")  # Set the theme to 'whitegrid'
    plt.figure(figsize=(20, 6))
    sns.boxplot(data=df_filtered, x='day_epoch', y='tau', order=unique_day_epoch_pairs['day_epoch'])
    
    plt.title(f'Distribution of Tau values for each Day-Epoch pair during {state_filter.capitalize()}')
    plt.xlabel('Day-Epoch')
    plt.ylabel('Tau')
    
    plt.show()

# Generate the box plots with complete day-epoch combinations
plot_tau_distribution_boxplot_complete(df_with_nan, 'wake')
plot_tau_distribution_boxplot_complete(df_with_nan, 'sleep')


# In[7]:


def plot_tau_distribution_boxplot_combined(df):
    """
    Generate a Seaborn box plot for the distribution of tau values
    for each day/epoch pair, combining 'wake' and 'sleep' states in a single plot.
    Different colors are used for the two states.
    
    Parameters:
        df (DataFrame): The DataFrame containing the data.
        
    Returns:
        None: The function will plot the data.
    """
    # Drop rows where any of the specified columns have NaN values
    df_filtered = df.dropna(subset=['day', 'epoch', 'tau', 'state'])
    
    # Get unique day-epoch pairs present in the DataFrame
    unique_day_epoch_pairs = df_filtered[['day', 'epoch']].drop_duplicates().sort_values(by=['day', 'epoch'])
    
    # Create a new column to represent the day/epoch pair
    unique_day_epoch_pairs['day_epoch'] = unique_day_epoch_pairs['day'].astype(str) + "-" + unique_day_epoch_pairs['epoch'].astype(str)
    
    # Create a new column to represent the day/epoch pair in the original DataFrame
    df_filtered['day_epoch'] = df_filtered['day'].astype(str) + "-" + df_filtered['epoch'].astype(str)
    
    # Plotting
    sns.set_theme(style="whitegrid")  # Set the theme to 'whitegrid'
    plt.figure(figsize=(20, 6))
    sns.boxplot(data=df_filtered, x='day_epoch', y='tau', hue='state', order=unique_day_epoch_pairs['day_epoch'])
    
    plt.title('Distribution of Tau values for each Day-Epoch pair (Wake and Sleep)')
    plt.xlabel('Day-Epoch')
    plt.ylabel('Tau')
    
    plt.show()

# Generate the combined box plot
plot_tau_distribution_boxplot_combined(df_with_nan)


# In[ ]:




