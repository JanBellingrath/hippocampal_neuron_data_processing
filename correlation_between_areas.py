#!/usr/bin/env python
# coding: utf-8

# In[9]:


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


# In[11]:


def calculate_area_correlations_by_state_with_viz(df, area_cols=['CA1', 'CA3'], state_col='state', variable='tau'):
    """
    Calculate and visualize correlations between areas for different states without aggregation.
    
    Parameters:
        df (DataFrame): The data
        area_cols (list): List of area names to include in the correlation
        state_col (str): Column name containing state labels (e.g., 'state')
        variable (str): The variable for which to calculate the correlation (e.g., 'tau')
    
    Returns:
        dict: A dictionary containing 2x2 correlation matrices for each state
    """
    correlation_matrices = {}
    
    for state in df[state_col].unique():
        # Subset the data for each state
        state_data = df[df[state_col] == state]
        
        # Create empty lists to store variable values for each area
        area_values = {area: [] for area in area_cols}
        
        # Populate the lists with variable values
        for area in area_cols:
            area_values[area] = state_data[state_data['area'] == area][variable].tolist()
        
        # Ensure the lists have the same length by truncating the longer one
        print(len(area_values[area_cols[0]]))
        print(len(area_values[area_cols[1]]))
        min_length = min(len(area_values[area_cols[0]]), len(area_values[area_cols[1]]))
        for area in area_cols:
            area_values[area] = area_values[area][:min_length]
        
        # Calculate the correlation between the areas for this state
        corr_value = np.corrcoef(area_values[area_cols[0]], area_values[area_cols[1]])[0, 1]
        correlation_matrices[state] = corr_value
        
        # Print the calculated correlation
        print(f"Correlation between {area_cols[0]} and {area_cols[1]} for state {state}: {corr_value:.3f}")
        
        # Prepare data for visualization
        corr_matrix = pd.DataFrame([[1, corr_value], [corr_value, 1]], index=area_cols, columns=area_cols)
        
        # Visualize the correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='crest', cbar=True, square=True, fmt='.2f', annot_kws={'size': 20})
        plt.title(f"Correlation Matrix Between Areas for State {state}")
        plt.show()
    
    return correlation_matrices


# Calculate and visualize the correlations
correlations_by_state_with_viz = calculate_area_correlations_by_state_with_viz(complete_data)
correlations_by_state_with_viz



# In[13]:


def calculate_area_correlations(df, area_cols=['CA1', 'CA3'], variable='tau'):
    """
    Calculate the correlation between specified areas without considering any other conditions.
    
    Parameters:
        df (DataFrame): The data
        area_cols (list): List of area names to include in the correlation
        variable (str): The variable for which to calculate the correlation (e.g., 'tau')
    
    Returns:
        float: The correlation between the specified areas
    """
    # Create empty lists to store variable values for each area
    area_values = {area: [] for area in area_cols}
    
    # Populate the lists with variable values
    for area in area_cols:
        area_values[area] = df[df['area'] == area][variable].tolist()
    
    print(len(area_values[area_cols[0]]))
    print(len(area_values[area_cols[1]]))
    # Ensure the lists have the same length by truncating the longer one
    min_length = min(len(area_values[area_cols[0]]), len(area_values[area_cols[1]]))
    for area in area_cols:
        area_values[area] = area_values[area][:min_length]
    
    # Calculate the correlation between the specified areas
    corr_value = np.corrcoef(area_values[area_cols[0]], area_values[area_cols[1]])[0, 1]
    
    print(f"Correlation between {area_cols[0]} and {area_cols[1]}: {corr_value:.3f}")
    
    # Prepare data for visualization
    corr_matrix = pd.DataFrame([[1, corr_value], [corr_value, 1]], index=area_cols, columns=area_cols)
    
    # Visualize the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='crest', cbar=True, square=True, fmt='.2f', annot_kws={'size': 20})
    plt.title("Correlation Matrix Between Specified Areas")
    plt.show()
    
    return corr_value

# Calculate and visualize the correlation
correlation = calculate_area_correlations(complete_data)
correlation


# In[15]:


def plot_separate_aggregated_area_correlations(df, area_cols=['CA1', 'CA3'], variable='tau'):
    """
    Plot the aggregated data points for specified areas to visualize potential correlation or trends.
    
    The data is aggregated by the unique combination of day, epoch, and time_chunk, 
    and then averaged across animals for each area. Each area has its own set of points based on the unique index.
    
    Parameters:
        df (DataFrame): The data
        area_cols (list): List of area names to include in the plot
        variable (str): The variable for which to plot the values (e.g., 'tau')
    """
    # Create a new column that combines day, epoch, and time_chunk into a unique index
    df['combined_index'] = df[['day', 'epoch', 'time_chunk']].apply(tuple, axis=1)
    
    plt.figure(figsize=(12, 8))
    
    for area in area_cols:
        # Filter the data for each area
        area_data = df[df['area'] == area]
        
        # Aggregate the data by the combined index and average across animals
        aggregated_data = area_data.groupby('combined_index')[variable].mean().reset_index()
        
        # Create a scatter plot for each area
        plt.scatter(range(len(aggregated_data)), aggregated_data[variable], label=f"{area} {variable}")
    
    plt.title(f"Aggregated Scatter Plot of {variable} for {', '.join(area_cols)}")
    plt.xlabel("Unique Index (day, epoch, time_chunk)")
    plt.ylabel(variable)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_separate_aggregated_area_correlations(complete_data)


# In[16]:


# Adjust the function to consider day-epoch combinations
def compute_weighted_correlations(df, area_cols=['CA1', 'CA3'], variable='tau'):
    """
    Compute weighted average correlation based on unique day, epoch, time_chunk combinations for each day-epoch combination.
    The weights are determined by the number of unique elements in each day-epoch combination.
    
    Parameters:
        df (DataFrame): The data
        area_cols (list): List of area names to include in the calculation
        variable (str): The variable for which to calculate the correlation (e.g., 'tau')
        
    Returns:
        float: Weighted average correlation across day-epoch combinations
        DataFrame: Number of unique elements in each day-epoch combination
    """
    # Create a new column that combines day, epoch, and time_chunk into a unique index
    df['combined_index'] = df[['day', 'epoch', 'time_chunk']].apply(tuple, axis=1)
    
    # Initialize variables to store weighted sum and total weight
    weighted_sum_corr = 0.0
    total_weight = 0
    
    # Initialize DataFrame to store number of unique elements in each day-epoch combination
    day_epoch_count_df = pd.DataFrame(columns=['Day', 'Epoch', 'Count'])
    
    for day, epoch in df[['day', 'epoch']].drop_duplicates().values:
        # Filter data for each day-epoch combination
        day_epoch_data = df[(df['day'] == day) & (df['epoch'] == epoch)]
        
        # Create lists to store variable values for each area
        area_values = {area: [] for area in area_cols}
        
        for idx in day_epoch_data['combined_index'].unique():
            # Filter data for each unique index within the day-epoch combination
            index_data = day_epoch_data[day_epoch_data['combined_index'] == idx]
            
            # Check if both areas are represented for this unique index
            if all(area in index_data['area'].values for area in area_cols):
                for area in area_cols:
                    # Get the variable value for this area and unique index
                    area_value = index_data[index_data['area'] == area][variable].values[0]
                    area_values[area].append(area_value)
        
        # Calculate correlation for this day-epoch combination if both areas have data
        if all(len(values) > 0 for values in area_values.values()):
            day_epoch_corr = np.corrcoef(area_values[area_cols[0]], area_values[area_cols[1]])[0, 1]
            
            # Calculate weight for this day-epoch combination (number of unique elements)
            weight = len(area_values[area_cols[0]])
            
            # Update weighted sum and total weight
            weighted_sum_corr += day_epoch_corr * weight
            total_weight += weight
            
            # Update DataFrame with day-epoch count
            day_epoch_count_df.loc[len(day_epoch_count_df)] = [day, epoch, weight]
    
    # Calculate weighted average correlation
    weighted_avg_corr = weighted_sum_corr / total_weight if total_weight > 0 else np.nan
    
    return weighted_avg_corr, day_epoch_count_df

# Since sample data is not desired, the function is not executed here
# You can try running this function with your 'complete_data' DataFrame to check its functionality

compute_weighted_correlations(complete_data)


# In[17]:


def filter_by_tau_threshold(df, threshold):
    return df[(df['tau'] > threshold) & df['x_position'].notna() & df['y_position'].notna()]

blob_df = filter_by_tau_threshold(complete_data, 4000)
sns.violinplot(x='animal', y='tau', data=blob_df)
plt.show()


# In[29]:


# Adjust the function to consider day-epoch combinations and incorporate an additional DataFrame for non-extreme values
def compute_weighted_correlations_with_fallback(df, df_fallback, area_cols=['CA1', 'CA3'], variable='tau'):
    """
    Compute weighted average correlation based on unique day, epoch, time_chunk combinations for each day-epoch combination.
    The weights are determined by the number of unique elements in each day-epoch combination. If no correspondence is found in the first DataFrame,
    the function looks for it in the second DataFrame (df_fallback).
    
    Parameters:
        df (DataFrame): The data for extreme values
        df_fallback (DataFrame): The data for non-extreme values
        area_cols (list): List of area names to include in the calculation
        variable (str): The variable for which to calculate the correlation (e.g., 'tau')
        
    Returns:
        float: Weighted average correlation across day-epoch combinations
        DataFrame: Number of unique elements in each day-epoch combination
    """
    # Create a new column that combines day, epoch, and time_chunk into a unique index for both DataFrames
    df['combined_index'] = df[['day', 'epoch', 'time_chunk']].apply(tuple, axis=1)
    df_fallback['combined_index'] = df_fallback[['day', 'epoch', 'time_chunk']].apply(tuple, axis=1)
    
    # Initialize variables to store weighted sum and total weight
    weighted_sum_corr = 0.0
    total_weight = 0
    
    # Initialize DataFrame to store number of unique elements in each day-epoch combination
    day_epoch_count_df = pd.DataFrame(columns=['Day', 'Epoch', 'Count'])
    
    for day, epoch in df[['day', 'epoch']].drop_duplicates().values:
        # Filter data for each day-epoch combination
        day_epoch_data = df[(df['day'] == day) & (df['epoch'] == epoch)]
        day_epoch_data_fallback = df_fallback[(df_fallback['day'] == day) & (df_fallback['epoch'] == epoch)]
        
        # Create lists to store variable values for each area
        area_values = {area: [] for area in area_cols}
        
        for idx in day_epoch_data['combined_index'].unique():
            # Filter data for each unique index within the day-epoch combination
            index_data = day_epoch_data[day_epoch_data['combined_index'] == idx]
            
            # Check if both areas are represented for this unique index
            if all(area in index_data['area'].values for area in area_cols):
                for area in area_cols:
                    # Get the variable value for this area and unique index
                    area_value = index_data[index_data['area'] == area][variable].values[0]
                    area_values[area].append(area_value)
            else:
                # Look for corresponding data in the fallback DataFrame
                index_data_fallback = day_epoch_data_fallback[day_epoch_data_fallback['combined_index'] == idx]
                if all(area in index_data_fallback['area'].values for area in area_cols):
                    for area in area_cols:
                        area_value = index_data_fallback[index_data_fallback['area'] == area][variable].values[0]
                        area_values[area].append(area_value)
        
        # Calculate correlation for this day-epoch combination if both areas have data
        if all(len(values) > 0 for values in area_values.values()):
            day_epoch_corr = np.corrcoef(area_values[area_cols[0]], area_values[area_cols[1]])[0, 1]
            
            # Calculate weight for this day-epoch combination (number of unique elements)
            weight = len(area_values[area_cols[0]])
            
            # Update weighted sum and total weight
            weighted_sum_corr += day_epoch_corr * weight
            total_weight += weight
            
            # Update DataFrame with day-epoch count
            day_epoch_count_df.loc[len(day_epoch_count_df)] = [day, epoch, weight]
    
    # Calculate weighted average correlation
    weighted_avg_corr = weighted_sum_corr / total_weight if total_weight > 0 else np.nan
    
    return weighted_avg_corr, day_epoch_count_df

# Since sample data is not desired, the function is not executed here
# You can try running this function with your 'complete_data' and 'df_fallback' DataFrames to check its functionality

compute_weighted_correlations_with_fallback(blob_df, complete_data)


# In[41]:


#just to check the above function
def unique_time_chunks(df, day, epoch):
    subset_df = df[(df['day'] == day) & (df['epoch'] == epoch)]
    return len(subset_df['time_chunk'].unique())

unique_time_chunks(complete_data, 1,5)


# In[ ]:


this does not match the value we obainted above (21). need to fix this function. 

