#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
import numpy as np
import seaborn as sns
import pandas as pd


def load_from_parquet(file_path):
    """
    Loads a DataFrame fhttp://lux18.ini.rub.de:8888/notebooks/Mixed-Effects%20Model%20with%20Position.ipynb#rom a Parquet file.

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


# In[4]:


# Function to generate a 3D interaction plot for area, day, and state
def analyze_and_plot_3D_interaction_by_area_day_state(dataframe, metric):
    sns.set_style("whitegrid")
    
    # Compute max_count and relative_counts for 'day' and 'state'
    day_state_counts = dataframe.groupby(['day', 'state']).size().reset_index(name='counts')
    max_count_day_state = day_state_counts['counts'].max()

    # Get colormap for 'day' and 'state'
    cmap_day_state = plt.get_cmap('viridis')

    # Initialize the 3D plot
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(1, 2, width_ratios=[1, 0.05])
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax2 = fig.add_subplot(gs[1])  # this will be used for the colorbar
    
    # Area to numerical mapping (for plotting purposes)
    area_to_num = {'CA1': 1, 'CA3': 2}
    
    # Create the plot
    for area in ["CA1", "CA3"]:
        for day in np.sort(dataframe['day'].unique()):
            for state in ["wake", "sleep"]:
                subset = dataframe[(dataframe['area'] == area) & (dataframe['day'] == day) & (dataframe['state'] == state)]
                if len(subset) > 0:
                    count_day_state = len(subset)
                    color = cmap_day_state(count_day_state / max_count_day_state)
                    marker = 'o' if state == 'wake' else 's'
                    ax1.scatter(day, area_to_num[area], np.mean(subset[metric]), c=[color], marker=marker, s=50)
                    ax1.plot([day, day], [area_to_num[area], area_to_num[area]], [0, np.mean(subset[metric])], c='grey', linestyle='--', linewidth=0.5)

    # Set the axis labels and title
    ax1.set_xlabel('Day', fontsize=14)
    ax1.set_ylabel('Area', fontsize=14)
    ax1.set_zlabel(f'{metric.capitalize()}', fontsize=14)
    ax1.set_title(f'3D Interaction between Area, Day, and State on {metric.capitalize()}', fontsize=16, fontweight='bold')
    
    # Setting the yticks to represent the areas
    ax1.set_yticks([1, 2])
    ax1.set_yticklabels(['CA1', 'CA3'])
    
    # Adding colorbar for color intensity
    sm = ScalarMappable(cmap=cmap_day_state, norm=plt.Normalize(0, max_count_day_state))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax2, orientation="vertical", label="Relative Number of Data Points per Day")
    
    # Adjust layout
    gs.tight_layout(fig, rect=[0, 0.05, 0.95, 1])
    
    # Adding legend for state with custom markers
    legend_labels = ["Wake", "Sleep"]
    legend_markers = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10),
                      plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10)]
    ax1.legend(legend_markers, legend_labels, title="State", loc="upper right")
    
    plt.show()

# Sample usage with hypothetical data; replace 'data' and 'metric' with your actual dataframe and metric of interest.
analyze_and_plot_3D_interaction_by_area_day_state_professional(complete_data, 'tau')


# In[9]:


def analyze_and_plot_3D_interaction_general_v2(dataframe, x, y, z):
    sns.set_style("whitegrid")
    
    # Detect data types for x and y
    x_is_categorical = pd.api.types.is_categorical_dtype(dataframe[x]) or dataframe[x].dtype == 'object'
    y_is_categorical = pd.api.types.is_categorical_dtype(dataframe[y]) or dataframe[y].dtype == 'object'
    
    # Create mappings for categorical variables
    x_mapping, y_mapping = None, None
    if x_is_categorical:
        x_mapping = {v: i for i, v in enumerate(sorted(dataframe[x].unique()))}
    if y_is_categorical:
        y_mapping = {v: i for i, v in enumerate(sorted(dataframe[y].unique()))}
    
    # Initialize the 3D plot
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(1, 2, width_ratios=[1, 0.05])
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax2 = fig.add_subplot(gs[1])  # this will be used for the colorbar
    
    # Get colormap for z
    cmap_z = plt.get_cmap('viridis')
    max_z = dataframe[z].max()
    min_z = dataframe[z].min()
    
    # Create the plot
    for xi in np.sort(dataframe[x].unique()):
        for yi in np.sort(dataframe[y].unique()):
            subset = dataframe[(dataframe[x] == xi) & (dataframe[y] == yi)]
            if len(subset) > 0:
                mean_z = np.mean(subset[z])
                color = cmap_z((mean_z - min_z) / (max_z - min_z))
                plot_x = x_mapping[xi] if x_is_categorical else xi
                plot_y = y_mapping[yi] if y_is_categorical else yi
                ax1.scatter(plot_x, plot_y, mean_z, c=[color], s=50)
                ax1.plot([plot_x, plot_x], [plot_y, plot_y], [0, mean_z], c='grey', linestyle='--', linewidth=0.5)

    # Set the axis labels and title
    ax1.set_xlabel(f'{x}', fontsize=14)
    ax1.set_ylabel(f'{y}', fontsize=14)
    ax1.set_zlabel(f'{z}', fontsize=14)
    ax1.set_title(f'2D Interaction between {x}, and {y}, on {z}', fontsize=16, fontweight='bold')
    
    # Modify axis ticks for categorical variables
    if x_is_categorical:
        ax1.set_xticks(list(x_mapping.values()))
        ax1.set_xticklabels(list(x_mapping.keys()))
    if y_is_categorical:
        ax1.set_yticks(list(y_mapping.values()))
        ax1.set_yticklabels(list(y_mapping.keys()))
    
    # Adding colorbar for color intensity
    sm = ScalarMappable(cmap=cmap_z, norm=plt.Normalize(min_z, max_z))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax2, orientation="vertical", label=f"{z}")
    
    # Adjust layout
    gs.tight_layout(fig, rect=[0, 0.05, 0.95, 1])
    
    plt.show()

# Sample usage with hypothetical data; replace 'data' and the axes with your actual dataframe and metric of interest.
analyze_and_plot_3D_interaction_general_v2(complete_data, 'linear_distance', 'area', 'tau')


# In[11]:


analyze_and_plot_3D_interaction_general_v2(complete_data, 'linear_speed', 'linear_distance', 'tau')


# In[ ]:




