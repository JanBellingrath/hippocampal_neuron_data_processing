#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle

def load_data_pickle(file_path, file_format='pickle', index_cols=None):
    """
    Loads a DataFrame from a file in various formats, with the option to specify multi-index columns.

    Parameters:
    -----------
    file_path : str
        The path to the file.
    file_format : str
        The format of the file ('parquet', 'csv', 'excel', 'json', 'pickle').
    index_cols : list or None
        A list of column names or indices to be used as the multi-index. If None, no multi-index is assumed.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the data loaded from the file.
    """
    
    try:
        if file_format == 'parquet':
            return pd.read_parquet(file_path)
        elif file_format == 'csv':
            return pd.read_csv(file_path, index_col=index_cols)
        elif file_format == 'excel':
            return pd.read_excel(file_path, index_col=index_cols)
        elif file_format == 'json':
            return pd.read_json(file_path, orient='split' if index_cols else 'records')
        elif file_format == 'pickle':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        print(f"Failed to load data from {file_path}. Error: {e}")
        return None

complete_data = load_data_pickle('/home/bellijjy/final_data_model.pickle')
complete_data


# In[52]:


import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
import matplotlib.ticker as ticker

def plot_violin_with_weights(dataframe, x, y, threshold = 30):
    sns.set_style("white")

    # Filter the DataFrame to include categories with more than 'threshold' data points
    filtered_df = dataframe.groupby(x).filter(lambda group: len(group) > threshold)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
     # Determine the order and create a color palette based on the number of observations in each category
    category_counts = filtered_df[x].value_counts()
    categories_ordered = category_counts.index.to_list()
    palette = sns.color_palette("viridis", n_colors=len(categories_ordered))

    # Create the violin plot with the 'count' scale to give weight according to the number of observations
    sns.violinplot(x=x, y=y, data=filtered_df, scale='count', inner="quartile", palette=palette, ax=ax1)
    
    # Title and labels
    ax1.set_title('Distance from criticality by task', fontsize=16, fontweight='bold')
    ax1.set_ylabel( r'closer         Distance to Criticality         farther', fontsize=16)
    ax1.set_ylim(-8, 2)
    
    #Increase the size of the tick labels
    ax1.tick_params(axis='x', labelsize=14) # For x-axis
    ax1.tick_params(axis='y', labelsize=12) # For y-axis

    # Get current y-ticks and format them without decimal points
    yticks = ['{:.0f}'.format(y) for y in ax1.get_yticks()]
    # Remove the uppermost two labels
    yticks = yticks[:-1]
    ax1.set_yticklabels(yticks, fontsize=16)
    
    ax1.set_xlabel('Task (relative extent of outbound tasks to all tasks)', fontsize=16)
    
    xticks = [0, 0.33, 0.5, .066, 1]
    ax1.set_xticklabels(xticks, fontsize=16)
    

    
    # Add significant signs if needed here
    # Example: ax1.text(0.5, max_value + offset, '*', transform=ax1.transAxes, ha='center', va='bottom')
    # Draw a downward-pointing arrow outside the left side of the figure
    arrow_props = dict(arrowstyle='-|>', color='black', linewidth=2)
    plt.annotate('', xy=(-0.10, 0.90), xytext=(-0.10, 0.10),
                 arrowprops=arrow_props, fontsize=14, ha='center', va='center', xycoords='axes fraction')

    plt.tight_layout()
    plt.show()

#Example usage:
plot_violin_with_weights(dataframe=complete_data, x='relative_outbound', y='distance_to_criticality', threshold= 30)


# In[57]:


import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
import matplotlib.ticker as ticker

def plot_violin_with_weights(dataframe, x, y, threshold = 30):
    sns.set_style("white")

    # Filter the DataFrame to include categories with more than 'threshold' data points
    filtered_df = dataframe.groupby(x).filter(lambda group: len(group) > threshold)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
     # Determine the order and create a color palette based on the number of observations in each category
    category_counts = filtered_df[x].value_counts()
    categories_ordered = category_counts.index.to_list()
    palette = sns.color_palette("viridis", n_colors=len(categories_ordered))

    # Create the violin plot with the 'count' scale to give weight according to the number of observations
    sns.violinplot(x=x, y=y, data=filtered_df, scale='count', inner="quartile", palette=palette, ax=ax1)
    
    # Title and labels
    ax1.set_title('Distance from criticality by task performace', fontsize=16, fontweight='bold')
    ax1.set_ylabel( r'closer         Distance to Criticality         farther', fontsize=16)
    ax1.set_ylim(-8, 2)
    
    #Increase the size of the tick labels
    ax1.tick_params(axis='x', labelsize=14) # For x-axis
    ax1.tick_params(axis='y', labelsize=12) # For y-axis

    # Get current y-ticks and format them without decimal points
    yticks = ['{:.0f}'.format(y) for y in ax1.get_yticks()]
    # Remove the uppermost two labels
    yticks = yticks[:-1]
    ax1.set_yticklabels(yticks, fontsize=16)
    
    ax1.set_xlabel('Task performance (relative extent of correctly fulfilled tasks)', fontsize=16)
    
    xticks = [0, 0.33, 0.5, .066, 1]
    ax1.set_xticklabels(xticks, fontsize=16)
    
    arrow_props = dict(arrowstyle='-|>', color='black', linewidth=2)
    plt.annotate('', xy=(-0.10, 0.90), xytext=(-0.10, 0.10),
                 arrowprops=arrow_props, fontsize=14, ha='center', va='center', xycoords='axes fraction')

    plt.tight_layout()
    plt.show()

#Example usage:
plot_violin_with_weights(dataframe=complete_data, x='relative_correct', y='distance_to_criticality', threshold= 30)


# In[65]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_interaction_means(dataframe, x, y, hue, threshold=30):
    sns.set_style("white")

    # Filter the DataFrame to include categories with more than 'threshold' data points in the x-axis variable
    filtered_df = dataframe.groupby(x).filter(lambda group: len(group) > threshold)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the line plot with hue for the interaction
    sns.lineplot(x=x, y=y, hue=hue, data=filtered_df, ax=ax, palette='viridis', marker='o', err_style="bars", ci=95)

    # Title and labels
    ax.set_title('Interaction Plot of Means', fontsize=16, fontweight='bold')
    ax.set_xlabel('Relative Correct', fontsize=14)
    ax.set_ylabel('Average Distance to Criticality', fontsize=14)

    # Increase the size of the tick labels and rotate x-ticks for better readability
    ax.tick_params(axis='both', labelsize=12)
    plt.xticks(rotation=45)

    # Adjust the legend
    ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

# Example usage:
plot_interaction_means(dataframe=complete_data, x='relative_correct', y='distance_to_criticality', hue='area', threshold=30)


# In[66]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_interaction_means(dataframe, x, y, hue, threshold=30):
    sns.set_style("white")

    # Filter the DataFrame to include categories with more than 'threshold' data points in the x-axis variable
    filtered_df = dataframe.groupby(x).filter(lambda group: len(group) > threshold)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the line plot with hue for the interaction
    sns.lineplot(x=x, y=y, hue=hue, data=filtered_df, ax=ax, palette='viridis', marker='o', err_style="bars", ci=95)

    # Title and labels
    ax.set_title('Interaction Plot of Means', fontsize=16, fontweight='bold')
    ax.set_xlabel('Relative Correct', fontsize=14)
    ax.set_ylabel('Average Distance to Criticality', fontsize=14)

    # Increase the size of the tick labels and rotate x-ticks for better readability
    ax.tick_params(axis='both', labelsize=12)
    plt.xticks(rotation=45)

    # Adjust the legend
    ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

# Example usage:
plot_interaction_means(dataframe=complete_data, x='relative_outbound', y='distance_to_criticality', hue='area', threshold=30)


# In[70]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_interaction_means(dataframe, x, y, hue, threshold=30):
    sns.set_style("white")

    # Filter the DataFrame to include categories with more than 'threshold' data points in the x-axis variable
    filtered_df = dataframe.groupby(x).filter(lambda group: len(group) > threshold)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the line plot with hue for the interaction
    sns.lineplot(x=x, y=y, hue=hue, data=filtered_df, ax=ax, palette='viridis', marker='o', err_style="bars", ci=95)

    # Title and labels
    ax.set_title('Interaction Plot of Means', fontsize=16, fontweight='bold')
    ax.set_xlabel('Relative Correct', fontsize=14)
    ax.set_ylabel('Average Distance to Criticality', fontsize=14)

    # Increase the size of the tick labels and rotate x-ticks for better readability
    ax.tick_params(axis='both', labelsize=12)
    plt.xticks(rotation=45)

    # Adjust the legend
    ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

# Example usage:
plot_interaction_means(dataframe=complete_data, x='relative_correct', y='distance_to_criticality', hue='relative_outbound', threshold=30)


# In[73]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_means(dataframe, x, y, threshold=30):
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot of means
    sns.scatterplot(x=x, y=y, data=dataframe, ax=ax)

    # Title and labels
    ax.set_title(f'Mean {y} by {x}', fontsize=16, fontweight='bold')
    ax.set_xlabel(x, fontsize=14)
    ax.set_ylabel(f'Mean {y}', fontsize=14)

    # Increase the size of the tick labels
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    plt.show()

# Example usage:
plot_means(dataframe=complete_data, x='linear_speed', y='distance_to_criticality', threshold=30)


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_means_with_category(dataframe, x, y, category, highlight, threshold=30):
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot of means
    sns.scatterplot(x=x, y=y, data=dataframe, ax=ax, color='blue')

    # Scatter plot for specific category
    category_data = dataframe[dataframe[category] == highlight]
    sns.scatterplot(x=x, y=y, data=category_data, ax=ax, color='yellow', label=f'{category}: {highlight}')

    # Title and labels
    ax.set_title('Distance to criticality by speed', fontsize=16, fontweight='bold')
    ax.set_xlabel(x, fontsize=14)
    ax.set_ylabel('Distance to criticaltiy', fontsize=14)
    #ax.set_ylim(-8,)
   
    
    # Increase the size of the tick labels
    ax.tick_params(axis='both', labelsize=14)



    plt.tight_layout()
    plt.show()

# Example usage:
plot_means_with_category(dataframe=complete_data, x='linear_speed', y='distance_to_criticality', category='area', highlight='specific_area', threshold=30)


# In[ ]:




