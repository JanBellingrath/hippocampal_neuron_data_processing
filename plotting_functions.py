#!/usr/bin/env python
# coding: utf-8

# In[2]:


import data_analysis_areas as da_area
import data_analysis_utilitites as da_utilities
import data_analysis_sleep_wake as da_wasl


# In[3]:


data = da_utilities.load_all_criticality_data_no_duplicate_files('/home/bellijjy/criticality_january_sm', ['_fra','_dud', '_con', '_cha','_Cor', '_egy', '_gov','_dav'], area=None, state=None, day=None, epoch=None, time_chunk=None)
data


# In[12]:


import numpy as np
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

data = add_distance_to_criticality_column(data)
data = data[data['tau'] <= 4999.9]


# In[13]:


da_wasl.analyze_and_plot_average_epochstate_by_animal(data, 'tau')


# In[5]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_tau_over_days_by_area(dataframe):
    sns.set_style("white")

    # Filtering the DataFrame for only CA1 and CA3 areas
    filtered_df = dataframe[dataframe['area'].isin(['CA1', 'CA3'])]

    # Sorting the DataFrame by 'day'
    sorted_df = filtered_df.sort_values(by='epoch')

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=sorted_df, x='epoch', y='distance_to_criticality', hue='area', marker='o', palette='icefire')

    plt.title('Timescale over Days for CA1 and CA3 (for both behavioral states)', fontsize=16, fontweight='bold')
    plt.xlabel('Day', fontsize=16)  # Increased font size for axis labels
    plt.ylabel( r'closer         Distance to Criticality         farther', fontsize=16)  # Increased font size for axis labels
    plt.legend(title='Area', fontsize=12)
    plt.xticks(fontsize=14)  # Increased font size for ticks
    plt.yticks(fontsize=14)  # Increased font size for ticks
    
        # Customize the legend
    legend = plt.legend(title='Area', fontsize=14, title_fontsize=16)
    for text in legend.get_texts():
        text.set_fontsize(16)  # Increase the font size of the legend labels
    
    # Draw a downward-pointing arrow outside the left side of the figure
    arrow_props = dict(arrowstyle='-|>', color='black', linewidth=2)
    plt.annotate('', xy=(-0.125, 0.925), xytext=(-0.125, 0.125),
                 arrowprops=arrow_props, fontsize=14, ha='center', va='center', xycoords='axes fraction')

    plt.tight_layout()
    plt.show()

# Assuming 'df' is your DataFrame
plot_tau_over_days_by_area(data)


# In[6]:


def plot_tau_over_days_by_state(dataframe):
    sns.set_style("white")

    # Filtering the DataFrame for only CA1 and CA3 areas
    filtered_df = dataframe[dataframe['state'].isin(['wake', 'sleep'])]

    # Sorting the DataFrame by 'day'
    sorted_df = filtered_df.sort_values(by='day')

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=sorted_df, x='day', y='distance_to_criticality', hue='state', marker='o', palette='icefire')

    plt.title('Timescale over Days for CA1 and CA3 (for both behavioral states)', fontsize=16, fontweight='bold')
    plt.xlabel('Day', fontsize=16)  # Increased font size for axis labels
    plt.ylabel( r'closer         Distance to Criticality         farther', fontsize=16)  # Increased font size for axis labels
    plt.legend(title='State', fontsize=12)
    plt.xticks(fontsize=14)  # Increased font size for ticks
    plt.yticks(fontsize=14)  # Increased font size for ticks
    
        # Customize the legend
    legend = plt.legend(title='State', fontsize=14, title_fontsize=16)
    for text in legend.get_texts():
        text.set_fontsize(16)  # Increase the font size of the legend labels
    
    # Draw a downward-pointing arrow outside the left side of the figure
    arrow_props = dict(arrowstyle='-|>', color='black', linewidth=2)
    plt.annotate('', xy=(-0.125, 0.925), xytext=(-0.125, 0.125),
                 arrowprops=arrow_props, fontsize=14, ha='center', va='center', xycoords='axes fraction')

    plt.tight_layout()
    plt.show()

# Assuming 'df' is your DataFrame
plot_tau_over_days_by_state(data)


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def plot_tau_by_state(dataframe):
    sns.set_style("white")

    plt.figure(figsize=(10, 6))

    # Create a palette dictionary to map states to colors
    palette = sns.color_palette('viridis', n_colors=len(dataframe['state'].unique()))

    # Plotting the line for each state without confidence interval
    sns.lineplot(data=dataframe, x='area', y='distance_to_criticality', hue='state', marker='o', palette=palette, errorbar=None, style='state', dashes=False)

    # Calculate the mean, standard error, and confidence interval for each group
    grouped = dataframe.groupby(['area', 'state'])['distance_to_criticality'].agg(['mean', 'sem', 'count']).reset_index()
    # Calculate the 95% confidence interval using the t-distribution
    confidence = 0.95
    grouped['ci'] = grouped['sem'] * stats.t.ppf((1 + confidence) / 2, grouped['count'] - 1)

    # Create a dictionary to map states to colors
    color_dict = dict(zip(dataframe['state'].unique(), palette))

    # Add error bars for each point, representing the confidence interval
    for _, row in grouped.iterrows():
        plt.errorbar(x=row['area'], y=row['mean'], yerr=row['ci'], fmt='none', color=color_dict[row['state']], capsize=5, elinewidth=2)

    plt.title('Timescale by Interaction of Subregion * State', fontsize=16, fontweight='bold')
    plt.xlabel('Subregion', fontsize=16)
    plt.ylabel( r'closer         Distance to Criticality         farther', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Customize the legend'
    legend = plt.legend(title='State', fontsize=14, title_fontsize=16)
    for text in legend.get_texts():
        text.set_fontsize(16)
    
     # Draw a downward-pointing arrow outside the left side of the figure
    arrow_props = dict(arrowstyle='-|>', color='black', linewidth=2)
    plt.annotate('', xy=(-0.125, 0.925), xytext=(-0.125, 0.125),
                 arrowprops=arrow_props, fontsize=14, ha='center', va='center', xycoords='axes fraction')

    plt.tight_layout()
    plt.show()

# Assuming 'df' is your DataFrame
plot_tau_by_state(data)


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
from matplotlib.patches import FancyArrowPatch

def analyze_and_plot_average_epoch_by_area(dataframe, metric):
    sns.set_style("white")

    # Increase the figure width to accommodate the arrow outside the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Adjust the width as needed

    # Creating a color palette
    unique_states = dataframe['area'].unique()
    palette = sns.color_palette("icefire", n_colors=len(unique_states))

    # Violin plot with different hues for different states
    sns.violinplot(x='area', y=metric, data=dataframe, ax=ax1, inner="quartile", palette=palette)
    ax1.set_title('Distribution of Distance to Criticality by Subregion', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Subregion', fontsize=16)
    ax1.set_ylabel( r'closer         Distance to Criticality         farther', fontsize=16)
    ax1.set_ylim(-8, 2)

    # Get current y-ticks and format them without decimal points
    yticks = ['{:.0f}'.format(y) for y in ax1.get_yticks()]
    # Remove the uppermost two labels
    yticks = yticks[:-1]
    ax1.set_yticklabels(yticks, fontsize=16)

    # Make mean/quartile lines thicker
    for l in ax1.lines:
        l.set_linewidth(2)

    # Increase the size of the x-axis tick labels
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=16)

    # Manually add asterisks between the first two groups
    x_coord = 0.5  # Halfway between the first two groups
    y_coord = max(dataframe[dataframe['area'] == unique_states[0]][metric].max(),
                  dataframe[dataframe['area'] == unique_states[1]][metric].max()) + 3  # Slightly above the max value of the two groups
    ax1.text(x_coord, y_coord, "*", horizontalalignment='center', verticalalignment='center', fontsize=20, color='black')

    # Manually adding statistical annotation using statannot
    box_pairs = [('wake', 'sleep')]  # Replace with your actual state names
    add_stat_annotation(ax1, data=dataframe, x='state', y=metric, box_pairs=box_pairs,
                        perform_stat_test=False, pvalues=[0.0001],
                        pvalue_thresholds=[(1e-4, '****'), (1e-3, '***'), (1e-2, '**'), (0.05, '*')],
                        loc='inside', text_format='star', line_height=0.22, text_offset=1)

    # Draw a downward-pointing arrow outside the left side of the figure
    arrow_props = dict(arrowstyle='-|>', color='black', linewidth=2)
    plt.annotate('', xy=(-0.1, 0.9), xytext=(-0.1, 0.1),
                 arrowprops=arrow_props, fontsize=14, ha='center', va='center', xycoords='axes fraction')

    plt.tight_layout()
    plt.show()

# Assuming 'data' is your DataFrame and 'distance_to_criticality' is the metric you're interested in
analyze_and_plot_average_epoch_by_area(data, 'distance_to_criticality')


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
from matplotlib.patches import FancyArrowPatch

def analyze_and_plot_average_epoch_by_area(dataframe, metric):
    sns.set_style("white")

    # Increase the figure width to accommodate the arrow outside the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Adjust the width as needed

    # Creating a color palette
    unique_states = dataframe['state'].unique()
    palette = sns.color_palette("viridis", n_colors=len(unique_states))

    # Violin plot with different hues for different states
    sns.violinplot(x='state', y=metric, data=dataframe, ax=ax1, inner="quartile", palette=palette)
    ax1.set_title('Distribution of Distance to Criticality by State', fontsize=16, fontweight='bold')
    ax1.set_xlabel('State', fontsize=16)
    ax1.set_ylabel( r'closer         Distance to Criticality         farther', fontsize=16)
    ax1.set_ylim(-8, 2)

    # Get current y-ticks and format them without decimal points
    yticks = ['{:.0f}'.format(y) for y in ax1.get_yticks()]
    # Remove the uppermost two labels
    yticks = yticks[:-1]
    ax1.set_yticklabels(yticks, fontsize=16)

    # Make mean/quartile lines thicker
    for l in ax1.lines:
        l.set_linewidth(2)

    # Increase the size of the x-axis tick labels
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=16)

    # Manually add asterisks between the first two groups
    x_coord = 0.5  # Halfway between the first two groups
    y_coord = max(dataframe[dataframe['state'] == unique_states[0]][metric].max(),
                  dataframe[dataframe['state'] == unique_states[1]][metric].max()) + 3  # Slightly above the max value of the two groups
    ax1.text(x_coord, y_coord, "***", horizontalalignment='center', verticalalignment='center', fontsize=20, color='black')

    # Manually adding statistical annotation using statannot
    box_pairs = [('wake', 'sleep')]  # Replace with your actual state names
    add_stat_annotation(ax1, data=dataframe, x='state', y=metric, box_pairs=box_pairs,
                        perform_stat_test=False, pvalues=[0.0001],
                        pvalue_thresholds=[(1e-4, '****'), (1e-3, '***'), (1e-2, '**'), (0.05, '*')],
                        loc='inside', text_format='star', line_height=0.22, text_offset=1)

    # Draw a downward-pointing arrow outside the left side of the figure
    arrow_props = dict(arrowstyle='-|>', color='black', linewidth=2)
    plt.annotate('', xy=(-0.1, 0.9), xytext=(-0.1, 0.1),
                 arrowprops=arrow_props, fontsize=14, ha='center', va='center', xycoords='axes fraction')

    plt.tight_layout()
    plt.show()

# Assuming 'data' is your DataFrame and 'distance_to_criticality' is the metric you're interested in
analyze_and_plot_average_epoch_by_area(data, 'distance_to_criticality')


# In[ ]:


da_area.area_difference_tau(data, ['_cha', '_con', '_dud', '_egy', '_Cor', '_fra', '_dav', '_bon'])


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

def area_difference_tau(data, animals, value_column='tau'):
    """
    Plot distribution of data points for each animal, colored by the number of data points relative to max number.
    
    Parameters:
    - data: DataFrame with data
    - animals: List of animals to consider
    - value_column: Column name in data to plot (default is 'tau')
    """
    for animal in animals:
        animal_data = data.loc[animal]

        max_count = animal_data['area'].value_counts().max()
        relative_counts = animal_data['area'].value_counts() / max_count
        
        # Get colormap
        cmap = plt.get_cmap('crest')
        
        # Create a color dictionary for areas
        colors = {area: cmap(relative_counts[area]) for area in relative_counts.keys()}
        
        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Violin plot
        sns.violinplot(x='area', y=value_column, data=animal_data, inner='quartile', palette=colors, ax=ax1)
        
        # Colorbar settings
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_count))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, orientation="vertical", label="Number of Data Points")
        
        ax1.set_title(f'Distribution of {value_column} for {animal}')
        ax1.set_ylabel(value_column)
        ax1.set_xlabel('Brain Area')
        
        # Box plot
        sns.boxplot(x='area', y=value_column, data=animal_data, palette=colors, ax=ax2)
        ax2.set_title(f'Box Plot of {value_column} for {animal}')
        ax2.set_ylabel(value_column)
        ax2.set_xlabel('Brain Area')
        
        plt.tight_layout()
        plt.show()

        # Pairwise Tests (updated from pairwise_ttests)
        results = pg.pairwise_tests(data=animal_data, dv=value_column, between='area', subject='animal', correction='auto', parametric = 'auto')
        results.index = [animal] * len(results)
        display(results)
        
        # Statistical Analysis
        # 1. Levene's Test for Equality of Variances
        areas = animal_data['area'].unique()
        area_data_lists = [animal_data[animal_data['area'] == area][value_column].tolist() for area in areas]
        W_stat, p_value_lev = levene(*area_data_lists)
        display(Latex(f"Levene's Test (Test for equal Variances) for {animal}: $W$-statistic = {W_stat:.4f}, $p$ = {p_value_lev:.4f}"))
        
         # Normality tests
        for area in areas:
            shapiro_stat, shapiro_p = stats.shapiro(animal_data[animal_data['area'] == area][value_column])
            display(Latex(f"Shapiro-Wilk Normality Test for {animal} - {area}: $W$ = {shapiro_stat:.4f}, $p$ = {shapiro_p:.4f}"))
    
    return


# In[12]:


from matplotlib.cm import ScalarMappable
from scipy.stats import levene, shapiro
from scipy import stats
from IPython.display import Latex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
area_difference_tau(data, ['_con', '_dud', '_egy', '_Cor', '_fra',  '_rem'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[25]:


data[data['tau']> 4000]


# In[ ]:





# In[ ]:


import pandas as pd

def calculate_key_differences(df, column_name):
    differences = []
    for index, row in df.iterrows():
        # Assuming the column contains dictionary-like strings
        dict_data = eval(row[column_name])
        if dict_data:  # Check if the dictionary is not empty
            keys = list(map(int, dict_data.keys()))  # Convert keys to integers if they are not already
            difference = keys[-1] - keys[0]  # Calculate difference between the last and first key
            differences.append(difference)
        else:
            differences.append(None)  # Append None or a default value if the dictionary is empty

    # Create a new DataFrame with the differences
    return pd.DataFrame(differences, columns=['Key Difference'])

# Usage example
# Assuming 'data' is your original DataFrame and 'original_data' is the column with dictionaries
differences_df = calculate_key_differences(data, 'original_data')


# In[10]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_tau_over_days_by_area_3d(dataframe):
    sns.set_style("white")

    # Mapping behavioral states to numeric values
    state_mapping = {'wake': 0, 'sleep': 1}
    dataframe['state_numeric'] = dataframe['state'].map(state_mapping)

    # Filtering the DataFrame for only CA1 and CA3 areas
    filtered_df = dataframe[dataframe['area'].isin(['CA1', 'CA3'])]

    # Sorting the DataFrame by 'day'
    sorted_df = filtered_df.sort_values(by='day')

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting for each area and behavioral state
    for area in ['CA1', 'CA3']:
        for state in sorted_df['state'].unique():
            df_area_state = sorted_df[(sorted_df['area'] == area) & (sorted_df['state'] == state)]
            ax.plot(df_area_state['day'], df_area_state['state_numeric'], df_area_state['distance_to_criticality'], 
                    label=f'{area} - {state}', marker='o')

    ax.set_title('Timescale over Days for CA1 and CA3 (for both behavioral states)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Day', fontsize=16)
    ax.set_ylabel('Behavioral State', fontsize=16)
    ax.set_zlabel(r'closer         Distance to Criticality         farther', fontsize=16)
    ax.legend(title='Area & State', fontsize=12)

    # Setting the tick font size
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Customize the legend
    legend = ax.legend(title='State', fontsize=14, title_fontsize=16, loc='upper left')
    for text in legend.get_texts():
        text.set_fontsize(16)

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Assuming 'data' is your DataFrame
plot_tau_over_days_by_area_3d(data)



# In[ ]:


for index, row in differences_df.iterrows():
    for col in differences_df.columns:
        print(f"Row {index}, Column '{col}': {row[col]}")


# In[20]:


#why are sometimes 28 missing?
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_avg_tau_over_days_by_area_3d(dataframe):
    sns.set_style("white")

    # Mapping behavioral states to numeric values
    state_mapping = {'wake': 0, 'sleep': 1}
    dataframe['state_numeric'] = dataframe['state'].map(state_mapping)

    # Aggregating the data to get the average distance_to_criticality for each day, state, and area
    avg_df = dataframe.groupby(['day', 'state', 'area']).agg({'distance_to_criticality': 'mean'}).reset_index()

    # Filtering the DataFrame for only CA1 and CA3 areas
    filtered_avg_df = avg_df[avg_df['area'].isin(['CA1', 'CA3'])]

    # Sorting the DataFrame by 'day'
    sorted_avg_df = filtered_avg_df.sort_values(by='day')

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting for each area and behavioral state
    for area in ['CA1', 'CA3']:
        for state in sorted_avg_df['state'].unique():
            df_area_state = sorted_avg_df[(sorted_avg_df['area'] == area) & (sorted_avg_df['state'] == state)]
            ax.plot(df_area_state['day'], df_area_state['state_numeric'], df_area_state['distance_to_criticality'], 
                    label=f'{area} - {state}', marker='o')

    ax.set_title('Average Timescale over Days for CA1 and CA3 (for both behavioral states)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Day', fontsize=16)
    ax.set_ylabel('Behavioral State', fontsize=16)
    ax.set_zlabel(r'closer         Distance to Criticality         farther', fontsize=16)
    ax.legend(title='Area & State', fontsize=12)

    # Setting the tick font size
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Customize the legend
    legend = ax.legend(title='State', fontsize=14, title_fontsize=16, loc='upper left')
    for text in legend.get_texts():
        text.set_fontsize(16)

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Assuming 'data' is your DataFrame
plot_avg_tau_over_days_by_area_3d(data)


# In[21]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_avg_tau_over_days_by_area_3d_with_error(dataframe):
    sns.set_style("white")

    # Mapping behavioral states to numeric values
    state_mapping = {'wake': 0, 'sleep': 1}
    dataframe['state_numeric'] = dataframe['state'].map(state_mapping)

    # Aggregating the data to get the average and standard deviation of distance_to_criticality
    agg_df = dataframe.groupby(['day', 'state', 'area']).agg({'distance_to_criticality': ['mean', 'std']}).reset_index()
    agg_df.columns = ['day', 'state', 'area', 'mean_distance', 'std_distance']

    # Including state_numeric in the aggregated DataFrame
    agg_df['state_numeric'] = agg_df['state'].map(state_mapping)

    # Filtering the DataFrame for only CA1 and CA3 areas
    filtered_agg_df = agg_df[agg_df['area'].isin(['CA1', 'CA3'])]

    # Sorting the DataFrame by 'day'
    sorted_agg_df = filtered_agg_df.sort_values(by='day')

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting for each area and behavioral state with error bars
    for area in ['CA1', 'CA3']:
        for state in sorted_agg_df['state'].unique():
            df_area_state = sorted_agg_df[(sorted_agg_df['area'] == area) & (sorted_agg_df['state'] == state)]
            ax.errorbar(df_area_state['day'], df_area_state['state_numeric'], df_area_state['mean_distance'], 
                        yerr=df_area_state['std_distance'], label=f'{area} - {state}', fmt='o')

    ax.set_title('Average Timescale over Days for CA1 and CA3 (with SD)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Day', fontsize=16)
    ax.set_ylabel('Behavioral State', fontsize=16)
    ax.set_zlabel(r'closer         Distance to Criticality         farther', fontsize=16)
    ax.legend(title='Area & State', fontsize=12)

    # Setting the tick font size
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Customize the legend
    legend = ax.legend(title='State', fontsize=14, title_fontsize=16, loc='upper left')
    for text in legend.get_texts():
        text.set_fontsize(16)

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Assuming 'data' is your DataFrame
plot_avg_tau_over_days_by_area_3d_with_error(data)


# In[ ]:


#there are 9000 steps (k) from the cor coeffs. 
#from a 60 second chunk. 
#if there were 1 ms time steps we would have 60 000. as there are 1000 in one second.
#Each step in a 60-second window with 9000 steps is approximately 0.00667 seconds long, or 6.6 ms. 

