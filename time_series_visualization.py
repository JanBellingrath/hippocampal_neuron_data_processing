#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[30]:


df_sorted = complete_data.sort_values(by=['animal', 'area', 'day', 'epoch', 'time_chunk']).reset_index(drop=True)


# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to generate a single population rate plot
def generate_population_rate_plot(df, i, ax, total_duration=30, palette='icefire'):
    try:
        # Extract the row data
        row = df.iloc[i]
        condition = "above 2.5 seconds" if row['tau'] > 2500 else "below 2.5 seconds"
        dt = total_duration / len(row['data'])
        spike_data = np.array(row['data']).astype(int)
        time_bins = np.linspace(0, total_duration, num=total_duration + 1)
        binned_spike_rate, _ = np.histogram(np.nonzero(spike_data)[0] * dt, bins=time_bins)
        time_array = time_bins[:-1]
        color = sns.color_palette(palette)[0 if condition == "above 2.5 seconds" else 1]
        sns.lineplot(x=time_array, y=binned_spike_rate, drawstyle='steps-pre', color=color, ax=ax)
        ax.set_title(f"{condition}\n(Tau = {row['tau']:.2f})", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Population Rate (Hz)")
        ax.set_xlim(0, total_duration)
        ax.set_ylim(0, np.max(binned_spike_rate) + 1)
    except Exception as e:
        print(f"An error occurred while generating the plot for index {i}: {e}")

# Function to plot 5 consecutive chunks with the middle one being critical
def plot_consecutive_chunks_with_critical(df, total_duration=30, n=1):
    df_sorted = df.sort_values(by=['animal', 'area', 'day', 'epoch', 'time_chunk']).reset_index(drop=True)
    try:
        critical_indices = df[df['tau'] > 2500].index
        plotted_combinations = set()  # Reset plotted combinations for each function call
            
        for _ in range(n):
            for critical_index in critical_indices:
                # Get the critical row and its combination
                critical_row = df.iloc[critical_index]
                critical_combination = tuple(critical_row[['animal', 'area', 'day', 'epoch']])
                
                # Skip if we've already plotted this combination
                if critical_combination in plotted_combinations:
                    continue
                
                # Find the indices for the surrounding chunks
                surrounding_indices = list(range(max(0, critical_index - 2), min(len(df), critical_index + 3)))

                # Ensure we have exactly 5 indices, skip otherwise
                if len(surrounding_indices) != 5:
                    
                    continue
                

                # Check if the surrounding indices are consecutive and match the critical combination
                if not all(tuple(df.iloc[i][['animal', 'area', 'day', 'epoch']]) == critical_combination and
                           df.iloc[i]['time_chunk'] == df.iloc[surrounding_indices[2]]['time_chunk'] - (2 - surrounding_indices.index(i))
                           for i in surrounding_indices):
                    continue

                # Add combination to plotted set
                plotted_combinations.add(critical_combination)
                #print(surrounding_indices)
                # Plot the selected chunks
                fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
                for plot_index, data_index in enumerate(surrounding_indices):
                    generate_population_rate_plot(df, data_index, axes[plot_index], total_duration)
                plt.tight_layout()
                plt.show()

                # Break after plotting n pairs
                if len(plotted_combinations) >= n:
                    return
            print(f"Finished plotting for {n} pairs.")
    except Exception as e:
        print(f"An error occurred while processing the data: {e}")

plot_consecutive_chunks_with_critical(df_sorted, n= 50)


# In[ ]:




