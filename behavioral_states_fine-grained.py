#!/usr/bin/env python
# coding: utf-8

# In[1]:


import data_analysis_utilitites as da_utilities
import data_analysis_areas as da_area
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pipeline.utilities as u
import glob
import json


# In[2]:


def load_all_criticality_data(directory, animals, area=None, state=None, day=None, epoch=None, time_chunk=None):
    all_dfs = []

    for animal in animals:
        # Create the file pattern based on the provided parameters
        pattern = f'{directory}/{animal}*_{area or "*"}_{state or "*"}_{day or "*"}_{epoch or "*"}_{time_chunk or "*"}.parquet'

        # Get a list of files that match the pattern
        files = glob.glob(pattern)

        # Read and concatenate all files into a single DataFrame for the current animal
        dfs = []
        for file in files:
            df = pd.read_parquet(file)
            
            # Optional: If you have serialized complex objects like 'pcov', you can deserialize them here
            # For example:
            df['pcov'] = df['pcov'].apply(json.loads)
            
            df['animal'] = animal
            dfs.append(df)

        animal_df = pd.concat(dfs)
        all_dfs.append(animal_df)

    # Combine the DataFrames for all animals
    combined_df = pd.concat(all_dfs)

    # Set the index to the 'animal' column
    combined_df.set_index('animal', inplace=True)

    return combined_df

data = load_all_criticality_data('/home/bellijjy/criticality_analysis', ['_dud'], area=None, state=None, day=None, epoch=None, time_chunk=None)


# In[3]:


data


# In[3]:


def extract_all_time_indices_by_animal(df, column_name):
    """Extract all keys from JSON-like strings in the specified column of the dataframe and append them to a dictionary.
    If an animal is specified, only extract keys from rows with that animal as the index."""
    
    keys = {}
    
    for (day, epoch, time_chunk), sub_row in df.groupby(['day', 'epoch', 'time_chunk']):
        original_data = eval(sub_row.iloc[0]['original_data'])  # Assuming only one row per group
        key_list = list(original_data.keys())
        min_key = float(key_list[0]) / 1500
        max_key = float(key_list[-1]) / 1500 # min and max key are the first and last time-point within each time-chunk
        
        keys[(day, epoch, time_chunk)] = [min_key, max_key]
        
    return keys


# In[4]:


import glob
import re
import pandas as pd

def load_all_criticality_data_behav_states(directory, animals, area=None, state=None, day=None, epoch=None, time_chunk=None):
    all_dfs = []
    details_dict = {}
    
    for animal in animals:
        # Generate the search pattern
        if time_chunk:
            pattern = f'{directory}/{animal}*_{area or "*"}_{state or "*"}_{day or "*"}_{epoch or "*"}_{time_chunk}.parquet'
        else:
            pattern = f'{directory}/{animal}_{area or "*"}_{state or "*"}_{day or "*"}_{epoch or "*"}.pkl'
        
        # Find files that match the pattern
        files = glob.glob(pattern)
        
        
        dfs = []
        for file in files:
            # Extract details from the filename
            match = re.search(r'{}/{}_([a-zA-Z0-9]+)_([a-zA-Z]+)_([0-9]+)_([0-9]+)(?:_[0-9]+)?\.parquet\.pkl'.format(directory, animal), file)
            if match:
                groups = match.groups()
                area, state, day, epoch = groups[:4]
                

                # Read data from the file into a DataFrame
                if file.endswith('.pkl'):
                    df = pd.read_pickle(file)
                else:
                    df = pd.read_parquet(file)
                
                # Use details as a multi-index key
                key = (area, state, int(day), int(epoch))
                details_dict[key] = df
    
    return details_dict

details_dict = load_all_criticality_data_behav_states('/home/bellijjy/', ['_dud'])


# In[6]:


details_dict['CA1', 'wake', 2, 4]['behavior_state'][2]


# In[ ]:


def associate_behav_states(df, details_dict):
    for animal in df.index.unique():
        animal_data = df.loc[animal]

        # Sort the animal data by day, epoch, and time_chunk
        animal_data = animal_data.sort_values(by=['day', 'epoch', 'time_chunk'])
        animal_data['behav_state_matches'] = animal_data.apply(lambda x: [], axis=1)  # Initialize with empty lists

        # Extract min and max time indices for the animal
        time_indices = extract_all_time_indices_by_animal(animal_data, 'original_data')

        for index, row in animal_data.iterrows():
            area, state, day, epoch = row[['area', 'state', 'day', 'epoch']]
            min_time, max_time = time_indices[(day, epoch, row['time_chunk'])]

            # Access the behavior state data from the dictionary
            behav_states = details_dict.get((area, state, day, epoch), {}).get('behavior_state', {})
            behav_series = pd.Series(behav_states)  # Convert to a Pandas Series

            # Initialize variables for tracking states
            current_state = None
            start_time = None
            match_info = []
            prev_time = None

            for time_key, state in behav_series.items():
                # State Change Detection
                if state != current_state:
                    # Handling the Previous State (if current_state is not None)
                    if current_state is not None:
                        end_time = prev_time  # End time of the previous state
                        match_type = determine_match_type(start_time, end_time, min_time, max_time)
                        match_info.append((current_state, match_type))

                    # Updating for New State
                    current_state = state
                    start_time = pd.to_timedelta(time_key).total_seconds()  # Update start_time at the change of state

                # Updating prev_time in every iteration
                prev_time = pd.to_timedelta(time_key).total_seconds()

            # Handling the Last State
            if current_state is not None:
                end_time = pd.to_timedelta(behav_series.index[-1]).total_seconds()
                match_type = determine_match_type(start_time, end_time, min_time, max_time)
                match_info.append((current_state, match_type))
            
            # Convert 'behav_state_matches' to a list (if necessary) and extend it
            match_cell = animal_data.at[index, 'behav_state_matches']
            
            if not isinstance(match_cell, list):
                match_cell = list(match_cell)
                
            match_cell.extend(match_info)
            
            # Convert list to JSON string to store in DataFrame
            animal_data.at[index, 'behav_state_matches'] = json.dumps(match_cell)

        # Group by day and epoch, then forward fill within each group
        animal_data['behav_state_matches'] = animal_data.groupby(['day', 'epoch'])['behav_state_matches'].ffill()

        # Update the main DataFrame with the processed data
        df.loc[animal] = animal_data

    return df

def determine_match_type(start_time, end_time, min_time, max_time):
    # Classification of match type based on time intervals
    if start_time >= min_time and end_time <= max_time:
        return "complete"
    elif (start_time >= min_time and start_time <= max_time) or (end_time >= min_time and end_time <= max_time):
        return "partial"
    else:
        return "none"

result = associate_behav_states(data, details_dict)


# In[ ]:


result

