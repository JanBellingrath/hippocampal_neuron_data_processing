#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import namedtuple
import pandas as pd
from scipy.io import loadmat
import numpy as np
import pandas as pd
from os.path import join

Animal = namedtuple('Animal', {'short_name', 'directory'})

conley = Animal('/home/bellijjy/Conley.tar/Conley/con', 'con')
dave = Animal('/home/bellijjy/Dave.tar/Dave/Dave/dav', 'dav')
chapati = Animal('/home/bellijjy/Chapati.tar/Chapati/Chapati/cha', 'cha')
corriander = Animal('/home/bellijjy/Corriander.tar/Corriander/corr', '_Cor')
dudley = Animal('/home/bellijjy/Dudley/dud', 'dud')
bond = Animal('/home/bellijjy/Bond/bon', 'bon')


#sometimes, due to a error which might occurr in loading dependencies, the animal short_name (e.g '/con', etc.) may have to be
#appended below.
animals = {'_con': Animal('con','/home/bellijjy/Conley.tar/Conley'),
           '_Cor': Animal('Cor','/home/bellijjy/Corriander.tar/Corriander'),
            '_cha': Animal('cha','/home/bellijjy/Chapati.tar/Chapati/Chapati'),
          '_dav': Animal('dav','/home/bellijjy/Dave.tar/Dave/Dave'),
           '_dud': Animal('dud','/home/bellijjy/Dudley'),
            '_bon' : Animal('bon', '/home/bellijjy/Bond')}

import loren_frank_data_processing.position as pos
import loren_frank_data_processing.visualization as viz
import loren_frank_data_processing.core as core
import loren_frank_data_processing.track_segment_classification as tsc
from logging import getLogger
logger = getLogger(__name__)


# In[2]:


def associate_time_intervals(segment_info_df, labeled_segments_df):
    # Initialize a dictionary to store the start, end times, and duration for each segment
    segment_time_data = {}

    # Convert the time index of labeled_segments_df to a format that allows for computations
    labeled_segments_df.index = pd.to_timedelta(labeled_segments_df.index.astype(str))

    # Loop through each segment in segment_info_df
    for segment in segment_info_df.index:
        # Check if this segment exists in labeled_segments_df
        if segment in labeled_segments_df['labeled_segments'].values:
            # Get the start and end times for this segment from labeled_segments_df
            segment_times = labeled_segments_df.index[labeled_segments_df['labeled_segments'] == segment]
            start_time = segment_times.min()
            end_time = segment_times.max()

            # Store the start and end times, and duration in the dictionary
            segment_time_data[segment] = {
                'start_time': start_time,
                'end_time': end_time,
                'segment_duration': end_time - start_time
            }

    # Convert the dictionary to a DataFrame
    time_data_df = pd.DataFrame.from_dict(segment_time_data, orient='index')

    # Merge the time data with segment_info_df
    merged_df = segment_info_df.merge(time_data_df, left_index=True, right_index=True)

    return merged_df


# In[6]:


import pickle
import data_analysis_utilitites as da_utilities

def load_object_from_pickle(file_path: str):
    """
    Load and return an object from a pickle file.

    Parameters:
    - file_path (str): The path to the pickle file.

    Returns:
    - object: The Python object loaded from the file.
    """
    with open(file_path, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object

my_object = load_object_from_pickle('criticality_analysis_states/target_dav.pkl')

data = da_utilities.load_all_criticality_data_no_duplicate_files('/home/bellijjy/criticality_analysis', ['_con', '_dav', '_dud', '_Cor', '_cha'], area=None, state=None, day=None, epoch=None, time_chunk=None)


# In[1]:


import data_analysis_areas as da_area
import data_analysis_utilitites as da_utilities
import data_analysis_sleep_wake as da_wasl
 
data = da_utilities.load_all_criticality_data_no_duplicate_files('/home/bellijjy/criticality_january_sm', ['_fra','_dud', '_dav', '_con', '_cha', '_Cor', '_egy', '_gov'], area=None, state=None, day=None, epoch=None, time_chunk=None)
data


# In[ ]:


def associate_time_chunks_with_info_df(info_df, big_data_frame, animal, area, day, epoch):
    # Initialize a dictionary to hold the match information for each time chunk
    matches_for_time_chunks = {}

    # Filter rows based on MultiIndex
    filtered_time_chunk_info_df = big_data_frame.xs((animal, area, day, epoch), level=('animal', 'area', 'day', 'epoch'))

    # Dictionary to store the min_key and max_key for each time_chunk
    keys = {}

    # Populate min_max_keys as the biggest and smallest time-indices for each time_chunk
    for time_chunk, sub_row in filtered_time_chunk_info_df.groupby('time_chunk'):
        original_data = eval(sub_row.iloc[0]['original_data'])
        key_list = list(original_data.keys())
        min_key = float(key_list[0]) / 1000
        max_key = float(key_list[-1]) / 1000
        keys[time_chunk] = [min_key, max_key]

    # Loop through each row in info_df
    for index, row in info_df.iterrows():
        start_time = row['start_time'].total_seconds()
        end_time = row['end_time'].total_seconds()

        # Loop through each row in filtered_time_chunk_info_df to find the corresponding time_chunk
        for time_chunk, time_chunk_row in filtered_time_chunk_info_df.iterrows():
            min_key, max_key = keys.get(time_chunk, (None, None))

            # Check if start_time and end_time are fully contained within the time_chunk
            if min_key is not None and max_key is not None:
                complete_fit = None

                if min_key <= start_time <= max_key and min_key <= end_time <= max_key:
                    complete_fit = "complete"
                elif min_key <= start_time <= max_key or min_key <= end_time <= max_key:
                    complete_fit = "partial"

                if complete_fit is not None:
                    # Initialize lists in the dictionary for a new time chunk
                    if time_chunk not in matches_for_time_chunks:
                        matches_for_time_chunks[time_chunk] = {col: [] for col in info_df.columns if col not in ['start_time', 'end_time', 'segment_duration']}
                        matches_for_time_chunks[time_chunk]['complete_fit'] = []

                    # Append data to the lists, excluding 'start_time' and 'end_time'
                    for col in [col for col in info_df.columns if col not in ['start_time', 'end_time', 'segment_duration']]:
                        matches_for_time_chunks[time_chunk][col].append(row[col])

                    # Append the fit type
                    matches_for_time_chunks[time_chunk]['complete_fit'].append(complete_fit)
    
    # Iterate through matches and update big_data_frame
    for time_chunk, match_info in matches_for_time_chunks.items():
        for col, data_list in match_info.items():
            # Update big_data_frame with the aggregated list for this column
            big_data_frame.at[(animal, area, day, epoch, time_chunk), col] = data_list

    return big_data_frame


# In[17]:


def wrapper_associate_time_chunks(new_df):
    # Extract unique combinations of animal, day, epoch
    new_df.drop(['level_0', 'index'], axis=1, inplace=True, errors='ignore')
    unique_combinations = new_df.reset_index()[['animal', 'area', 'day', 'epoch']].drop_duplicates().values.tolist()
    
    # Initialize a DataFrame to hold the final result, starting with a copy of the input DataFrame
    new_df = new_df.copy()
    

    # Loop through each unique combination of animal, day, epoch
    for animal, area, day, epoch in unique_combinations:
        try:
            print(f"Processing: Animal = {animal}, , Area = {area}, Day = {day}, Epoch = {epoch}")

            # Fetch position data
            pos_df = pos.get_position_dataframe((animal, day, epoch), animals, use_hmm=False)

            # Get information and labeled segments
            information_df, labeled_segments = pos.get_segments_df((animal, day, epoch), animals, pos_df)

            # Associate time intervals
            segment_info_df = associate_time_intervals(information_df, labeled_segments)
            
            if not isinstance(new_df.index, pd.MultiIndex):
                new_df = new_df.reset_index()
                new_df.set_index(['animal', 'area', 'day', 'epoch', 'time_chunk'], inplace=True)

            # Perform the merge operation
            base_df  = associate_time_chunks_with_info_df(segment_info_df, new_df, animal, area, day, epoch)
            
            
            # If merge is successful, update the base DataFrame
            if new_df is not None:
                new_df = base_df
            else:
                print("Merge returned None. Skipping update for this iteration.")
        except IndexError as ie:
            print(f"Ignoring IndexError: {ie}")
            continue
            
        except Exception as e:
            print(f"An exception occurred: {e}")
            continue

    return new_df


finished_df = wrapper_associate_time_chunks(data)


# In[23]:


import os
import pandas as pd

def store_data_frames(data_frames_list, file_names=['datacomplete2111'], destination_folder='/home/bellijjy', file_type='pickle'):
    """
    Store a list of data frames in a specified folder on your PC as files.
    This function preserves the multi-index of the data frames.

    Arguments:
    data_frames_list -- A list of Pandas data frames to store.
    file_names -- A list of file names.
    destination_folder (optional) -- The destination folder. Default is '/home/bellijjy'.
    file_type (optional) -- The type of the files. Default is 'csv'.

    Returns:
    None
    """
    if len(data_frames_list) != len(file_names):
        print("Error: The number of data frames and file names should be the same.")
        return

    valid_file_types = ['csv', 'json', 'pickle']  # Valid file types for data frames

    if file_type not in valid_file_types:
        print(f"Error: Invalid file type '{file_type}'. Valid file types for data frames are: {', '.join(valid_file_types)}")
        return

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    for df, name in zip(data_frames_list, file_names):
        # Construct the file destination path
        file_destination = os.path.join(destination_folder, f"{name}.{file_type}")

        # Serialize the data frame to the file
        if file_type == 'csv':
            df.to_csv(file_destination)
        elif file_type == 'json':
            df.to_json(file_destination, orient='split')
        elif file_type == 'pickle':
            df.to_pickle(file_destination)

    print("Data frames stored successfully!")


# In[20]:


finished_data_short = finished_df.drop('original_data', axis = 1)


# In[27]:


store_data_frames([finished_data_short])


# In[26]:


finished_data_short


# In[ ]:




