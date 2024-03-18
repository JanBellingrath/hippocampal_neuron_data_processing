#!/usr/bin/env python
# coding: utf-8

# In[37]:


from collections import namedtuple
import pandas as pd
from scipy.io import loadmat
import numpy as np
import pandas as pd
from os.path import join


# In[41]:


Animal = namedtuple('Animal', {'short_name', 'directory'})

conley = Animal('/home/bellijjy/Conley.tar/Conley', 'con')
dave = Animal('/home/bellijjy/Dave.tar/Dave/Dave', 'dav')
chapati = Animal('/home/bellijjy/Chapati.tar/Chapati/Chapati', 'cha')
corriander = Animal('/home/bellijjy/Corriander.tar/Corriander', 'Cor')
dudley = Animal('/home/bellijjy/Dudley/dud', 'dud')
bond = Animal('/home/bellijjy/Bond/bon', 'bon')
frank = Animal('/local2/Jan/Frank/Frank', 'fra')
government = Animal('/local2/Jan/Government/Government/gov', 'gov')
egypt = Animal('/local2/Jan/Egypt/Egypt/egy', 'egy')
remy = Animal('/local2/Jan/Remy/Remy/remy', 'remy')
five = Animal("/home/dekorvyb/Downloads/Fiv", "Fiv")
bon = Animal("/home/dekorvyb/Downloads/Bon", "bon")


animals = {'con': Animal('con','/home/bellijjy/Conley.tar/Conley/con'),
           'Cor': Animal('Cor','/home/bellijjy/Corriander.tar/Corriander/Cor'),
            'cha': Animal('cha','/home/bellijjy/Chapati.tar/Chapati/Chapati/cha'),
          'dav': Animal('dav','/home/bellijjy/Dave.tar/Dave/Dave/dav'),
           'dud': Animal('dud','/home/bellijjy/Dudley/dud'),
            #bon' : Animal('bon', '/home/bellijjy/Bond/bon'),
              'fra' : Animal('fra', '/local2/Jan/Frank/Frank/fra'),
              'gov' : Animal('gov', '/local2/Jan/Government/Government/gov'),
            'egy' : Animal('egy', '/local2/Jan/Egypt/Egypt/egy'), 
          'remy': Animal('remy', '/local2/Jan/Remy/Remy/rem'),
          "Fiv" : Animal("Fiv", "/home/dekorvyb/Downloads/Fiv"),
          "bon" : Animal("bon", "/home/dekorvyb/Downloads/Bon")}

['_Cor' '_bon' '_cha' '_con' '_dud']


# In[3]:


import loren_frank_data_processing.position as pos
import loren_frank_data_processing.track_segment_classification as tsc
import loren_frank_data_processing.core as core
from logging import getLogger
logger = getLogger(__name__)


# In[70]:


#It is essential to add the animal_short name, e.g. 'con' into the animal dict before running this function.. 
#No wait wtf... sometimes you need to add and sometimes you need to delete... check modules etc.

pos._get_pos_dataframe(('fra',4,5), animals)#['x_position']


# In[ ]:


import networkx as nx

def my_to_scipy_sparse_matrix(G, weight='weight'):
    # Your custom implementation here
    return nx.adjacency_matrix(G, weight=weight)

# Monkey patch the function in the library
nx.to_scipy_sparse_matrix = my_to_scipy_sparse_matrix


#sometimes an error is thrown in which "there is no field "linear_cord" ... this is only because of the specific day, epoch combination
pos.get_position_dataframe(('con',3,2), animals, use_hmm=False)


# In[84]:


EDGE_ORDER = [0, 2, 4, 1, 3]
EDGE_SPACING = [15, 0, 15, 0]


def _get_pos_dataframe(epoch_key, animals):
    try:
        animal, day, epoch = epoch_key

        # Check if the animal exists in the given dictionary
        if animal not in animals:
            print(f"Animal {animal} not found in the animals dictionary.")
            return None

        # Fetch the structure for the given epoch key
        struct = get_data_structure(animals[animal], day, 'pos', 'pos')

        # Check if the structure is None
        if struct is None:
            print(f"No data structure returned for {epoch_key}.")
            return None

        # Check if the epoch index is out of range
        if epoch - 1 >= len(struct):
            print(f"Epoch {epoch} is out of range for the data structure. Max index is {len(struct) - 1}.")
            return None

        # Fetch the specific epoch data
        struct = struct[epoch - 1]

        # Check if 'data' exists and is array-like
        #if 'data' not in struct or not hasattr(struct['data'], 'shape'):
         #   print(f"Data structure for {epoch_key} is not as expected.")
          #  return None

        # Fetch the position data
        position_data = struct['data'][0, 0]

        # Define the field names
        FIELD_NAMES = ['time', 'x_position', 'y_position', 'head_direction',
                       'speed', 'smoothed_x_position', 'smoothed_y_position',
                       'smoothed_head_direction', 'smoothed_speed']

        # Check if there are enough columns
        if position_data.shape[1] < 5:
            print(f"Insufficient columns in position_data for {epoch_key}.")
            return None

        # Create a time index
        time = pd.TimedeltaIndex(position_data[:, 0], unit='s', name='time')

        # Return the appropriate DataFrame based on the number of columns
        if position_data.shape[1] > 5:
            NEW_NAMES = {'smoothed_x_position': 'x_position',
                         'smoothed_y_position': 'y_position',
                         'smoothed_head_direction': 'head_direction',
                         'smoothed_speed': 'speed'}
            return (pd.DataFrame(position_data[:, 5:], columns=FIELD_NAMES[5:], index=time)
                    .rename(columns=NEW_NAMES))
        else:
            return pd.DataFrame(position_data[:, 1:5], columns=FIELD_NAMES[1:5], index=time)

    except Exception as e:
        print(f"Failed to get DataFrame for {epoch_key}. Error: {e}")
        return None    
    

def get_position_dataframe(epoch_key, animals, use_hmm=True,
                           max_distance_from_well=5,
                           route_euclidean_distance_scaling=1,
                           min_distance_traveled=50,
                           sensor_std_dev=5,
                           diagonal_bias=1E-1,
                           edge_spacing=EDGE_SPACING,
                           edge_order=EDGE_ORDER,
                           skip_linearization=False):
    '''Returns a list of position dataframes with a length corresponding
     to the number of epochs in the epoch key -- either a tuple or a
    list of tuples with the format (animal, day, epoch_number)

    Parameters
    ----------
    epoch_key : tuple
        Unique key identifying a recording epoch. Elements are
        (animal, day, epoch)
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    position : pandas dataframe
        Contains information about the animal's position, head direction,
        and speed.

    '''

    # Fetch the position dataframe
    position_df = pos._get_pos_dataframe(epoch_key, animals)

    if position_df is None:
        print(f"No position data returned for {epoch_key}. Skipping...")
        return None

    if not skip_linearization:
        print(f"Applying linearization for {epoch_key}...")
        if use_hmm:
            position_df = pos._get_linear_position_hmm(
                epoch_key, animals, position_df,
                max_distance_from_well, route_euclidean_distance_scaling,
                min_distance_traveled, sensor_std_dev, diagonal_bias,
                edge_order=edge_order, edge_spacing=edge_spacing)
        else:
            linear_position_df = pos._get_linpos_dataframe(
                epoch_key, animals, edge_order=edge_order,
                edge_spacing=edge_spacing)

            if linear_position_df is None:
                print(f"No linear position data returned for {epoch_key}. Skipping...")
                return None

            position_df = position_df.join(linear_position_df)

    return position_df

    #except Exception as e:
     #   print(f"Failed to process position data for {epoch_key}. Error: {e}")
      #  return None

def get_data_filename_2(animal, day, file_type):
    '''Returns the Matlab file name assuming it is in the Raw Data
    directory.

    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.
    day : int
        Day of recording
    file_type : str
        Data structure name (e.g. linpos, dio)

    Returns
    -------
    filename : str
        Path to data file

    '''
    
    filename = '{animal.short_name}/{animal.directory}{file_type}{day:02d}.mat'.format(
        animal=animal,
        file_type=file_type,
        day=day)
    return join(animal.directory, filename)

import logging

# Initialize the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG for detailed output

def get_data_structure(animal, day, file_type, variable):
    logger.debug(f'Starting to process: Animal={animal}, Day={day}, FileType={file_type}, Variable={variable}')
    
    # Initialize file variable to None for scope coverage
    file = None
    
    # Try to load using the first filename function
    try:
        first_method_filename = core.get_data_filename(animal, day, file_type)
        logger.debug(f'Trying to load file using first method: {first_method_filename}')
        file = loadmat(first_method_filename)
        logger.debug('Successfully loaded file using first method.')
    except Exception as e:
        logger.warn(f'Failed to load file using first method: {first_method_filename}. Exception: {e}')
        
        # Fallback to using the second filename function
        try:
            second_method_filename = get_data_filename_2(animal, day, file_type)
            logger.debug(f'Trying to load file using second method: {second_method_filename}')
            file = loadmat(second_method_filename)
            logger.debug('Successfully loaded file using second method.')
        except Exception as e2:
            logger.error(f'Failed to load file using both methods: {e2}')
            return None, 'File loading failed'
    
    # Process the file if it's successfully loaded
    try:
        logger.debug(f'Trying to process variable: {variable}')
        n_epochs = file[variable][0, -1].size
        logger.debug(f'Successfully processed variable. Number of epochs: {n_epochs}')
        return [file[variable][0, -1][0, ind] for ind in np.arange(n_epochs)], 'Success'
    except Exception as e3:
        logger.error(f'Failed to process variable {variable}. Exception: {e3}')
        return None, f'Failed to process variable {variable}'

# Replace the function calls and variable names with actual values
# result, message = get_data_structure(animal, day, file_type, variable)


# In[6]:


def gather_animal_data(animals, max_days=10, max_epochs=10):
    """
    Gathers animal data across multiple days and epochs into a single DataFrame.
    
    Parameters:
    -----------
    animals : dict
        A dictionary containing animal identifiers and related data.
    get_position_dataframe_func : function
        The function used to fetch the position data. It should accept an epoch_key and animals dict.
    max_days : int, optional
        The maximum number of days to consider for each animal. Default is 10.
    max_epochs : int, optional
        The maximum number of epochs to consider for each day for each animal. Default is 10.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing concatenated position data for all animals, days, and epochs.
        The DataFrame will also include 'animal', 'day', and 'epoch' columns.
    """
    
    # Initialize an empty DataFrame to store all data
    all_data = pd.DataFrame()
    
    # Iterate over all animals, days, and epochs to gather data
    for animal in animals.keys():
        for day in range(1, max_days + 1):  # Days from 1 to max_days
            for epoch in range(1, max_epochs + 1):  # Epochs from 1 to max_epochs
                epoch_key = (animal, day, epoch)
                try:
                    # Fetch the data using the provided get_position_dataframe function
                    df = pos.get_position_dataframe(epoch_key, animals, use_hmm=False)
                    #I NEED TO GO IN HERE TO MODIFY THE ACCESSING OF THE OTHER ANIMALS.
                    # Add 'animal', 'day', and 'epoch' columns to identify each row
                    df['animal'] = animal
                    df['day'] = day
                    df['epoch'] = epoch
                    
                    # Append to the main DataFrame
                    all_data = pd.concat([all_data, df.reset_index()], ignore_index=True)
                except Exception as e:
                    # Handle any exceptions that arise
                    print(f"Failed to get data for {epoch_key}. Error: {e}")

    return all_data


# In[85]:


get_position_dataframe(('fra', 4,5), animals, use_hmm=False)



# In[59]:


position_speed_30_01_24 = gather_animal_data(animals, max_days=10, max_epochs=10)


# In[72]:


# Count unique combinations of 'animal', 'day', 'epoch'
unique_combinations = position_speed_30_01_24.groupby(['animal', 'day', 'epoch']).ngroups
unique_combinations


# In[12]:


import pandas as pd
import data_analysis_areas as da_area
import data_analysis_utilitites as da_utilities
import data_analysis_sleep_wake as da_wasl

data = da_utilities.load_all_criticality_data_no_duplicate_files('/home/bellijjy/criticality_january_sm', ['_fra','_dud', '_dav', '_con', '_cha', '_Cor', '_egy', '_gov'], area=None, state=None, day=None, epoch=None, time_chunk=None)
data


# In[69]:


import os

def save_df_as_parquet(df, file_name):
    """
    Save a DataFrame as a Parquet file in the user's home directory.
    
    Parameters:
        df (DataFrame): The DataFrame to save.
        file_name (str): The name of the file where the DataFrame will be saved.
        
    Returns:
        None
        
    """
    
    df['animal'] = df.index.get_level_values('animal')

    try:
        # Determine the home directory
        home_directory = os.path.expanduser("~")
        
        # Create the full file path
        full_file_path = os.path.join(home_directory, file_name)
        
        df.to_parquet(full_file_path, index=False)
        print(f"DataFrame successfully saved at {full_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


save_df_as_parquet(coarse_grained_df_with_speed, 'coarse_grained_df_with_speed.parquet_30_01')


# In[ ]:


import pandas as pd

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
test = load_from_parquet('/home/bellijjy/position_dataframe_complete.parquet')


# In[ ]:


df = load_from_parquet('/home/bellijjy/data_trimmed.parquet')


# In[ ]:


save_df_as_parquet(joined_df, 'completedataframeallrats')


# In[ ]:


import pandas as pd
import numpy as np

#this function is relative to time-chunks which we don't have atm anymore
def join_and_aggregate_dfs(df1, df2):
    """
    This function joins two dataframes based on common columns ['animal', 'day', 'epoch', 'time_chunk']
    and aggregates the first dataframe's columns by the mean (for numerical columns) or the mode (for categorical columns).
    
    Parameters:
    df1 (DataFrame): The first dataframe, with detailed information.
    df2 (DataFrame): The second dataframe, to which columns from df1 will be appended.
    
    Returns:
    DataFrame: A new dataframe that is the result of joining df2 with aggregated columns from df1.
    """
    
    # Remove leading underscores from animal names if present
    df1['animal'] = df1['animal'].apply(remove_leading_underscores)
    df2.index = df2.index.map(remove_leading_underscores)
    
    # Convert time to total_seconds and compute time_chunk relative to each animal-day-epoch
    df1['total_seconds'] = df1['time'].dt.total_seconds()
    df1['relative_seconds'] = df1.groupby(['animal', 'day', 'epoch'])['total_seconds'].transform(lambda x: x - x.min())
    df1['time_chunk'] = df1['relative_seconds'] // 30
    df1['time_chunk'] = df1['time_chunk'].astype(int)
    
    # Aggregate df1 based on time_chunk
    cols_to_append = [col for col in df1.columns if col not in ['animal', 'day', 'epoch', 'time', 'total_seconds', 'relative_seconds', 'time_chunk']]
    averaged = df1.groupby(['animal', 'day', 'epoch', 'time_chunk'])[cols_to_append].apply(
        lambda group: pd.Series({
            col: group[col].mean() if group[col].dtype != 'object' else group[col].value_counts().idxmax()
            for col in cols_to_append
        })
    ).reset_index()
    
    # Merge df2 with the aggregated df1
    joined_df = pd.merge(df2, averaged, how='left', on=['animal', 'day', 'epoch', 'time_chunk'])
    
    return joined_df

# Helper function to remove leading underscores from animal names
def remove_leading_underscores(animal_name):
    return animal_name.lstrip('_')



# Testing the function
joined_df = join_and_aggregate_dfs(test, df)
joined_df


# In[31]:





# In[61]:


import pandas as pd

def append_averages(df_with_speed, df_to_append):
    # Reset index if 'animal', 'day', 'epoch' are in a MultiIndex
    df_with_speed_reset = df_with_speed.reset_index()
    df_with_speed_reset['animal'] = '_' + df_with_speed_reset['animal']

    df_to_append_reset = df_to_append.reset_index()
    
    # Calculate averages for each 'animal', 'day', 'epoch' group
    averages = df_with_speed_reset.groupby(['animal', 'day', 'epoch']).agg({
        'speed': 'mean',
        'linear_distance': 'mean',
        'linear_speed': 'mean',
        'linear_velocity': 'mean'
    }).reset_index()
    print(averages['animal'].unique())
    # Merge averages with the second dataframe
    merged_df = pd.merge(df_to_append_reset, averages, on=['animal', 'day', 'epoch'], how='left')

    # Set 'animal' as part of a MultiIndex again
    merged_df.set_index(['animal'], inplace=True)

    return merged_df

# Assuming df_with_speed and df_to_append are your dataframes
result_df = append_averages(position_speed_30_01_24, data)


# In[63]:


coarse_grained_df_with_speed = result_df


# In[57]:


num_non_nan = result_df['speed'].notna().sum()


# In[58]:


num_non_nan


# In[64]:


coarse_grained_df_with_speed


# In[ ]:




