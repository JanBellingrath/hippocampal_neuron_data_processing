#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import utilities as u
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import utilities as u
import glob
import json
import pingouin as pg

def load_criticality_data(directory, animal=None, area=None, state=None, day=None, epoch=None, time_chunk=None):
    # Create the file pattern based on the provided parameters
    pattern = f'{directory}/{animal or "*"}_{area or "*"}_{state or "*"}_{day or "*"}_{epoch or "*"}_{time_chunk or "*"}.parquet'
    
    # Get a list of files that match the pattern
    files = glob.glob(pattern)
    
    # Read and concatenate all files into a single DataFrame
    df = pd.concat([pd.read_parquet(file) for file in files])
    # Optional: If you have serialized complex objects like 'pcov', you can deserialize them here
    # For example:
    df['pcov'] = df['pcov'].apply(json.loads)

    return df

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


def auto_plot(dataframe, plotting_functions):
    """
    Automatically plot data for each unique combination of animal and area.

    Parameters:
    - dataframe (DataFrame): The data containing information for various animals.
    - plotting functions (list): A list of plotting functions to be called for each combination.
    """
    # Fetching unique animals
    unique_animals = dataframe.index.unique()
    
    for animal in unique_animals:
        # Filter data for the current animal
        animal_data = dataframe.loc[animal]
        
        # Fetching unique areas for the current animal
        unique_areas = animal_data['area'].unique()

        for area in unique_areas:
            # Further filter data for the specific area of the current animal
            filtered_data = animal_data[animal_data['area'] == area]
            sorted_area_df = filtered_data.sort_values(by=['day', 'epoch', 'time_chunk'])
            for func in plotting_functions:
                func(sorted_area_df, animal=animal)

def load_all_criticality_data_no_duplicate_files(directory, animals, area=None, state=None, day=None, epoch=None, time_chunk=None):
    all_dfs = []
    loaded_files = set()  # Keep track of files we've already imported

    for animal in animals:
        # Create the file pattern based on the provided parameters
        pattern = f'{directory}/{animal}*_{area or "*"}_{state or "*"}_{day or "*"}_{epoch or "*"}_{time_chunk or "*"}.parquet'

        # Get a list of files that match the pattern
        files = glob.glob(pattern)

        # Read and concatenate all files into a single DataFrame for the current animal
        dfs = []
        for file in files:
            if file not in loaded_files:
                df = pd.read_parquet(file)
                
                # Optional: If you have serialized complex objects like 'pcov', you can deserialize them here
                # For example:
                df['pcov'] = df['pcov'].apply(json.loads)
                
                df['animal'] = animal
                dfs.append(df)
                loaded_files.add(file)  # Mark this file as loaded

        animal_df = pd.concat(dfs)
        all_dfs.append(animal_df)

    # Combine the DataFrames for all animals
    combined_df = pd.concat(all_dfs)

    # Set the index to the 'animal' column
    combined_df.set_index('animal', inplace=True)

    return combined_df

def convert_to_binary_and_save_space(df):
    # Create a new DataFrame to store the results
    new_df = df.copy()
    
    # Initialize an empty list to store the transformed Series
    transformed_data = []
    
    # Initialize an empty list to store the new dt values
    new_dt_values = []
    
    for index, row in df.iterrows():
        # Parse the JSON string
        original_data = json.loads(row['original_data'])
        
        # Sort the keys (time)
        sorted_keys = sorted(map(int, original_data.keys()))
        
        # Create a list that will store the binary sequence
        binary_sequence = [1 if original_data[str(key)] != 0 else 0 for key in sorted_keys]
        
        # Calculate the new dt for this series
        new_dt = 30000 / len(binary_sequence)  # 30 seconds = 30000 ms
        
        # Append the binary sequence and new dt to the lists
        transformed_data.append(binary_sequence)
        new_dt_values.append(new_dt)
    
    # Drop the specified columns along with the original_data column
    columns_to_drop = ['original_data', 'popt', 'ssres', 'pcov', 'steps', 'dt', 'dtunit', 
                       'quantiles', 'mrequantiles', 'tauquantiles', 'description']
    new_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    # Add the transformed data and new dt values to the new DataFrame
    new_df['data'] = transformed_data
    new_df['dt'] = new_dt_values  # in ms
    
    return new_df

def save_df_as_parquet(df, file_name):
    """
    Save a DataFrame as a Parquet file in the user's home directory.
    
    Parameters:
        df (DataFrame): The DataFrame to save.
        file_name (str): The name of the file where the DataFrame will be saved.
        
    Returns:
        None
    """
    try:
        # Determine the home directory
        home_directory = os.path.expanduser("~")
        
        # Create the full file path
        full_file_path = os.path.join(home_directory, file_name)
        
        df.to_parquet(full_file_path)
        print(f"DataFrame successfully saved at {full_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")



import os
import pandas as pd

def load_df_from_parquet(file_name):
    """
    Load a DataFrame from a Parquet file located in the user's home directory.
    
    Parameters:
        file_name (str): The name of the Parquet file to load.
        
    Returns:
        DataFrame: The loaded DataFrame.
    """
    try:
        # Determine the home directory
        home_directory = os.path.expanduser("~")
        
        # Create the full file path
        full_file_path = os.path.join(home_directory, file_name)
        
        # Load the DataFrame
        df = pd.read_parquet(full_file_path)
        print(f"DataFrame successfully loaded from {full_file_path}")
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

