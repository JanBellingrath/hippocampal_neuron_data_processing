#!/usr/bin/env python
# coding: utf-8

# In[1]:


import data_analysis_utilitites as da_utilities
import data_analysis_sleep_wake as da_wasl


# In[62]:


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


# In[5]:


import pandas as pd

def load_data(file_path, file_format='parquet', index_cols=None):
    """
    Loads a DataFrame from a file in various formats, with the option to specify multi-index columns.

    Parameters:
    -----------
    file_path : str
        The path to the file.
    file_format : str
        The format of the file ('parquet', 'csv', 'excel', 'json').
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
            # For JSON, orient='split' can be used if the file was saved with this orientation.
            return pd.read_json(file_path, orient='split' if index_cols else 'records')
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        print(f"Failed to load data from {file_path}. Error: {e}")
        return None

# Example usage for a CSV file with a multi-index
cha = load_data('/home/bellijjy/cha_new.csv', file_format='csv', index_cols=[0, 1, 2])
dav = load_data('/home/bellijjy/dav_new.csv', file_format='csv', index_cols=[0, 1, 2])
dud = load_data('/home/bellijjy/dud_new.csv', file_format='csv', index_cols=[0, 1, 2])
con = load_data('/home/bellijjy/con_new.csv', file_format='csv', index_cols=[0, 1, 2])
cor = load_data('/home/bellijjy/cor_new.csv', file_format='csv', index_cols=[0, 1, 2])


# In[ ]:


import pandas as pd
import pickle

def load_data_pickle(file_path, file_format='parquet', index_cols=None):
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

data_df = load_data_pickle('/home/bellijjy/datacomplete2111.pickle', file_format='pickle')
data_df


# In[70]:


#dav
#cha is always in A expect for 3 epochs, dud expect for 1, dav same
#con and corr switch from day 4 on to have 2/3 B

#we could look at a peak for those outlier days, indicating high generalization at those days, before inconsistent detail discovered
#we could also see whether 

#problem is there is that free variable which we dont know namely the time it takes for general representations to form.. could be super fast, 
#could also take several days though.

data_df


# In[79]:


import pandas as pd

def merge_dataframes(df1, df2, merge_columns):
    """
    Merges two DataFrames based on a multi-index and adds specific columns from the second DataFrame to the first one.

    Parameters:
    -----------
    df1 : pd.DataFrame
        The first DataFrame.
    df2 : pd.DataFrame
        The second DataFrame, already set with a multi-index.
    merge_columns : list
        A list of column names in df2 to be added to df1.

    Returns:
    --------
    pd.DataFrame
        The merged DataFrame.
    """
     # Remove underscore from animal names in df2 if it's part of the multi-index
    if 'animal' in df2.index.names:
        df2.index = df2.index.set_levels(df2.index.levels[0].str.replace('^_', '', regex=True), level='animal')

    display(df2)
    # Set multi-index on df1 if not already set
    multi_index_columns = ['animal', 'area', 'day', 'epoch', 'time_chunk']
    if not all(col in df1.index.names for col in multi_index_columns):
        df1 = df1.set_index(multi_index_columns)

    # Select only the necessary columns from df2
    df2_subset = df2[merge_columns]
    #display(df2_subset)
    ##display(df1)
    # Merge the DataFrames
    merged_df = df1.merge(df2_subset, left_index=True, right_index=True, how='left')

    return merged_df

updated_df = merge_dataframes(complete_data, data_df, ['from_well', 'to_well','task','is_correct','turn','complete_fit'])


# In[80]:


updated_df


# In[94]:


def merge_environment_into_main(main_df, env_dfs, animal_names):
    """
    Merges the main DataFrame with multiple environment DataFrames into a single 'environment' column,
    matching only on 'animal', 'day', and 'epoch'.

    Parameters:
    -----------
    main_df : pd.DataFrame
        The main DataFrame with the primary data.
    env_dfs : list of pd.DataFrame
        List of DataFrames with 'environment' data to merge.
    animal_names : list of str
        List of animal names, ordered to correspond with the DataFrames in env_dfs.

    Returns:
    --------
    pd.DataFrame
        The merged DataFrame with a unified 'environment' column.
    """

    
    # Concatenate all the environment DataFrames into one
    concatenated_env_df = pd.concat([
        df.reset_index().assign(animal=lambda x: name if 'animal' not in x.columns else x['animal'].map(lambda _: name))
        for df, name in zip(env_dfs, animal_names)
    ]).drop_duplicates()
    
    concatenated_env_df.set_index(['animal', 'day', 'epoch'], inplace=True)
    
    if 'animal' in concatenated_env_df.index.names:
        concatenated_env_df.index = concatenated_env_df.index.set_levels(concatenated_env_df.index.levels[0].str.replace('^_', '', regex=True), level='animal')

    # Temporarily reduce main_df to match the concatenated_env_df indices
    reduced_main_df = main_df.reset_index()
    display(reduced_main_df)
    reduced_main_df = reduced_main_df.set_index(['animal', 'day', 'epoch'])

    # Merge with the main DataFrame
    merged_df = reduced_main_df.merge(concatenated_env_df, left_index=True, right_index=True, how='left')
    
    # Restore the original index structure of main_df
    final_merged_df = merged_df.reset_index().set_index(main_df.index.names)

    return final_merged_df




final_df = merge_environment_into_main(updated_df, [dav, dud, con, cor, cha], ['_dav', '_dud', '_con', '_cor', '_cha'] )


# In[95]:


final_df


# In[96]:


def check_environment_nan(df):
    """
    Checks if the 'environment' column in the DataFrame is NaN everywhere.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check.

    Returns:
    --------
    bool
        True if 'environment' is NaN everywhere, False otherwise.
    """
    #if 'environment' not in df.columns:
     #   raise ValueError("'environment' column not found in the DataFrame")

    return df['environment'].isna().all()
# Assuming df is your DataFrame
is_env_nan_everywhere = check_environment_nan(final_df)
print(is_env_nan_everywhere)


# In[54]:


updated_df


# In[98]:


import os
import pandas as pd

def store_data_frames(data_frames_list, file_names=['data_complete_for_model_2111_v2'], destination_folder='/home/bellijjy', file_type='pickle'):
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
store_data_frames([final_df])


# In[56]:


updated_df


# In[ ]:




