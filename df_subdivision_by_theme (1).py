#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


def split_neuron_dataframe_informationally(df, split_cols):
    '''Splits a DataFrame into multiple DataFrames based on specified
    column values.
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be split.
    split_cols : list of str
        The names of the columns to use for splitting the DataFrame.
    Returns
    -------
    dfs : dict
        A dictionary containing the split DataFrames. The keys are tuples
        containing the unique combinations of the specified column values.
    '''
    dfs = {}
    for idx, group in df.groupby(split_cols):
        dfs[idx] = group.copy()
    return dfs


def get_matching_pairs(dataframe, multiindex):
    valid_entries = [row_index[2:] for row_index in multiindex]
    
    df_entries = [(row_index[2],row_index[4]) for row_index in dataframe.index]
    
    matching_entries = [entry for entry in valid_entries if entry in df_entries]
    filtered_dataframe = pd.DataFrame()
    
    for matching_entry in matching_entries:
        temp_dataframe = dataframe.loc[(dataframe.index.get_level_values('day') == matching_entry[0]) & (dataframe.index.get_level_values('epoch') == matching_entry[1])]
        filtered_dataframe = pd.concat([filtered_dataframe, temp_dataframe])

    return filtered_dataframe


# In[ ]:




