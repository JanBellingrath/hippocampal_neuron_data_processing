#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.io import loadmat
from glob import glob
from os.path import join
from collections import namedtuple

Animal = namedtuple('Animal', {'short_name', 'directory'})

conley = Animal('/home/bellijjy/Conley.tar/Conley', 'con')
dave = Animal('/home/bellijjy/Dave.tar/Dave/Dave', 'dav')
chapati = Animal('/home/bellijjy/Chapati.tar/Chapati/Chapati', 'cha')
corriander = Animal('/home/bellijjy/Corriander.tar/Corriander', 'Cor')
dudley = Animal('/home/bellijjy/Dudley', 'dud')
bond = Animal('/home/bellijjy/Bond', 'bon')
frank = Animal('/local2/Jan/Frank/Frank', 'fra')
government = Animal('/local2/Jan/Government/Government/gov', 'fra')
egypt = Animal('/local2/Jan/Egypt/Egypt/egy', 'egy')
remy = Animal('local2/Jan/Remy/Remy/rem', 'rem')


animals = {'con': Animal('con','/home/bellijjy/Conley.tar/Conley'),
           'Cor': Animal('Cor','/home/bellijjy/Corriander.tar/Corriander'),
            'cha': Animal('cha','/home/bellijjy/Chapati.tar/Chapati/Chapati'),
          'dav': Animal('dav','/home/bellijjy/Dave.tar/Dave/Dave'),
           'dud': Animal('dud','/home/bellijjy/Dudley'),
            'bon' : Animal('bon', '/home/bellijjy/Bond'),
              'fra' : Animal('fra', '/local2/Jan/Frank/Frank'),
              'gov' : Animal('gov', 'local2/Jan/Government/Government'),
            'egy' : Animal('egy', 'local2/Jan/Egypt/Egypt'), 
          'rem': Animal('rem', 'local2/Jan/Remy/Remy')}

def get_task(animal):
    '''Loads all experimental information for all days for a given animal.

    Parameters
    ----------
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    task_information : pandas.DataFrame

    '''
    if animal == corriander:
        task_files = glob(join(animal.short_name, 'Cortask*.mat'))
    elif animal == conley:
        task_files = glob(join(animal.short_name, 'contask*.mat'))
    elif animal == dave:
        task_files = glob(join(animal.short_name, 'davtask*.mat'))
    elif animal == dudley:
        task_files = glob(join(animal.short_name, 'dudtask*.mat'))
    elif animal == egypt:
        task_files = glob(join(animal.directory, 'egytask*.mat'))
    elif animal == government:
        task_files = glob(join(animal.directory, 'govtask*.mat'))
    elif animal == remy:
        task_files = glob(join(animal.directory, 'remtask*.mat'))
    elif animal == frank:
        task_files = glob(join(animal.directory, 'fratask*.mat'))
    elif animal == chapati:
        task_files = glob(join(animal.short_name, 'chatask*.mat'))
    elif animal == bond:
        task_files = glob(join(animal.short_name, 'bontask*.mat'))
    
    return pd.concat(load_task(task_file, animal)
                     for task_file in task_files)


def load_task(file_name, animal):
    '''Loads task information for a specific day and converts it to a pandas
    DataFrame.

    Parameters
    ----------
    file_name : str
        Task file name for an animal and recording session day.
    animal : namedtuple
        Information about data directory for the animal.

    Returns
    -------
    task_information : pandas.DataFrame

    '''
    data = loadmat(file_name, variable_names=('task'))['task']
    day = data.shape[-1]
    epochs = data[0, -1][0]
    n_epochs = len(epochs)
    index = pd.MultiIndex.from_product(
        ([animal.short_name], [day], np.arange(n_epochs) + 1),
        names=['animal', 'day', 'epoch'])
    
    return pd.DataFrame(
        [{name: epoch[name].item().squeeze()
          for name in epoch.dtype.names
          if name in ['environment', 'type']}
         for epoch in epochs]).set_index(index).assign(
        environment=lambda df: df.environment.astype(str),
        type=lambda df: df.type.astype(str))

    

def compute_exposure(epoch_info):
    df = epoch_info.groupby(['environment']).apply(
        _count_exposure)
    df['exposure'] = df.exposure.where(
        ~epoch_info.type.isin(['sleep', 'rest', 'nan', 'failed sleep']))
    return df

def _count_exposure(df):
    df['exposure'] = np.arange(len(df)) + 1
    return df


def make_epochs_dataframe(animals, corriander, conley, dave, dudley, bond, chapati):
    '''Experimental conditions for all recording epochs.

    Index is a unique identifying key for that recording epoch.

    Parameters
    ----------
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.
    corriander : str
        Full name of the corriander animal.
    conley : str
        Full name of the conley animal.

    Returns
    -------
    epoch_information : pandas.DataFrame

    '''
    animal_full_names = [corriander, conley, dave, dudley, bond, chapati]
    return compute_exposure(
        pd.concat([get_task(animal_name) for animal_name in animal_full_names])
        .sort_index())


# In[4]:


animal_full_names = [corriander, conley, dave, dudley, bond, chapati]
#pd.concat(
task = [get_task(animal_name) for animal_name in animal_full_names]
#get_task(conley)
#conley.directory
#make_epochs_dataframe(animals, corriander, conley, dave, dudley, bond, chapati)
task[1]


# In[6]:


import os
import pandas as pd

def store_data_frames(data_frames_list, file_names=['cor_new', 'con_new', 'dav_new', 'dud_new', 'bon_new', 'cha_new'], destination_folder='/home/bellijjy', file_type='csv'):
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

# Example usage
store_data_frames(task)


# In[ ]:




