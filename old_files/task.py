#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
from scipy.io import loadmat
from glob import glob
from collections import namedtuple
from os.path import join


Animal = namedtuple('Animal', {'short_name', 'directory'})

conley = Animal('/home/bellijjy/Conley.tar/Conley', 'con')
dave = Animal('/home/bellijjy/Dave.tar/Dave/Dave', 'dav')
chapati = Animal('/home/bellijjy/Chapati.tar/Chapati/Chapati', 'cha')
#frank = Animal('/home/bellijjy/Frank.tar/Frank', 'fra')
#egypt = Animal('/home/bellijjy/Egypt.tar/Egypt', 'egy')
#remi = Animal('/home/bellijjy/Remi.tar/Remi', 'rem')
corriander = Animal('/home/bellijjy/Corriander.tar/Corriander', 'Cor')
dudley = Animal('/home/bellijjy/Dudley', 'dud')
bond = Animal('/home/bellijjy/Bond', 'bon')
#government = Animal('/home/bellijjy/Government.tar/Government', 'gov')


animals = {'con': Animal('con','/home/bellijjy/Conley.tar/Conley/con'),
           'Cor': Animal('Cor','/home/bellijjy/Corriander.tar/Corriander/Cor'),
            #'fra': Animal('fra','/home/bellijjy/Frank.tar/Frank/fra'),
            'cha': Animal('cha','/home/bellijjy/Chapati.tar/Chapati/Chapati/cha'),
          #'egy': Animal('egy','/home/bellijjy/Egypt.tar/Egypt/egy'),
          #'rem': Animal('rem','/home/bellijjy/Remi.tar/Remi/rem'),
          'dav': Animal('dav','/home/bellijjy/Dave.tar/Dave/Dave/dav'),
           'dud': Animal('dud','/home/bellijjy/Dudley'),
        # 'gov' : Animal('gov','/home/bellijjy/Government.tar/Government/gov'),
        'bon' : Animal('bon', '/home/bellijjy/Bond/bon')}
          #} 

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
    #elif animal == egypt:
     #   task_files = glob(join(animal.short_name, 'egytask*.mat'))
    #elif animal == frank:
     #   task_files = glob(join(animal.short_name, 'fratask*.mat'))
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

