#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
from scipy.io import loadmat
from collections import namedtuple

from loren_frank_data_processing.core import reconstruct_time
from utilities import _convert_to_dict


Animal = namedtuple('Animal', {'short_name', 'directory'})


def get_LFP_dataframe(tetrode_key, animals):
    '''Gets the LFP data for a given epoch and tetrode.
    Parameters
    ----------
    _data_to_dataframe.py tuple
        Unique key identifying the tetrode. Elements are
        (animal_short_name, day, epoch, tetrode_number).
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.


    Returns
    -------
    LFP : pandas Series dataframe
        Contains the electric potential with corresponding time index
    '''
    filename = get_LFP_filename_modified(tetrode_key, animals)
    try:
        lfp_file = loadmat(filename)
        lfp_data = lfp_file['eeg'][0, -1][0, -1][0, -1]
        lfp_time = reconstruct_time(
            lfp_data['starttime'][0, 0].item(),
            lfp_data['data'][0, 0].size,
            float(lfp_data['samprate'][0, 0].squeeze()))
        return pd.Series(
            data=lfp_data['data'][0, 0].squeeze().astype(float),
            index=lfp_time,
            name='{0}_{1:02d}_{2:02}_{3:03}'.format(*tetrode_key))
    
    except (FileNotFoundError, TypeError):
        print('Failed to load file: {0}'.format(filename))

    #I could add other try blocks to handle data of different format



def get_LFP_filename_modified(tetrode_key, animals):
    #check parameters: is animals dictionary here even neccessary?

    '''Returns a file name for the tetrode file LFP for an epoch.
    Parameters
    ----------
    tetrode_key : tuple
        Unique key identifying the tetrode. Elements are
        (animal_short_name, day, epoch, tetrode_number).
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.
    Returns
    -------
    filename : str
        File path to tetrode file LFP
    '''
    animal, day, epoch, tetrode_number = tetrode_key
    
    filename = ('eeg{day:02d}-{epoch}-'
                '{tetrode_number:02d}.mat').format(
                    day=day, epoch=epoch,
                    tetrode_number=tetrode_number)
   
    animal_paths = {
        'Cor': '/home/bellijjy/Corriander.tar/Corriander/EEG/Cor',
        'dav': '/home/bellijjy/Dave.tar/Dave/Dave/EEG/dav',
        'dud': '/home/bellijjy/Dudley/EEG/dud',
        'con': '/home/bellijjy/Conley.tar/Conley/EEG/con',
        'cha': '/home/bellijjy/Chapati.tar/Chapati/Chapati/EEG/cha',
        'bon': '/home/bellijjy/Bond/EEG/bon',
        'rem': '/home/bellijjy/Remi.tar/Remi/EEG/rem'
        }

    if animal in animal_paths:
        filename = animal_paths[animal] + filename

    else:
        print("error: no match to tetrode key found")
    
    #the above is not yet finished, I need to insert the values for the other animals - IFF everything works with the target loc
    #if animal == dave:
    #filename = '/home/bellijjy/Dave.tar/Dave/Dave/EEG/dav' + filename wrong here PROBABLY
    return filename



def get_tetrode_info_path(animal):
    '''Returns the Matlab tetrode info file name assuming it is in the
    Raw Data directory.
    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.
        
    Returns
    -------
    filename : str
        The path to the information about the tetrodes for a given animal.'''

    animal_paths = {
        'Cor': '/home/bellijjy/Corriander.tar/Corriander/Cor',
        'dav': '/home/bellijjy/Dave.tar/Dave/Dave/dav',
        'dud': '/home/bellijjy/Dudley/dud',
        'con': '/home/bellijjy/Conley.tar/Conley/con',
        'cha': '/home/bellijjy/Chapati.tar/Chapati/Chapati/cha',
        'bon': '/home/bellijjy/Bond/bon',
        'rem': '/home/bellijjy/Remi.tar/Remi/rem'
        }

    if animal in animal_paths:
        filename = animal_paths[animal] + 'tetinfo.mat'

    return filename
    #the above is not yet finished, I need to insert the values for the other animals - IFF everything works with the target loc
    #if animal == dave:
        #   filename = '/home/bellijjy/Dave.tar/Dave/Dave/EEG/dav' + "tetinfo.mat"


def make_tetrode_dataframe(animals, epoch_key=None):
    """Information about all tetrodes such as recording location.
    Parameters
    ----------
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. 
    epoch_key (optional) : list
        contains information to identify specific epoch for an animal

    Returns
    -------
    tetrode_infomation : pandas.DataFrame
    """
    tetrode_info = []
    if epoch_key:
        animal, day, epoch = epoch_key
        
        file_name = get_tetrode_info_path(animal)
        tet_info = loadmat(file_name, squeeze_me=True)["tetinfo"]
        tetrode_info.append(
            convert_tetrode_epoch_to_dataframe(
                tet_info[day - 1][epoch - 1], epoch_key))
        return pd.concat(tetrode_info, sort=True)


def make_tetrode_dataframe_old(animals, epoch_key=None):
    """Information about all tetrodes such as recording location.
    Parameters
    ----------
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. 
    Returns
    -------
    tetrode_infomation : pandas.DataFrame
    """
    tetrode_info = []
    if epoch_key is not None:
        animal, day, epoch = epoch_key
        
        file_name = get_tetrode_info_path(animal)
        tet_info = loadmat(file_name, squeeze_me=True)["tetinfo"]
        tetrode_info.append(
            convert_tetrode_epoch_to_dataframe(
                tet_info[day - 1][epoch - 1], epoch_key))
        return pd.concat(tetrode_info, sort=True)

    for animal in animals.values():
        
        file_name = get_tetrode_info_path(animal)
        
        tet_info = loadmat(file_name, squeeze_me=True)["tetinfo"]
        try:
            for day_ind, day in enumerate(tet_info):
                try:
                    for epoch_ind, epoch in enumerate(day):
                        epoch_key = (
                            animal.short_name,
                            day_ind + 1,
                            epoch_ind + 1,
                        )  
                        tetrode_info.append(
                            convert_tetrode_epoch_to_dataframe(
                                epoch, epoch_key)
                        )
                except IndexError:
                    pass
                
        except TypeError:
            # Only one day of recording
            try:
                day_ind = 0
                for epoch_ind, epoch in enumerate(tet_info):
                    epoch_key = animal.short_name, day_ind + 1, epoch_ind + 1
                    tetrode_info.append(
                        convert_tetrode_epoch_to_dataframe(epoch, epoch_key))
            except IndexError:
                pass

    return pd.concat(tetrode_info, sort=True)



def get_trial_time(epoch_key, animals):
    """Time in the recording session in terms of the LFP.
    This will return the LFP time of the first tetrode found (according to the
    tetrode info). This is useful when there are slightly different timings
    for the recordings and you need a common time.
    Parameters
    ----------
    epoch_key : tuple
        Unique key identifying a recording epoch with elements
        (animal, day, epoch)
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.
    Returns
    -------
    time : pandas.Index
    """
    tetrode_info = make_tetrode_dataframe(animals, epoch_key=epoch_key)
 
    for tetrode_key in tetrode_info.index:
        lfp_df = get_LFP_dataframe(tetrode_key, animals)
        
        if lfp_df is not None:
            break
    if lfp_df is not None:
        return lfp_df.index.rename("time")


def _get_tetrode_id(dataframe):
    '''Unique string identifier for a tetrode
    constructed from relevant information, such as animal, day, ...'''
    return (
        dataframe.animal.astype(str) + '_' +
        dataframe.day.map('{:02d}'.format).astype(str) + '_' +
        dataframe.epoch.map('{:02d}'.format).astype(str) + '_' +
        dataframe.tetrode_number.map('{:02d}'.format).astype(str)
    )


def convert_tetrode_epoch_to_dataframe(tetrodes_in_epoch, epoch_key):
    '''Convert tetrode information data structure to dataframe.
    Parameters
    ----------
    tetrodes_in_epoch : matlab data structure
    epoch_key : tuple
        Unique key identifying a recording epoch. Elements are
        (animal, day, epoch)
    Returns
    -------
    tetrode_info : dataframe
    '''
    animal, day, epoch = epoch_key
    tetrode_dict_list = [_convert_to_dict(tetrode) for tetrode in tetrodes_in_epoch]
    
    # Create a dictionary to hold the values for the 'animal', 'day', and 'epoch' columns
    assign_dict = {
        'animal': [animal] * len(tetrode_dict_list),
        'day': [day] * len(tetrode_dict_list),
        'epoch': [epoch] * len(tetrode_dict_list)
    }

    return (pd.DataFrame(tetrode_dict_list)
              .assign(**assign_dict)
              .assign(tetrode_number=lambda x: x.index + 1)
              .assign(tetrode_id=_get_tetrode_id)
              .set_index(['animal', 'day', 'epoch', 'tetrode_number'])
              .sort_index()
            )


# In[ ]:

'''
Questions for Jan:
    - make animals a global variable? 
        -> wouldnt need to pass it as a variable every time
    - difference between tetrode_dataframe and LFP_dataframe?
    - in get_LFP_dataframe: why? 
        lfp_data = lfp_file['eeg'][0, -1][0, -1][0, -1] ?
        (always last element?)
    - in make_tetrode_dataframe:
        why make epoch_key optional?
        also: why does epoch_key contain the Animal(named_tuple)? 
            Wouldn´t just the short_name be sufficient?
        in the end, why use the concat() function? Isn´t there only on dataframe in the list?
    - in get_trial_time:
        is this the main function? 
        why does it return after the first tetrode dataset?
        Wouldn´t it be more efficient to construct all LFP-dataframes, since they will
            be used later anyway?
        What about the tetrode_info dataframe? Is it used in other instances, too?
            If not, it would seem simpler to just construct a list containing all possible tetrode names
            I can see make_tetrode_dataframe() being imported to compactifying_functions,
                but it isn´t used
    - also, help for downloading hc-6 dataset would be great!!


        
'''


