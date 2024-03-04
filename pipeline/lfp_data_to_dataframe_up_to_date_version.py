#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#!/usr/bin/env python
# coding: utf-8

# In[14]:

import pandas as pd
import numpy as np
from scipy.io import loadmat
from collections import namedtuple

from loren_frank_data_processing.core import reconstruct_time


# I can´t quite figure out what is going on:
# if lfp_data_to_dataframe is imported to a script in the main directory,
# everything should work fine

# But if lfp_data_to_dataframe is executed directly, it can´t import "pipeline.utilities"
# instead, it will only find "utilities"
# this could be, if wanted, solved with a try-statement

# Again, when lfp_data_to_dataframe is imported to a script in ANOTHER package
# nothing works at all
# in that case, sys.path or PYTHONPATH might need to be edited maually
# However, I don´t think scripts should rely on modules from different packages in this code

# But as long as modules are imported to files in the main branch, everything should work fine
from pipeline.utilities import _convert_to_dict


Animal = namedtuple('Animal', {'short_name', 'directory'})


def get_LFP_dataframe(tetrode_key):
    '''Gets the LFP data for a given epoch and tetrode.
    Parameters
    ----------
    tetrode_key : tuple
        Unique key identifying the tetrode. Elements are
        (animal_short_name, day, epoch, tetrode_number).
    Returns
    -------
    LFP : pandas dataframe
        Contains the electric potential and time
    '''
    filename = get_LFP_filename_modified(tetrode_key)
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
        print(f'Failed to load file: {filename}')
    #I could add other try blocks to handle data of different format

def get_LFP_filename_modified(tetrode_key):

    '''Returns a file name for the tetrode file LFP for an epoch.
    Parameters
    ----------
    tetrode_key : tuple
        Unique key identifying the tetrode. Elements are
        (animal_short_name, day, epoch, tetrode_number).
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
   
    if animal == 'Cor':
        filename = '/home/bellijjy/Corriander.tar/Corriander/EEG/Cor' + filename
    elif animal == 'dav':
        filename = '/home/bellijjy/Dave.tar/Dave/Dave/EEG/dav' + filename
    elif animal == 'dud':
        filename = '/home/bellijjy/Dudley/EEG/dud' + filename
    elif animal == 'con':
        filename = '/home/bellijjy/Conley.tar/Conley/EEG/con' + filename
    elif animal == 'cha':
        filename = '/home/bellijjy/Chapati.tar/Chapati/Chapati/EEG/cha' + filename
    elif animal == 'bon':
        filename = '/home/bellijjy/Bond/EEG/bon' + filename
    elif animal == 'rem':
        filename = '/home/bellijjy/Remi.tar/Remi/EEG/rem' + filename
    elif animal == 'fra':
        filename = '/local2/Jan/Frank/Frank/EEG/fra' + filename
    elif animal == 'gov':
        filename = '/local2/Jan/Government/Government/EEG/gov' + filename
    elif animal == 'egy':
        filename = '/local2/Jan/Egypt/Egypt/EEG/egy' + filename
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
    
    filename = 'tetinfo.mat'
   
    if animal == 'Cor':
        joined = '/home/bellijjy/Corriander.tar/Corriander/Cor' + filename
    elif animal == 'dav':
        joined = '/home/bellijjy/Dave.tar/Dave/Dave/dav' + filename
    elif animal == 'dud':
        joined = '/home/bellijjy/Dudley/dud' + filename
    elif animal == 'con':
        joined = '/home/bellijjy/Conley.tar/Conley/con' + filename
    elif animal == 'cha':
        joined = '/home/bellijjy/Chapati.tar/Chapati/Chapati/cha' + filename
    elif animal == 'bon':
        joined = '/home/bellijjy/Bond/bon' + filename
    elif animal == 'rem':
        joined = '/home/bellijjy/Remi.tar/Remi/rem' + filename
    elif animal == 'fra':
        joined = '/local2/Jan/Frank/Frank/fra' + filename
    elif animal == 'gov':
        joined = '/local2/Jan/Government/Government/gov' + filename
    elif animal == 'egy':
        joined = '/local2/Jan/Egypt/Egypt/egy' + filename
    return joined
#the above is not yet finished, I need to insert the values for the other animals - IFF everything works with the target loc

def make_tetrode_dataframe(epoch_key=None):
    """Information about all tetrodes such as recording location.
    Parameters
    ----------
   epoch_key : tuple
        Unique key identifying an epoch. Elements are
        (animal_short_name, day, epoch).
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



def get_trial_time(epoch_key):
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
    tetrode_info = make_tetrode_dataframe(epoch_key=epoch_key)
 
    for tetrode_key in tetrode_info.index:
        lfp_df = get_LFP_dataframe(tetrode_key)
        
        if lfp_df is not None:
            return lfp_df.index.rename("time")
        
    return None


def _get_tetrode_id(dataframe):
    '''Unique string identifier for a tetrode
    Parameters
    ---------
    dataframe : pandas dataframe
        datapoints of tetrode dataframe
    Returns
    --------
    tetrode id : string
        unique id identifying tetrode for animal and epoch
     '''
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

    # Convert list of dictionaries to pandas dataframe
    # add columns
    return (pd.DataFrame(tetrode_dict_list)
              .assign(**assign_dict)
              .assign(tetrode_number=lambda x: x.index + 1)
              .assign(tetrode_id=_get_tetrode_id)
              .set_index(['animal', 'day', 'epoch', 'tetrode_number'])
              .sort_index()
            )


# In[ ]:
