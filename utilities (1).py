#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy.io import loadmat

#this function is unused currently
def _convert_to_dict_modified(struct_array):
    try:
        return {name: struct_array[name].item().item()
                for name in struct_array.dtype.names
                if struct_array[name].item().size == 1}
    except TypeError:
        return {}
    #added in the AttributeError
    except AttributeError:
        return {}
    
def _convert_to_dict(struct_array):
    try:
        return {name: struct_array[name].item()
                for name in struct_array.dtype.names}
    except TypeError:
        return {}

    
def dict_to_numpy(my_dict):
    my_array = np.array([v for v in my_dict.values() if v is not None])
    return my_array

def dict_to_list(my_dict):
    my_list = [v for v in my_dict.values() if v is not None]
    return my_list

def pad_arrays(arrays):
    """
    Pads a list of arrays with zeros so that they are all the same length.

    Parameters:
    -----------
    arrays : list of numpy arrays
        The arrays to pad.

    Returns:
    --------
    padded_arrays : numpy array
        The padded arrays, stacked along the first axis.
    """
    # Get the maximum length of all the arrays
    max_len = max(len(arr) for arr in arrays if arr is not None)

    # Pad all the arrays to make them the same length
    padded_arrays = []
    for arr in arrays:
        if arr is not None:
            padded_array = np.pad(arr, (0, max_len - len(arr)), 'constant')
            padded_arrays.append(padded_array)
            print(padded_array)

    # Stack the padded arrays together
    if len(padded_arrays) > 0:
        padded_arrays = np.vstack(padded_arrays)

    return padded_arrays

def chunk_array(arr, n):
    # Determine the size of each subarray
    size = (len(arr) + n - 1) // n

    # Create the 2D list
    result = [[] for _ in range(n)]

    # Iterate over the original array and insert each element into the appropriate subarray
    for i, elem in enumerate(arr):
        subarray_index = i // size
        result[subarray_index].append(elem)

    return result






#below is not yet imported into the module!!

#sums 2DArray of trials over time, over all trials, such that one cummulative time-series results
def sum_time_series_dict(time_series_dict):
    
    num_bins = max([len(x) for x in time_series_dict.values() if x is not None])
    res = np.zeros(num_bins)
    for arr in time_series_dict.values():
        if arr is not None:
            res[:len(arr)] += arr
    return res


def get_data_structure(animal, day, file_type, variable):
    '''Returns data structures corresponding to the animal, day, file_type
    for all epochs

    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.
    day : int
        Day of recording
    file_type : str
        Data structure name (e.g. linpos, dio)
    variable : str
        Variable in data structure

    Returns
    -------
    variable : list, shape (n_epochs,)
        Elements of list are data structures corresponding to variable

    '''
    try:
        file = loadmat(get_data_filename(animal, day, file_type))
        n_epochs = file[variable][0, -1].size
        return [file[variable][0, -1][0, ind]
                for ind in np.arange(n_epochs)]
    except (IOError, TypeError):
        logger.warn('Failed to load file: {0}'.format(
            get_data_filename(animal, day, file_type)))
        return None
    
#ADD TO DICT AND CONVERT TO DICT are not defined/imported for the function below
    
#this function below is also not being called somewhere.. what is going on? (refers to spike_data_to_dataframe)
def convert_neuron_epoch_to_dataframe_modified(tetrodes_in_epoch, animal, day, epoch):
    '''
    Given an neuron data structure, return a cleaned up DataFrame
    '''
    DROP_COLUMNS = ['ripmodtag', 'thetamodtag', 'runripmodtag',
                    'postsleepripmodtag', 'presleepripmodtag',
                    'runthetamodtag', 'ripmodtag2', 'runripmodtag2',
                    'postsleepripmodtag2', 'presleepripmodtag2',
                    'ripmodtype', 'runripmodtype', 'postsleepripmodtype',
                    'presleepripmodtype', 'FStag', 'ripmodtag3',
                    'runripmodtag3', 'ripmodtype3', 'runripmodtype3',
                    'tag', 'typetag', 'runripmodtype2',
                    'tag2', 'ripmodtype2', 'descrip']

    NEURON_INDEX = ['animal', 'day', 'epoch',
                    'tetrode_number', 'neuron_number']

    neuron_dict_list = [_add_to_dict(
        _convert_to_dict(neuron), tetrode_ind, neuron_ind)
        for tetrode_ind, tetrode in enumerate(
        tetrodes_in_epoch[0][0])
        for neuron_ind, neuron in enumerate(tetrode[0])
        #if neuron.size > 0
    ]
    try:
        return (pd.DataFrame(neuron_dict_list)
                  .drop(DROP_COLUMNS, axis=1, errors='ignore')
                  .assign(animal=animal)
                
                  .assign(day=day)
                  .assign(epoch=epoch)
                  .assign(neuron_id=_get_neuron_id
                         )
                # set index to identify rows
                  .set_index(NEURON_INDEX)
                  .sort_index())
    except AttributeError:
        logger.debug(f'Neuron info {animal}, {day}, {epoch} not processed')


# In[ ]:
def time_array_from_index_of_series(series):
    time_array = [i for i in speed_array.index.values]
    return time_array




