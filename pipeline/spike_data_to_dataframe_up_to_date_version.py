#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from scipy.io import loadmat
from os.path import join 

import loren_frank_data_processing.neurons as lf_neurons 

try:
    from pipeline.lfp_data_to_dataframe_up_to_date_version import get_trial_time
except:
    from lfp_data_to_dataframe_up_to_date_version import get_trial_time



def make_neuron_dataframe_modified(animals_dict):
    '''Information about all recorded neurons such as brain area.
    The index of the dataframe corresponds to the unique key for that neuron
    and can be used to load spiking information.
    Parameters
    ----------
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.
    Returns
    -------
    neuron_information : pandas.DataFrame
    '''

    neuron_file_names = [(lf_neurons.get_neuron_info_path(animals_dict[animal]), animal) for animal in animals_dict]
    #print("neuron_file_names:", neuron_file_names)
    
    neuron_data = [(loadmat(file_name[0]), animal_name) for animal_name, file_name in zip(animals_dict.keys(), neuron_file_names)]

    return pd.concat([
        lf_neurons.convert_neuron_epoch_to_dataframe(
            epoch, animal, day_ind + 1, epoch_ind + 1)
        for cellfile, animal in neuron_data
        for day_ind, day in enumerate(cellfile['cellinfo'].T)
        for epoch_ind, epoch in enumerate(day[0].T)
    ]).sort_index()


def spike_time_index_association(neuron_key, animals, time_function=get_trial_time):
    ''' Calls get_trial_time for reference of dataframe size.
    Fits recorded data of neuron into the time bins.
    Parameters
    --------
    neuron_key : tuple
        key for specific neuron
    animals : dict
        file paths to all animal directories
    time_function : function
        optional
        by default get_trial_time
        determine size of dataframe (total recording time, recording frequency, ...)
    returns
    ---------
    spikes_df : dataframe
        number of spikes summed up for each time bin (-> activity)
    '''
    time = time_function(neuron_key[:3])
    spikes_df = get_spikes_series(neuron_key, animals)
    
    time_index = None
    
    try:
        time_index = np.digitize(spikes_df.index.total_seconds(),
                             time.total_seconds())
 
        time_index[time_index >= len(time)] = len(time) -1
        return (spikes_df.groupby(time[time_index]).sum().reindex(index=time, fill_value=0))
    
    except AttributeError: 
        print('No spikes here; data is emtpy')
        return None
    


def generate_spike_indicator_dict(neuron_key_list, animals):
    '''Creates dictionary of spike_time arrays for each neuron in neuron_key_list
    (if possible).
    Parameters
    ----------
    neuron_key_list : list
        contains unique keys (str) to identify single neurons
    animals : dict
        contains file paths to all animals based on their short names
    Returns
    --------
    spike_indicator_dict : dict
        dictionary of spiking time arrays for all neurons on neuron_key_list
        The key is the neuron_key
    '''
    spike_indicator_dict = {}
    for neuron_key_str in neuron_key_list:
        animal_short_name, day_number, epoch_number, tetrode_number, neuron_number = neuron_key_str.split("_")
        neuron_key = (animal_short_name, int(day_number), int(epoch_number), int(tetrode_number), int(neuron_number))
        #print("Neuron key:", neuron_key)
        try:
            spike_time_array = spike_time_index_association(neuron_key, animals).values.astype(np.int32)
            
        except AttributeError:
            spike_time_array = None
            print(f"No spike indicator data for neuron: {neuron_key}")
            
        spike_indicator_dict[neuron_key] = spike_time_array
    return spike_indicator_dict

def get_data_filename(animal, day, file_type):
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
    filename = '{animal_directory}/{animal_short_name}{file_type}{day:02d}.mat'.format(
        animal_directory=animal.directory,
        animal_short_name = animal.short_name,
        file_type = file_type,
        day=day)

    return filename
    
def get_spikes_series(neuron_key, animals):
    '''Spike times for a particular neuron.

    Parameters
    ----------
    neuron_key : tuple
        Unique key identifying that neuron. Elements of the tuple are
        (animal_short_name, day, epoch, tetrode_number, neuron_number).
        Key can be retrieved from `make_neuron_dataframe` function.
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    spikes_dataframe : pandas.DataFrame
        np.ones array, indexed with spiking times
    '''
    animal, day, epoch, tetrode_number, neuron_number = neuron_key
    filename = get_data_filename(animals[animal], day, 'spikes')
    neuron_file = []
    spike_time = []

    try:
        neuron_file = loadmat(filename)
        spike_time = neuron_file['spikes'][0, -1][0, epoch - 1][0, tetrode_number - 1][0, neuron_number - 1][0]['data'][0][:, 0]
        spike_time = pd.TimedeltaIndex(spike_time, unit='s', name='time')
    
    except (FileNotFoundError, TypeError, IndexError):
        spike_time = []
        print('Failed to load file: {0}'.format(filename))

    '''
    print(pd.Series(np.ones_like(spike_time, dtype=int),
                    index=spike_time, name='{0}_{1:02d}_{2:02}_{3:03}_{4:03}'
                    .format(*neuron_key)))
    '''
    return pd.Series(
        np.ones_like(spike_time, dtype=int), index=spike_time,
        name='{0}_{1:02d}_{2:02}_{3:03}_{4:03}'.format(*neuron_key))








def time_index_dict(state_day_epoch_neuron_key_dict, animals):
    '''
    create a  nested time_dict  which contains the structure (df.index)
    for each spiking dataframe for each state, day and epoch
    Parameters
    -------
    state_day_epoch_neuron_key_dict: embedded state_day_epoch_neuron_key dict
    returns
    --------
    spike indicator dict of time index of spike trains per day epoch combination
    '''
    time_dict = {}
    for state in state_day_epoch_neuron_key_dict:
        time_dict[state] = {}
        for day in state_day_epoch_neuron_key_dict[state]:
            time_dict[state][day] = {}
            for epoch in state_day_epoch_neuron_key_dict[state][day]:
                time_dict[state][day][epoch] = {}
                for neuron_key_str in state_day_epoch_neuron_key_dict[state][day][epoch]:
                    
                    animal_short_name, day_number, epoch_number, tetrode_number, neuron_number = neuron_key_str.split("_")
                    neuron_key = (animal_short_name, int(day_number), int(epoch_number), int(tetrode_number), int(neuron_number))
                    try:
                        time_dict[state][day][epoch][neuron_key] = spike_time_index_association(neuron_key, animals).index
                    except:
                        continue
    return time_dict
    
    
#The problem with the function below is that it takes too long too evaluate, because the time index is taken
#from every neuron individually. Is no problem though, cause we just need to determine the time-array-resolution from them
#and having the resolution we can generate the index ourselves (and put it onto the whole epoch, as oppposed to every individual spike train)
#(actally, after checking, the time index is the same for all neuron in an epoch, which solves the problem.)I let this function here in case I need it sometime
def time_index_and_coarse_grained_spike_generator_dict(state_day_epoch_neuron_key_dict, animals):
    '''Input: state_day_epoch_neuron_key_dict: embedded state_day_epoch_neuron_key dict
        Returns: spike indicator dict of spike trains per day epoch combination
    '''
    output_dict = {}
    for state in state_day_epoch_neuron_key_dict:
        output_dict[state] = {}
        for day in state_day_epoch_neuron_key_dict[state]:
            output_dict[state][day] = {}
            for epoch in state_day_epoch_neuron_key_dict[state][day]:
                output_dict[state][day][epoch] = {}
                for neuron_key_str in state_day_epoch_neuron_key_dict[state][day][epoch]:
                    
                    animal_short_name, day_number, epoch_number, tetrode_number, neuron_number = neuron_key_str.split("_")
                    neuron_key = (animal_short_name, int(day_number), int(epoch_number), int(tetrode_number), int(neuron_number))
                    try:
                        spike_array_values = spike_time_index_association(neuron_key, animals).values.astype(np.int32)

                        spike_array_index = spike_time_index_association(neuron_key, animals).index
                    except AttributeError:
                        spike_array_values = None
                        spike_array_index = None
                        print(f"No spike indicator data for neuron: {neuron_key}")
                    print(spike_array_values)
                    if spike_array_values is not None and spike_array_index is not None:
                        spike_train_dict = dict(zip(spike_array_index, spike_array_values))
                        output_dict[state][day][epoch][neuron_key_str] = spike_train_dict
                        
    return state_day_epoch_neuron_key_dict



# In[ ]:


