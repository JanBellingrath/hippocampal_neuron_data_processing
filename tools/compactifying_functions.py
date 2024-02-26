

import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import dask
from scipy.io import loadmat
from collections import namedtuple

import mrestimator as mre
import spike_data_to_dataframe_final as sdd
from spike_data_to_dataframe_final import spike_time_index_association
import utilities as u
from lfp_data_to_dataframe_final import make_tetrode_dataframe, get_sampling_rate
import df_subdivision_by_theme as divis
import subdivision_by_behav_state as behav
import sum_and_seperate_spike_trains as sum_sep
#import criticality_analysis as can

import task_new_final as task

import loren_frank_data_processing
import loren_frank_data_processing.core
import loren_frank_data_processing.task
from loren_frank_data_processing.core import logger, get_epochs, get_data_structure, reconstruct_time
from loren_frank_data_processing.tetrodes import convert_tetrode_epoch_to_dataframe, _get_tetrode_id
from loren_frank_data_processing.neurons import get_spikes_dataframe, convert_neuron_epoch_to_dataframe, _add_to_dict, _get_neuron_id


# In[3]:


Animal = namedtuple('Animal', {'short_name', 'directory'})

conley = Animal('/home/bellijjy/Conley.tar/Conley', 'con')
dave = Animal('/home/bellijjy/Dave.tar/Dave/Dave', 'dav')
chapati = Animal('/home/bellijjy/Chapati.tar/Chapati/Chapati', 'cha')
corriander = Animal('/home/bellijjy/Corriander.tar/Corriander', 'Cor')
dudley = Animal('/home/bellijjy/Dudley', 'dud')
bond = Animal('/home/bellijjy/Bond', 'bon')



animals = {'con': Animal('con','/home/bellijjy/Conley.tar/Conley'),
           'Cor': Animal('Cor','/home/bellijjy/Corriander.tar/Corriander'),
            'cha': Animal('cha','/home/bellijjy/Chapati.tar/Chapati/Chapati'),
          'dav': Animal('dav','/home/bellijjy/Dave.tar/Dave/Dave'),
           'dud': Animal('dud','/home/bellijjy/Dudley'),
            'bon' : Animal('bon', '/home/bellijjy/Bond')}


# In[4]:


def neuron_ids_for_specific_animal_and_subarea(area, animal):
    #Information about all recorded neurons such as brain area.
    df = sdd.make_neuron_dataframe_modified(animals)
    
    #subdividing the meta-index by theme
    splitted_df = divis.split_neuron_dataframe_informationally(df, ['area', 'animal'])
    
    #subdividing to acess task/behav. state -- this is joined with the meta-index above via 'get_matching_pairs' below
    
    #HAVE TO ADD OTHER ANIMAL NAMES
    splitted_epoch_dataframe = divis.split_neuron_dataframe_informationally(task.make_epochs_dataframe(animals, corriander, conley, dave, dudley, bond, chapati), ['type'])
    
    area_animal_df = splitted_df[area, animal]
    
    sleep_area_animal_df = divis.get_matching_pairs(area_animal_df, splitted_epoch_dataframe['sleep',].index)
    wake_area_animal_df = divis.get_matching_pairs(area_animal_df, splitted_epoch_dataframe['run',].index)
    
    
    neuron_list_id_sleep = sleep_area_animal_df['neuron_id'].tolist()
    neuron_list_id_wake = wake_area_animal_df['neuron_id'].tolist()
    
    #Getting the epochs per day, relative to brain region (input) and sleep/run (input)
    sleep_epochs_per_day = behav.get_epochs_by_day(sleep_area_animal_df.index)
    wake_epochs_per_day = behav.get_epochs_by_day(wake_area_animal_df.index)

    #generating a dict, relative to brain region and sleep/run, of the neuron_ids per epoch per day
    embedded_day_epoch_dict_sleep = behav.embedded_day_epoch_dict(sleep_epochs_per_day, neuron_list_id_sleep)
    embedded_day_epoch_dict_wake = behav.embedded_day_epoch_dict(wake_epochs_per_day, neuron_list_id_wake)

    #checking for overlap of the epoch/day combinations for sleep/run and generating the final conjoinded dict of run/sleep, day, epoch, neuron_id
    conjoined_key_epochs_per_day_dict = behav.conjoined_dict_with_overlap_checked(embedded_day_epoch_dict_wake, embedded_day_epoch_dict_sleep)
    
    return conjoined_key_epochs_per_day_dict


# In[6]:


def get_spike_data(conjoined_key_epochs_per_day_dict, time_len_splitted_chunks, area, animal):
    '''
    Input: conjoined_key_epochs_per_day_dict: neuron ids relative to day, epoch and behav state
            time_len_splitted_chunks: desired length of temporal intervals for which the parameters get evaluated
    Returns: embedded dict (day/epoch/time_chunk) with the neuronal data as values and time as index
    '''
    
    #redefining the animal dict with the short_name appended at the end
    animals = {'con': Animal('con','/home/bellijjy/Conley.tar/Conley/con'),
           'Cor': Animal('Cor','/home/bellijjy/Corriander.tar/Corriander/Cor'),
            'cha': Animal('cha','/home/bellijjy/Chapati.tar/Chapati/Chapati/cha'),
          'dav': Animal('dav','/home/bellijjy/Dave.tar/Dave/Dave/dav'),
           'dud': Animal('dud','/home/bellijjy/Dudley/dud'),
            'bon' : Animal('bon', '/home/bellijjy/Bond/bon')}
    
    #getting the spike data itself
    spike_dict = behav.coarse_grained_spike_generator_dict(conjoined_key_epochs_per_day_dict, animals)
    
    neuron_ids = neuron_ids_for_specific_animal_and_subarea(area, animal)
     
    #getting the time index of the neuronal data
    time_dict = sdd.time_index_dict(neuron_ids, animals)  
    
    #summing over the neurons in each epoch and associating the spike trains with time index
    spike_dict_summed = sum_sep.sum_time_series_with_time_index_from_embedded_state_day_epoch_key_values_dict(spike_dict, time_dict)
    print(spike_dict_summed)
    #subdividing time index into intervals of 5 seconds
    splitted_by_sec_spike_dict = sum_sep.time_index_seperator(spike_dict_summed, time_len_splitted_chunks)
    
    return splitted_by_sec_spike_dict


def get_spikes(conjoined_key_epochs_per_day_dict, area, animal):
    '''
    Function from above without time chunks.
    Input: conjoined_key_epochs_per_day_dict: neuron ids relative to day, epoch and behav state
            
    Returns: embedded dict (day/epoch/time_chunk) with the neuronal data as values and time as index
    '''
    
    #redefining the animal dict with the short_name appended at the end
    animals = {'con': Animal('con','/home/bellijjy/Conley.tar/Conley/con'),
           'Cor': Animal('Cor','/home/bellijjy/Corriander.tar/Corriander/Cor'),
            'cha': Animal('cha','/home/bellijjy/Chapati.tar/Chapati/Chapati/cha'),
          'dav': Animal('dav','/home/bellijjy/Dave.tar/Dave/Dave/dav'),
           'dud': Animal('dud','/home/bellijjy/Dudley/dud'),
            'bon' : Animal('bon', '/home/bellijjy/Bond/bon')}
    
    #getting the spike data itself
    spike_dict = behav.coarse_grained_spike_generator_dict(conjoined_key_epochs_per_day_dict, animals)
    
    neuron_ids = neuron_ids_for_specific_animal_and_subarea(area, animal)
     
    #getting the time index of the neuronal data
    time_dict = sdd.time_index_dict(neuron_ids, animals)  
    
    #summing over the neurons in each epoch and associating the spike trains with time index
    spike_dict_summed = sum_sep.sum_time_series_with_time_index_from_embedded_state_day_epoch_key_values_dict(spike_dict, time_dict)
    
    return spike_dict_summed

# In[ ]:


