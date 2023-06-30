#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import dask
from scipy.io import loadmat
from collections import namedtuple

import mrestimator as mre
import spike_data_to_dataframe_final as sdd
import utilities as u
from lfp_data_to_dataframe_final import make_tetrode_dataframe, get_trial_time
import df_subdivision_by_theme as divis
import subdivision_by_behav_state as behav

import task_new_final as task

import loren_frank_data_processing
import loren_frank_data_processing.core
import loren_frank_data_processing.task
from loren_frank_data_processing.core import logger, get_epochs, get_data_structure, reconstruct_time
from loren_frank_data_processing.tetrodes import convert_tetrode_epoch_to_dataframe, _get_tetrode_id
from loren_frank_data_processing.neurons import get_spikes_dataframe, convert_neuron_epoch_to_dataframe, _add_to_dict, _get_neuron_id
#don't know if all functions are imported automatically, please find out


# In[197]:


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


animals = {'con': Animal('con','/home/bellijjy/Conley.tar/Conley'),
           'Cor': Animal('Cor','/home/bellijjy/Corriander.tar/Corriander/Cor'),
            #'fra': Animal('fra','/home/bellijjy/Frank.tar/Frank/fra'),
            'cha': Animal('cha','/home/bellijjy/Chapati.tar/Chapati/Chapati'),
          #'egy': Animal('egy','/home/bellijjy/Egypt.tar/Egypt/egy'),
          #'rem': Animal('rem','/home/bellijjy/Remi.tar/Remi/rem'),
          'dav': Animal('dav','/home/bellijjy/Dave.tar/Dave/Dave'),
           'dud': Animal('dud','/home/bellijjy/Dudley'),
        # 'gov' : Animal('gov','/home/bellijjy/Government.tar/Government/gov'),
        'bon' : Animal('bon', '/home/bellijjy/Bond')}
          #} 


# In[195]:


#Information about all recorded neurons such as brain area.
#The index of the dataframe corresponds to the unique key for that neuron and can be used to load spiking information.
df = sdd.make_neuron_dataframe_modified(animals)

#subdividing the meta-index by theme
splitted_df = divis.split_neuron_dataframe_informationally(df, ['area', 'animal'])

#specifically to acess task and behav. state -- this is joined with the meta-index above via get_matching_pairs below
splitted_epoch_dataframe = divis.split_neuron_dataframe_informationally(task.make_epochs_dataframe(animals, corriander, conley, dave, dudley, bond, chapati), ['type'])

'''
I have to add the other animal names above
'''

#manually iterate over present areas for animal
CA3_con = splitted_df['CA3','Cor']
CA1_con = splitted_df['CA1','Cor']
#CA2_con = splitted_df['CA2','con']
#DG_con = splitted_df['DG','con']
MEG_con = splitted_df['MEC', 'Cor']


#get_matching_pairs associates the splitted_df with the more specific access index such as behavioral state.
sleep_CA1 = divis.get_matching_pairs(CA1_con, splitted_epoch_dataframe['sleep',].index)
wake_CA1 = divis.get_matching_pairs(CA1_con, splitted_epoch_dataframe['run',].index)

sleep_CA3 = divis.get_matching_pairs(CA3_con, splitted_epoch_dataframe['sleep',].index)
wake_CA3 = divis.get_matching_pairs(CA3_con, splitted_epoch_dataframe['run',].index)

##ADD DIVIS BELOW
#sleep_CA2 = get_matching_pairs(CA2_con, splitted_epoch_dataframe['sleep',].index)
#wake_CA2 = get_matching_pairs(CA2_con, splitted_epoch_dataframe['run',].index)

sleep_MEG = divis.get_matching_pairs(MEG_con, splitted_epoch_dataframe['sleep',].index)
wake_MEG = divis.get_matching_pairs(MEG_con, splitted_epoch_dataframe['run',].index)

#sleep_DG = get_matching_pairs(DG_con, splitted_epoch_dataframe['sleep',].index)
#wake_DG = get_matching_pairs(DG_con, splitted_epoch_dataframe['run',].index)


list_id_wake = wake_CA3['neuron_id'].tolist()
list_id_sleep = sleep_CA3['neuron_id'].tolist()

#list_id = wake_CA2['neuron_id'].tolist()
#list_id = sleep_CA2['neuron_id'].tolist()

#list_id_wake = wake_CA1['neuron_id'].tolist()
#list_id_sleep = sleep_CA1['neuron_id'].tolist()

#list_id = wake_DG['neuron_id'].tolist()
#list_id = sleep_DG['neuron_id'].tolist()

#list_id = wake_MEG['neuron_id'].tolist()
#list_id = sleep_MEG['neuron_id'].tolist()


#get the neuron_ids generally for non-specific analyses
#neuron_ids = df['neuron_id'].tolist()


#note that there may be a bug in dudley, and you may need to pull out the last day 06


# In[5]:


df
'''Instructions for running everything: Sometimes the kernel has to be restarted due to an unknown acessing problem.
Usually it works after 2-3 times at most. The animal short name /Cor, etc. has to  be added to the animal_file_path before
generating the spike indicator dict. Care must be taken in updating the list_ID for the correct area and behav state.
'''


# In[196]:


#Getting the epochs per day, relative to brain region and sleep/run
sleep_epochs_per_day = behav.get_epochs_by_day(sleep_CA3.index)
wake_epochs_per_day = behav.get_epochs_by_day(wake_CA3.index)

#generating a dict, relative to brain region and sleep/run, of the neuron_ids per epoch per day
embedded_day_epoch_dict_sleep = behav.embedded_day_epoch_dict(sleep_epochs_per_day, list_id_sleep)
embedded_day_epoch_dict_wake = behav.embedded_day_epoch_dict(wake_epochs_per_day, list_id_wake)

#checking for overlap of the epoch/day combinations for sleep/run and generating the final conjoinded dict of run/sleep, day, epoch, neuron_id
conjoined_key_epochs_per_day_dict = behav.conjoined_dict_with_overlap_checked(embedded_day_epoch_dict_sleep, embedded_day_epoch_dict_wake)


# In[ ]:


#spike_dict = sdd.generate_spike_indicator_dict(list_id, animals)
spike_dict = behav.coarse_grained_spike_generator_dict(conjoined_key_epochs_per_day_dict, animals)


# In[39]:


def run_analysis(embedded_spike_time_series_dict, numboot, coefficientmethod, targetdir, title, dt, dtunit, tmin, tmax, fitfuncs):
    output_dict = {}
    for state in embedded_spike_time_series_dict:
        for day in embedded_spike_time_series_dict[state]:
            for epoch in embedded_spike_time_series_dict[state][day]:
                output_dict[state][day][epoch] = mre.full_analysis(
                    data= embedded_spike_time_series_dict[state][day][epoch],
                    numboot = 5,
                    coefficientmethod='ts',
                    targetdir='./output',
                    title='Full Analysis',
                    dt=4, dtunit='ms',
                    tmin=0, tmax=8000,
                    fitfuncs=['complex'])
    return output_dict



def sum_time_series_embedded_state_day_epoch_key_values_dict(time_series_dict):
    state_day_epoch_time_series_dict = {}
    for state in time_series_dict:
        for day in time_series_dict[state]:
            for epoch in time_series_dict[state][day]:
                for neuron_key_str in range(len(time_series_dict[state][day][epoch])):
                    if type(time_series_dict[state][day][epoch][neuron_key_str]) != str:
                        num_bins = max([len(neuron_key_str) for neuron_key_str in time_series_dict[state][day][epoch] if neuron_key_str is not None])
                        res = np.zeros(num_bins)
                        for arr in time_series_dict[state][day][epoch]:
                            if type(arr) != str and arr is not None:
                                res[:len(arr)] += arr
                        state_day_epoch_time_series_dict.setdefault(state, {}).setdefault(day, {})[epoch] = res
    return state_day_epoch_time_series_dict


# In[93]:


data


# In[40]:


#type(spike_dict['wake'][3][1][0])
spike_dict_summed = sum_time_series_embedded_state_day_epoch_key_values_dict(spike_dict)


# In[41]:


def add_padded_arrays_by_epoch(time_series_dict):
    state_day_epoch_time_series_dict = {}
    
    for state in time_series_dict:
        state_day_epoch_time_series_dict[state] = {}
        
        for day in time_series_dict[state]:
            for epoch in time_series_dict[state][day]:
                if epoch not in state_day_epoch_time_series_dict[state]:
                    state_day_epoch_time_series_dict[state][epoch] = time_series_dict[state][day][epoch]
                else:
                    curr_arr = state_day_epoch_time_series_dict[state][epoch]
                    new_arr = time_series_dict[state][day][epoch]
                    
                    max_len = max(len(curr_arr), len(new_arr))
                    padded_curr_arr = np.pad(curr_arr, (0, max_len - len(curr_arr)))
                    padded_new_arr = np.pad(new_arr, (0, max_len - len(new_arr)))
                    
                    state_day_epoch_time_series_dict[state][epoch] = padded_curr_arr + padded_new_arr

    return state_day_epoch_time_series_dict

data = add_padded_arrays_by_epoch(spike_dict_summed)


def run_analysis(embedded_spike_time_series_dict, numboot, coefficientmethod, targetdir, title, dt, dtunit, tmin, tmax, fitfuncs):
    output_dict = {'wake': {}, 'sleep': {}}  
    
    for state in embedded_spike_time_series_dict:
        for epoch in embedded_spike_time_series_dict[state]:
            if state not in output_dict:  
                output_dict[state] = {} 
            
            output_dict[state][epoch] = mre.full_analysis(
                data=embedded_spike_time_series_dict[state][epoch],
                numboot=numboot,
                coefficientmethod=coefficientmethod,
                targetdir=targetdir,
                title=title,
                dt=dt,
                dtunit=dtunit,
                tmin=tmin,
                tmax=tmax,
                fitfuncs=fitfuncs
            )
    
    return output_dict

output_handler_dict = run_analysis(data, numboot=5, coefficientmethod='ts', targetdir='./output', title='My Analysis', dt=4, dtunit='ms', tmin=0, tmax=8000, fitfuncs=['complex'])


# In[189]:


def output_handler_dict_data_generator(output_handler_dict, data, data_all_animals_dict):
    output_dict = {}
    for state in data:
        output_dict[state] = {}
        for day in data[state]:
            output_dict[state][day] = {}
            output_dict[state][day]['tau'] = output_handler_dict[state][day].fits[0].tau
            output_dict[state][day]['branching_factor'] = output_handler_dict[state][day].fits[0].mre
    
    return output_dict
                        


# In[190]:


animal_results = output_handler_dict_data_generator(output_handler_dict, data, data_all_animals_dict)


# In[191]:


animal_results


# In[ ]:


#does not yet work
def output_handler_dict_data_generator(output_handler_dict, animal_name, data, data_all_animals_dict, area):
    output_dict = {}
    for state in data:
        output_dict[state] = {}
        for day in data[state]:
            output_dict[state][day] = {}
            output_dict[state][day]['tau'] = output_handler_dict[state][day].fits[0].tau
            output_dict[state][day]['branching_factor'] = output_handler_dict[state][day].fits[0].mre
    
    if animal_name in data_all_animals_dict:
        if area in data_all_animals_dict[animal_name]:
            for state in output_dict:
                if state in data_all_animals_dict[animal_name]:
                    for day in output_dict[state]:
                        if day in data_all_animals_dict[animal_name][state]:
                            data_all_animals_dict[animal_name][state][area][day].update(output_dict[state][day][area])
                        else:
                            data_all_animals_dict[animal_name][area][state][day] = output_dict[state][day][area]
                else:
                    data_all_animals_dict[animal_name][area][state] = output_dict[state]
    else:
        data_all_animals_dict[animal_name][area] = output_dict
    
    return data_all_animals_dict

