#!/usr/bin/env python
# coding: utf-8

# # This is the analysis pipeline for the calculation of tau and branching factor of neural data from the hippocampus. 
# 
# The data are take from: https://datadryad.org/stash/dataset/doi:10.7272/Q61N7ZC3

# ### Importing the branching factor/tau estimator
# 
# For documentation regards the estimator see https://mrestimator.readthedocs.io/en/latest/doc/gettingstarted.html, Spitzner, F. P., Dehning, J., Wilting, J., Hagemann, A., P. Neto, J., Zierenberg, J., & Priesemann, V. (2021).

# In[1]:

import mrestimator as mre
import numpy as np


# ### Importing relevant non-standard modules
# All of which can be found on github https://github.com/JanBellingrath/Hippocampal_Neuron_Data_Processing

# In[2]:


import pipeline.utilities as u
import criticality_analysis as can
import compactifying_functions_up_to_date_version as compact

# ### Defining each animal via its short name and its dir
# ectory

# In[3]:


from collections import namedtuple

Animal = namedtuple('Animal', {'short_name', 'directory'})

conley = Animal('/home/bellijjy/Conley.tar/Conley', 'con')
dave = Animal('/home/bellijjy/Dave.tar/Dave/Dave', 'dav')
chapati = Animal('/home/bellijjy/Chapati.tar/Chapati/Chapati', 'cha')
corriander = Animal('/home/bellijjy/Corriander.tar/Corriander', 'Cor')
dudley = Animal('/home/bellijjy/Dudley/dud', 'dud')
bond = Animal('/home/bellijjy/Bond/bon', 'bon')
frank = Animal('/local2/Jan/Frank/Frank', 'fra')
government = Animal('/local2/Jan/Government/Government/gov', 'fra')
egypt = Animal('/local2/Jan/Egypt/Egypt/egy', 'egy')
remy = Animal('/local2/Jan/Remy/Remy/remy', 'remy')


animals = {'con': Animal('con','/home/bellijjy/Conley.tar/Conley'),
           'Cor': Animal('Cor','/home/bellijjy/Corriander.tar/Corriander'),
            'cha': Animal('cha','/home/bellijjy/Chapati.tar/Chapati/Chapati'),
          'dav': Animal('dav','/home/bellijjy/Dave.tar/Dave/Dave'),
           'dud': Animal('dud','/home/bellijjy/Dudley'),
            'bon' : Animal('bon', '/home/bellijjy/Bond'),
              'fra' : Animal('fra', '/local2/Jan/Frank/Frank'),
              'gov' : Animal('gov', '/local2/Jan/Government/Government'),
            'egy' : Animal('egy', '/local2/Jan/Egypt/Egypt'), 
          'remy': Animal('remy', '/local2/Jan/Remy/Remy')}


# In[4]:


#checl the dir below its pc relative


# In[5]:


import pandas as pd
import gc
import json
import os
from tqdm.notebook import tqdm
from itertools import product

# Define your coefficient function
def get_coefficients_trial_seperated(spike_dict, state, day, epoch, time_chunk):
    return mre.coefficients(spike_dict[state][day][epoch][time_chunk], dt=6.67, dtunit='ms', method = 'ts')

#probably need to add padding below
def get_coefficients_stationary_mean(spike_dict, state, day, epoch):
    #extracting all values (in an array for each time_chunk) per epoch (per stat, day)
    all_time_chunks = spike_dict[state][day][epoch].values()
    # Assuming each time_chunk is a 1D array, combine them into a 2D array (num_trials, data_length)
    combined_trials = np.stack(all_time_chunks)

    # Calculate coefficients using the specified method for the combined trials
    coefficients = mre.coefficients(combined_trials, dt=6.67, dtunit='ms', method= 'sm')
    
    return coefficients

    
# Define your fit function
def fitting(coefficients):
    return mre.fit(coefficients.coefficients, fitfunc='f_complex')

def process_time_chunk(spike_dict, state, day, epoch, animal, area, method, time_chunk):
    
    if method == 'sm':
        
        all_time_chunks = spike_dict[state][day][epoch]

        # Convert all time chunks into a serializable format
        # Assuming each time chunk is already in a format that can be serialized (like a list or a numpy array)
        original_data = {time_chunk: data.tolist() for time_chunk, data in all_time_chunks.items()}

        # Serialize the data to JSON
        original_data_json = json.dumps(original_data)
        
        #if sm, then you fit all time chunks within one epoch.
        coefficients = get_coefficients_stationary_mean(spike_dict, state, day, epoch)
        
        #mre.fit in fitting automatically fits the array structure
        output_handler = fitting(coefficients)

        # Extract additional data from the output_handler
        additional_data = {
            'popt': output_handler.popt,
            'ssres': output_handler.ssres,
            #'fitfunc': output_handler.fitfunc,
            'pcov': [],
            'steps': output_handler.steps,
            'dt': output_handler.dt,
            'dtunit': output_handler.dtunit,
            'quantiles': output_handler.quantiles,
            'mrequantiles': output_handler.mrequantiles,
            'tauquantiles': output_handler.tauquantiles,
            'description': output_handler.description
        }

        data = {
            'animal': animal,
            'area': area,
            'state': state,
            'day': day,
            'epoch': epoch,
            'original_data': original_data_json,
            'tau': output_handler.tau,
            'branching_factor': output_handler.mre,
        }

        # Merge the additional data into the main data dictionary
        data.update(additional_data)

        #Add covariance matrix after conversion
        data['pcov'] = json.dumps(data['pcov'])

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Save individual time_chunk to disk
        directory = '/home/bellijjy/criticality_january_sm/'
        file_name = f'{directory}_{animal}_{area}_{state}_{day}_{epoch}_{time_chunk}.parquet'
        df.to_parquet(file_name, index=True)
        print('saved')
        
        #return df
    
    else: 
        
        original_data = spike_dict[state][day][epoch][time_chunk]
        original_data_json = original_data.to_json()
        
        coefficients = get_coefficients_trial_seperated(spike_dict, state, day, epoch, time_chunk)
        output_handler = get_output_handler(coefficients)

        # Extract additional data from the output_handler
        additional_data = {
            'popt': output_handler.popt,
            'ssres': output_handler.ssres,
            #'fitfunc': output_handler.fitfunc,
            'pcov': [],
            'steps': output_handler.steps,
            'dt': output_handler.dt,
            'dtunit': output_handler.dtunit,
            'quantiles': output_handler.quantiles,
            'mrequantiles': output_handler.mrequantiles,
            'tauquantiles': output_handler.tauquantiles,
            'description': output_handler.description
        }

        data = {
            'animal': animal,
            'area': area,
            'state': state,
            'day': day,
            'epoch': epoch,
            'time_chunk': time_chunk,
            'original_data': original_data_json,
            'tau': output_handler.tau,
            'branching_factor': output_handler.mre,
        }

        # Merge the additional data into the main data dictionary
        data.update(additional_data)

        #Add covariance matrix after conversion
        data['pcov'] = json.dumps(data['pcov'])

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Save individual time_chunk to disk
        #directory = '/local2/Jan/criticality_analysis_areas_state_sm/'
        directory = '/home/bellijjy/criticality_january_sm/'
        file_name = f'{directory}_{animal}_{area}_{state}_{day}_{epoch}_{time_chunk}.parquet'
        df.to_parquet(file_name, index=True)
                            

def compute_rk_and_tau_from_splitted_by_sec_spike_dict_trial_seperated(spike_dict, animal, area): # this function may need updating
    states = spike_dict.keys()
    days = list(set(day for state in states for day in spike_dict[state].keys()))
    epochs = list(set(epoch for state in states for day in days if day in spike_dict[state] for epoch in spike_dict[state][day].keys()))
    time_chunks = list(set(time_chunk for state in states for day in days if day in spike_dict[state] for epoch in epochs if epoch in spike_dict[state][day] for time_chunk in spike_dict[state][day][epoch].keys()))
    
    total_time_chunks = len(time_chunks)
    progress_bar = tqdm(total=total_time_chunks, desc="Processing", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    
    for state, day, epoch, time_chunk in product(states, days, epochs, time_chunks):
        if state in spike_dict and day in spike_dict[state] and epoch in spike_dict[state][day] and time_chunk in spike_dict[state][day][epoch]:
            
            process_time_chunk(spike_dict, state, day, epoch, time_chunk, animal, area)
            #except:
          #      continue
                
            progress_bar.update()
            gc.collect()
            
    progress_bar.close()
    return ('Finished and saved to disk')

def compute_rk_and_tau_from_splitted_by_sec_spike_dict_stationary_mean(spike_dict, animal, area): # this function may need updating
    states = spike_dict.keys()
    days = list(set(day for state in states for day in spike_dict[state].keys()))
    epochs = list(set(epoch for state in states for day in days if day in spike_dict[state] for epoch in spike_dict[state][day].keys()))
    
    
    for state, day, epoch  in product(states, days, epochs):
        if state in spike_dict and day in spike_dict[state] and epoch in spike_dict[state][day]:
            try:
                process_time_chunk(spike_dict = spike_dict, state = state, day = day, epoch = epoch, animal = animal, area = area, method = 'sm', time_chunk = None)
            except Exception as e:
                print(f"Error processing for state: {state}, day: {day}, epoch; {epoch}. Error: {e}")
                continue
                
            
            gc.collect()
            
    return ('Finished and saved to disk')



# In[ ]:


animal_list = ['egy', 'gov', 'fra', 'con', 'cha', 'Cor', 'dud', 'dav']
area_list = ['CA3', 'CA1']
len_time_chunk = 45

def intrinsic_time_scale_estimation(animal, area, len_time_chunk):
    neuron_ids = compact.neuron_ids_for_specific_animal_and_subarea(area, animal)
    splitted_by_sec_spike_dict = compact.get_spike_data(neuron_ids, len_time_chunk, area, animal)
    compute_rk_and_tau_from_splitted_by_sec_spike_dict_stationary_mean(splitted_by_sec_spike_dict, animal, area)
    return 'Done'

def intrinsic_time_scale_estimation_for_all_animals(animal_list, area_list, len_time_chunk):
    for animal in animal_list:
        for area in area_list:
            
            try:
                # Attempt to run the estimation function for each combination
                print(f"Processing for animal: {animal}, area: {area}")
                intrinsic_time_scale_estimation(animal, area, len_time_chunk)
            except Exception as e:
             #   # If an error occurs, print the error message and continue with the next combination
                print(f"Error processing for animal: {animal}, area: {area}. Error: {e}")
                
    return 'All combinations processed'

intrinsic_time_scale_estimation_for_all_animals(animal_list, area_list, len_time_chunk)

