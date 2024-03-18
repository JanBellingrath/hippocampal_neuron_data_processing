#!/usr/bin/env python
# coding: utf-8

# In[4]:

import numpy as np
import pandas as pd

def sum_time_series_with_time_index_from_embedded_state_day_epoch_key_values_dict(time_series_dict, time_dict):
    '''
    Input: time_series_dict = embedded dict with spike trains of neurons stored under neuron_key_str. 
            time_dict = embedded dict with time index of spike trains
    Returns: embedded dict with summed spike trains per epoch (summation over all neurons) that are associated with a time index.'''
    
    state_day_epoch_time_series_dict = {}
    for state in time_series_dict:
        state_day_epoch_time_series_dict[state] = {}
        for day in time_series_dict[state]:
            state_day_epoch_time_series_dict[state][day] = {}
            for epoch in time_series_dict[state][day]:
                state_day_epoch_time_series_dict[state][day][epoch] = {}
                for neuron_key_str in range(len(time_series_dict[state][day][epoch])):
                    
                    if type(time_series_dict[state][day][epoch][neuron_key_str]) is not str and time_series_dict[state][day][epoch][neuron_key_str] is not None:

                        num_bins = max([len(neuron_key_str) for neuron_key_str in time_series_dict[state][day][epoch] if neuron_key_str is not None])
                        
                        res = np.zeros(num_bins)
                        for arr in time_series_dict[state][day][epoch]:
                            if type(arr) != str and arr is not None:
                                res[:len(arr)] += arr
                        
                        
                        
                        keys_list = list(time_dict[state][day][epoch].keys())
                        
                        
                        element_key = keys_list[0]
                            
                        time = time_dict[state][day][epoch][element_key]
                        
                        series = pd.Series(res,time)
                        
                        state_day_epoch_time_series_dict[state][day][epoch] = series
                            
                        #except:
                         #   continue
    return state_day_epoch_time_series_dict

def time_index_seperator(spike_dict_summed, len_chunk = 5):
    '''
    Input: spike_dict_summed: embedded dict of summed spike trains (over all neurons) per epoch, with associated time index.
            len_chunk: len of the desired chunk of the individual parts of the spike train for analyses in s (e.g 5 s, 10 s, etc.)'''
    chunks = {}
    for state in spike_dict_summed:
        chunks[state] = {}
        for day in spike_dict_summed[state]:
            chunks[state][day] = {}
            for epoch in spike_dict_summed[state][day]:
                try:
                    chunks[state][day][epoch] = {}
                    start_time = spike_dict_summed[state][day][epoch].index[0]
                    finish_time = spike_dict_summed[state][day][epoch].index[-1]
                    time_length = (finish_time - start_time).total_seconds()
                
                    num_chunks_per_epoch = int(time_length//len_chunk)
                    remainder = time_length % len_chunk
                    #ignore remainder for now
                    start = 0
                    for i in range(num_chunks_per_epoch):
                        end = start + len_chunk*1500 #upscale to seconds again
                    
                        chunks[state][day][epoch][i]= spike_dict_summed[state][day][epoch][start:end]
                        start = end
                except:
                    continue
                    
    return chunks

