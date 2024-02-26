#!/usr/bin/env python
# coding: utf-8

# In[4]:

import spike_data_frame_new as sdd
import numpy as np
import pandas as pd


def sleep_border_detection(speed_series, speed_threshold_sleep, length_no_pre_movement, current_element):
    ''''
    This function returns a binary value of whether the animal was still for all last n seconds. Can be used as a demarcation of 
    whether the animal was sleeping.
    '''
    current_element = pd.Timedelta(current_element, unit='s')
    start_time = current_element - pd.Timedelta(seconds=length_no_pre_movement)
    last_n_seconds = speed_series.loc[start_time:current_element]
    
    if all(speed <= speed_threshold_sleep for speed in last_n_seconds.values):
        return True
    else:
        return False

def awake_border_detection(speed_series, speed_threshold_sleep, length_no_pre_movement, current_element):
    ''''
    This function returns a binary value of whether there was movement in the last n seconds. Can be used as a demarcation of 
    whether the animal is wake in rest epoch.
    '''
    current_element = pd.Timedelta(current_element, unit='s')
    start_time = current_element - pd.Timedelta(seconds=length_no_pre_movement)
    last_n_seconds = speed_series.loc[start_time:current_element]
    
    if any(speed >= speed_threshold_sleep for speed in last_n_seconds.values):
        return True
    else:
        return False



def subdivision_run_sleep_rest_times_dict(speed_series, speed_threshold_sleep, length_no_pre_movement, epoch_type, awake_last_n_seconds):
    #Do I need speed and head_speed, or is running also individuated by head_speed?
    #I think I have to give the rest/sleep epoch as an input arguement since it may be relative to the epoch.
    
    if epoch_type == 'resting':
        rest_epoch = True
        run_epoch = False
    if epoch_type == 'running':
        run_epoch = True
        rest_epoch = False
        
    subdivised_by_behav_state_dict = {}
    
    awake_series_rest = pd.Series()
    resting_series_rest = pd.Series()
    sleeping_series_rest =  pd.Series()
    
    running_series_run = pd.Series()
    resting_series_run = pd.Series()
    sleeping_series_run =  pd.Series()
    
    for i in speed_series.index:
        
        #If the (head?)speed of the animal is above threshold in the run_epoch
        if speed_series[i]> speed_threshold_sleep and run_epoch == True:  
            
            temp_series = pd.Series(speed_series[i], index=[i])
            running_series_run = pd.concat([running_series_run, temp_series])
         
        #If you are in the rest epoch, you are simply not running (that's how they did it in the nature paper)
        #If you are in the rest period and the head_speed is above threshold, then you are "awake"
        elif speed_series[i]> speed_threshold_sleep and rest_epoch:  
            
            temp_series = pd.Series(speed_series[i], index=[i])
            awake_series_rest = pd.concat([awake_series_rest, temp_series])
            
        #If the head_speed is below threshold, but it has been above threshold somewhere in the last 7 seconds --> awake
        elif speed_series[i]< speed_threshold_sleep and awake_border_detection(speed_series, speed_threshold_sleep, awake_last_n_seconds, i) == True and rest_epoch:
            
            temp_series = pd.Series(speed_series[i], index=[i])
            awake_series_rest = pd.concat([awake_series_rest, temp_series])
            
            
            
        #if youre resting, in run or rest epoch
        elif speed_series[i] <= speed_threshold_sleep and run_epoch and sleep_border_detection(speed_series, speed_threshold_sleep, length_no_pre_movement, i) == False:
            #you have to make a buffer condition here that deals with the buffer from the paper (the buffer may be relevant
            #other analyses methods as well)
            #I also have to add the SWR condition here
            temp_series = pd.Series(speed_series[i], index=[i])
            resting_series_run = pd.concat([resting_series_run, temp_series])
            
        elif speed_series[i] <= speed_threshold_sleep and rest_epoch and sleep_border_detection(speed_series, speed_threshold_sleep, length_no_pre_movement, i) == False:
            #you have to make a buffer condition here that deals with the buffer from the paper (the buffer may be relevant
            #other analyses methods as well)
            #I also have to add the SWR condition here
            temp_series = pd.Series(speed_series[i], index=[i])
            resting_series_rest = pd.concat([resting_series_rest, temp_series])
            
            
            
        #if youre sleeping (60 s no movement), for either run or rest 
        elif speed_series[i] <= speed_threshold_sleep and run_epoch and sleep_border_detection(speed_series, speed_threshold_sleep, length_no_pre_movement, i) == True:
            temp_series = pd.Series(speed_series[i], index=[i])
            sleeping_series_run = pd.concat([sleeping_series_run, temp_series])
            
        elif speed_series[i] <= speed_threshold_sleep and rest_epoch and sleep_border_detection(speed_series, speed_threshold_sleep, length_no_pre_movement, i) == True:
            temp_series = pd.Series(speed_series[i], index=[i])
            sleeping_series_rest = pd.concat([sleeping_series_rest, temp_series])
            
    subdivised_by_behav_state_dict['sleeping run epoch'] = sleeping_series_run #run epoch AND not above speed threshold for n (60) last seconds  
    subdivised_by_behav_state_dict['resting run epoch'] = resting_series_run #run epoch AND above th. somewhere in last n seconds but not now
    subdivised_by_behav_state_dict['running run epoch'] = running_series_run #run epoch AND head speed above threshold 
    
    subdivised_by_behav_state_dict['sleeping rest epoch'] = sleeping_series_rest #rest epoch AND not above speed thr. for n (60) last seconds
    subdivised_by_behav_state_dict['resting rest epoch'] = resting_series_rest # rest epoch AND above th. somewhere in last n (60) seconds, but not now
    subdivised_by_behav_state_dict['awake rest epoch'] = awake_series_rest #rest epoch AND somewhere in the last 7 seconds or now above threshold
    
    
    return subdivised_by_behav_state_dict

 

#subdivised_behav_state_dict = subdivision_run_sleep_rest_times_dict(head_speed_over_time, speed_threshold_sleep= 4, length_no_pre_movement= 60, epoch_type = 'resting', awake_last_n_seconds=5)


# In[24]:


#subdivised_behav_state_dict


# In[5]:


def get_epochs_by_day(multi_index):
    '''
    Returns: A dict with keys days, and values lists of epochs per day
    '''
    epochs_by_day = {}
    
    for index in multi_index:
        day = index[1]
        epoch = index[2]
        
        if day in epochs_by_day:
            if epoch not in epochs_by_day[day]:
                epochs_by_day[day].append(epoch)    
        else:
            epochs_by_day[day] = [epoch]
    
    return epochs_by_day

#awake_epochs_per_day_dict = get_epochs_by_day(wake_CA3.index) # getting all the epochs per day for specific animal
#asleep_epochs_per_day_dict = get_epochs_by_day(sleep_CA3.index)
#use this output to colout the resulting plot

#get_epochs_by_day(CA1_con.index)

def embedded_day_epoch_dict(get_epochs_per_day_dict, neuron_id_list):
    '''
    Input: epochs_per_day_dict with days as keys and lists of epochs per day as values
    
    Returns: embedded dict of dicts, where the outer dict contains the days as keys and the epoch dicts as (values/sub-keys).
             The inner values are the neuron ids per epoch (inner dict) per day (outer dict)
    '''
    epochs_per_day_dict = {}
    neuron_id_component_list = []
    
    for neuron_key_str in neuron_id_list:
        animal_short_name, day_number, epoch_number, tetrode_number, neuron_number = neuron_key_str.split("_")
        neuron_id_component_list.append((animal_short_name, day_number, epoch_number, tetrode_number, neuron_number))
    
    for key in get_epochs_per_day_dict:
        inner_dict = {}
        for value in get_epochs_per_day_dict[key]:
            inner_dict[value] = []
            for neuron_id in neuron_id_component_list:
                if neuron_id[1] == '0'+str(key) and neuron_id[2] == '0'+str(value):
                    inner_dict[value].append('_'.join(neuron_id))
        epochs_per_day_dict[key] = inner_dict
    return epochs_per_day_dict

#embedded_day_epoch_dict(get_epochs_by_day(CA1_con.index), neuron_ids)


# In[ ]:


def day_getter(state_day_epoch_neuron_key_dict, day, epoch):
    '''Input: embedded state_day_epoch_neuron_key dict
        Returns: all neuron keys from one day (one day is already either sleep or run)'''
    if state_day_epoch_neuron_key_dict['wake'][day][epoch] != None:
        return state_day_epoch_neuron_key_dict['wake'][day][epoch] 
    elif state_day_epoch_neuron_key_dict['sleep'][day][epoch] != None:
        return state_day_epoch_neuron_key_dict['sleep'][day][epoch]    
    


def generate_spike_indicator_dict_relative_to_behav_state(behav_state_dict, day_specific_state_day_epoch_neuron_key_dict, animals):
    '''Input: - behav_state_dict: behav state classification by head speed
            - day_specific_state_day_epoch_neuron_key_dict: embedded day specific state_day_epoch_neuron_key dict
        Returns: spike indicator dict of spike trains per category
    '''
    
    association_dict = {}

   
    for neuron_key_str in day_specific_state_day_epoch_neuron_key_dict:
        
                    
        animal_short_name, day_number, epoch_number, tetrode_number, neuron_number = neuron_key_str.split("_")
        neuron_key = (animal_short_name, int(day_number), int(epoch_number), int(tetrode_number), int(neuron_number))
        
        
                    #try:
        spike_time_array = spike_time_index_association(neuron_key, animals)
                        
        association_dict = time_association_behav_state_spike_time(spike_time_array, day_specific_state_day_epoch_neuron_key_dict, behav_state_dict)
    
                    #except AttributeError:
                        #spike_time_array = None
                        #print(f"No spike indicator data for neuron: {neuron_key}")
    return association_dict
                



def conjoined_dict_with_overlap_checked(wake_day_epoch_dict, sleep_day_epoch_dict):
    '''Input: wake_day_epoch_dict, sleep_day_epoch_dict -- Behav State specific (relative to epoch categorization) day epoch combinations'
    Returns: embedded dict with outher keys behav state, inner keys day, and double inner keys epoch, with overlapping combinations deleted'
    '''
    final_dict = {'wake': {}, 'sleep': {}}
    overlap_counter = 0
    
    for i in wake_day_epoch_dict.keys():
        for j in wake_day_epoch_dict[i].keys():
            if i in sleep_day_epoch_dict and j in sleep_day_epoch_dict[i]:
                if wake_day_epoch_dict[i][j] == sleep_day_epoch_dict[i][j]:
                    overlap_counter += 1
            else:
                final_dict['wake'].setdefault(i, {})[j] = wake_day_epoch_dict[i][j]
    
    for i in sleep_day_epoch_dict.keys():
        for j in sleep_day_epoch_dict[i].keys():
            if i in wake_day_epoch_dict and j in wake_day_epoch_dict[i]:
                if wake_day_epoch_dict[i][j] == sleep_day_epoch_dict[i][j]:
                    overlap_counter += 1
            else:
                final_dict['sleep'].setdefault(i, {})[j] = sleep_day_epoch_dict[i][j]
    
     
                
    return final_dict 

      
#This function does not yet work due to the time-index
def time_association_behav_state_spike_time(spike_time_array, day_specific_state_day_epoch_neuron_key_dict, behav_state_dict):
    '''Input: - day_epoch_neuron_key_dict: embedded dict with outer key day, inner key epoch, and values neuron ids per day, epoch
              - subdivised_behav_state_within_epoch_dict: df with time as index and categorized head speed per behavioral subcategory within epoch
        Returns: embedded dict with values of spike trains categorized according to behavioral category within epoch (day, epoch, behav cat are keys)
    '''   
    
                           
    behav_state_dict_index = subdivised_behav_state_dict.index.tolist()
    spike_time_array_index = spike_time_array.index.tolist()
    
    behave_times = [pd.Timestamp('1900-01-01') + index for index in behav_state_dict_index]
    behave_time_values = [str(time) for time in behave_times]
    
    
    spike_times = [pd.Timestamp('1900-01-01') + index for index in spike_time_array_index]

    for i in spike_times:
        print(spike_times)
    if i in behave_time_values:
        print('yes')
    else:
        print('fuck')
    for i in spike_times:
        if i in behave_times:
            #key_value = behav_state_dict[i].key()
            i = pd.Timedelta(i)
            key_value = list(behave_state_dict.keys())[i]
            print(key_value)
            day_specific_state_day_epoch_neuron_key_dict += spike_time_array
        #else:
         #   print(type(i)) #+ 'is not in behav state dict index..')
            
            
    return state_day_epoch_neuron_key_dict

def coarse_grained_spike_generator_dict(state_day_epoch_neuron_key_dict, animals):
    '''Input: state_day_epoch_neuron_key_dict: embedded state_day_epoch_neuron_key dict
        Returns: spike indicator dict of spike trains per day epoch combination
    '''
    for state in state_day_epoch_neuron_key_dict:
        for day in state_day_epoch_neuron_key_dict[state]:
            for epoch in state_day_epoch_neuron_key_dict[state][day]:
                neuron_key_counter = 0
                for neuron_key_str in state_day_epoch_neuron_key_dict[state][day][epoch]:
                    if type(neuron_key_str) == str:
                        animal_short_name, day_number, epoch_number, tetrode_number, neuron_number = neuron_key_str.split("_")
                        neuron_key = (animal_short_name, int(day_number), int(epoch_number), int(tetrode_number), int(neuron_number))
                        spike_time_array = None
                        try:
                            spike_time_array = sdd.spike_time_index_association(neuron_key, animals).values.astype(np.int32)
                            neuron_key_counter += 1
                            state_day_epoch_neuron_key_dict[state][day][epoch][neuron_key_counter] = spike_time_array
                            
                        except AttributeError:
                            spike_time_array = None
                            print(f"No spike indicator data for neuron: {neuron_key}")
                        
                        #neuron_key_list = []
                        #for i in range(len(state_day_epoch_neuron_key_dict[state][day][epoch])-1):
                         #   neuron_key_list.append(i)
                            
                        
                        
    return state_day_epoch_neuron_key_dict


                