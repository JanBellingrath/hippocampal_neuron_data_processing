
 import mrestimator as mre
import pandas as pd

def add_padded_arrays_by_epoch(time_series_dict):
    '''Input: embedded dict with summed spike trains (over neurons) without time index
        Returns: embedded dict without "day" as all the days have been summed per epoch (day 1, epoch 1, day 2, epoch 1, etc.)'''
    
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

def run_analysis_without_time_index_summed_over_days(embedded_spike_time_series_dict, numboot, coefficientmethod, targetdir, title, dt, dtunit, tmin, tmax, fitfuncs):
    '''
    Input: embedded_dict with spike trains summed over neurons and days (result from add_padded_arrays_by_epoch), other parameters are from MRE
        Returns: output_dict of parameters for state and epoch
       '''
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


def run_analysis_with_time_index_and_time_chunks(embedded_spike_time_series_dict, numboot, coefficientmethod, targetdir, title, dt, dtunit, tmin, tmax, fitfuncs):
     
    output_dict = {}  
    
    for state in embedded_spike_time_series_dict:
        output_dict[state] = {}
        for day in embedded_spike_time_series_dict[state]:
            output_dict[state][day]= {}
            for epoch in embedded_spike_time_series_dict[state][day]:
                output_dict[state][day][epoch] = {}
                for time_chunk in embedded_spike_time_series_dict[state][day][epoch]:
                    output_dict[state][day][epoch][time_chunk] = {}
                    
                    try:
                        output_dict[state][day][epoch][time_chunk] = mre.full_analysis(
                            data=embedded_spike_time_series_dict[state][day][epoch][time_chunk].values,
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
                    except:
                        continue
                        
    final_output_dict = {}
    for state in output_dict:
        final_output_dict = {}
        for day in output_dict[state]:
            final_output_dict[state][day] = {}
            for epoch in output_dict[state][day]:
                final_output_dict[state][day][epoch] = {}
                for time_chunk in output_dict[state][day][epoch]:
                    final_output_dict[state][day][epoch][time_chunk]['tau'] = output_handler_dict[state][day][epoch][time_chunk].fits[0].tau
                    final_output_dict[state][day][epoch][time_chunk]['branching_factor'] = output_handler_dict[state][day][epoch][time_chunk].fits[0].mre
    print('Data analysis successfull!')
    return final_output_dict               
   

def run_analysis_without_time_sep(embedded_spike_time_series_dict, numboot, coefficientmethod, targetdir, title, dt, dtunit, tmin, tmax, fitfuncs):
    output_dict = {}
    for state in embedded_spike_time_series_dict:
        output_dict[state] = {}
        for day in embedded_spike_time_series_dict[state]:
            output_dict[state][day] = {}
            for epoch in embedded_spike_time_series_dict[state][day]:
                output_dict[state][day][epoch] = {}
                
                try: 
                    output_dict[state][day][epoch] = mre.full_analysis(
                        data= embedded_spike_time_series_dict[state][day][epoch].values,
                        numboot = 5,
                        coefficientmethod='ts',
                        targetdir='./output',
                        title='Full Analysis',
                        dt=dt, dtunit=dtunit,
                        tmin=tmin, tmax=tmax,
                        fitfuncs=['complex'])
                except:
                    pass
                
    return output_dict

def output_handler_dict_data_generator_small(output_handler_dict, data):
    output_dict = {}
    for state in data:
        output_dict[state] = {}
        for day in data[state]:
            output_dict[state][day] = {}
            for epoch in data[state][day]:
                output_dict[state][day]['tau'] = output_handler_dict[state][day].fits[0].tau
                output_dict[state][day]['branching_factor'] = output_handler_dict[state][day].fits[0].mre
    
    return output_dict

def output_handler_dict_data_generator(output_handler_dict, data):
    output_dict = {}
    for state in data:
        output_dict[state] = {}
        for day in data[state]:
            output_dict[state][day] = {}
            for epoch in data[state][day]:
                for time_chunk in data[state][day][epoch]:
                    output_dict[state][day][epoch][time_chunk]['tau'] = output_handler_dict[state][day][epoch].fits[0].tau
                    output_dict[state][day][epoch][time_chunk]['branching_factor'] = output_handler_dict[state][day].fits[0].mre
    
    return output_dict
                        
    
#does not yet work
def output_handler_dict_data_generator_what(output_handler_dict, animal_name, data, data_all_animals_dict, area):
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



def bijective_output_to_index_mapping(embedded_spike_time_series_dict, data):
    '''
    This function is only important for the case in which the estimator is run once (with all data in trial structure)
    Input: data: output data from the estimator and associates it back to the state/day/epoch/time_chunk combination
            dict: dict used for bijective index combination association
    Returns: final data for animal as dataframe
    '''
    
    final_output_dict = {}
    
    #this num is used to associate the list of outputs with specific day/epoch/... combinations..
    list_dict_association_num = 0
    for state in embedded_spike_time_series_dict:
        final_output_dict[state] = {}
        for day in embedded_spike_time_series_dict[state]:
            final_output_dict[state][day] = {}
            for epoch in embedded_spike_time_series_dict[state][day]:
                final_output_dict[state][day][epoch] = {}
                for time_chunk in embedded_spike_time_series_dict[state][day][epoch]:
                    final_output_dict[state][day][epoch][time_chunk]['tau'] = data[list_dict_association_num].fits[0].tau
                    final_output_dict[state][day][epoch][time_chunk]['branching_factor'] = data[list_dict_association_num].fits[0].mre
                    list_dict_association_num += 1
                    
    return pd.DataFrame(final_output_dict)


def non_finite_element_checker(splitted_by_sec_spike_dict):
    '''
    Returns the unique state, day, epoch, time_chunk combination in which a corr coeff is resultant which is non-finite
    '''
    
    non_finite_element_list = []
    for state in splitted_by_sec_spike_dict:
        for day in splitted_by_sec_spike_dict[state]:
            for epoch in splitted_by_sec_spike_dict[state][day]:
                for time_chunk in splitted_by_sec_spike_dict[state][day][epoch]:
                    rk = mre.coefficients(splitted_by_sec_spike_dict[state][day][epoch][time_chunk].values, dtunit=0.000666)
                    for i in rk.coefficients:
                        if not np.isfinite(i):
                            time_chunk_key = '_'.join([str(state), str(day) ,str(epoch), str(time_chunk)])
                            non_finite_element_list.append(time_chunk_key)  
                            
    non_finites_unique = set(non_finite_element_list)
    non_finite_element_list = list(non_finites_unique)
    return non_finite_element_list

import numpy as np

def run_analysis(embedded_spike_time_series_dict, numboot, coefficientmethod, targetdir, title, dt, dtunit, tmin, tmax, fitfuncs):
    '''
    Difference to 'run_analysis_with_time_index_and_time_chunks': this function runs the mre estimator once, not per time_chunk.
    Input: 
        embedded_spike_time_series_dict: time index associated neuronal data per sub category
        other inputs: specific to estimator
    Returns: final data per animal relative to day/epoch/etc. as dataframe
    '''
    
    trial_ndarray = None  
    non_finite_element_list = non_finite_element_checker(embedded_spike_time_series_dict)

    for state in embedded_spike_time_series_dict:
        for day in embedded_spike_time_series_dict[state]:
            for epoch in embedded_spike_time_series_dict[state][day]:
                for time_chunk in embedded_spike_time_series_dict[state][day][epoch]:
                    values = list(embedded_spike_time_series_dict[state][day][epoch][time_chunk].values)
                    
                    ndarray_values = np.array(values)
                    key = '_'.join([str(state), str(day), str(epoch), str(time_chunk)])
                    
                    if key in non_finite_element_list:
                        continue
                        
                    else:
                        if trial_ndarray is None:
                            trial_ndarray = ndarray_values
                        else:
                            trial_ndarray = np.vstack((trial_ndarray, ndarray_values))
    
    output_list = mre.full_analysis(
                            data = trial_ndarray,
                            numboot=numboot,
                            coefficientmethod= 'ts',
                            targetdir=targetdir,
                            title=title,
                            dt=dt,
                            dtunit=dtunit,
                            tmin=tmin,
                            tmax=tmax,
                            fitfuncs=fitfuncs)
    
    
    return output_list
# In[ ]:






