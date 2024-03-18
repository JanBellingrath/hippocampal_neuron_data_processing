import pipeline.compactifying_functions_up_to_date_version as compact
import mrestimator as mre
import numpy as np
import pandas as pd
import gc
import json
from itertools import product


def intrinsic_time_scale_estimation_for_all_animals(animal_list, area_list, len_time_chunk, animals_dict, mode="smooth"):
    '''
    iterates over animal_list and area_list.
    Mode either set to "smooth" (automatic error handling),
    or to "debug" (analysis will be aborted when error occurs, but it will be easier to trace back)
    '''
    for animal in animal_list:
        for area in area_list:
            
            if mode == "smooth":
                try:
                    print(f"Processing for animal: {animal}, area: {area}")
                    intrinsic_time_scale_estimation(animal, area, len_time_chunk, animals_dict)
            
                except Exception as e:
                    print(f"Error processing for animal: {animal}, area: {area}. Error: {e}")
            

            elif mode == "debug":
                print(f"Processing for animal: {animal}, area: {area}")
                intrinsic_time_scale_estimation(animal, area, len_time_chunk)
                
    return 'All combinations processed'



def intrinsic_time_scale_estimation(animal, area, len_time_chunk, animals_dict):
    '''
    gets the unique neuron_ids for animal and area.
    '''
    neuron_ids = compact.neuron_ids_for_specific_animal_and_subarea(area, animal, animals_dict)
    splitted_by_sec_spike_dict = compact.get_spike_data(neuron_ids, len_time_chunk, area, animal)
    compute_rk_and_tau_from_splitted_by_sec_spike_dict_trial_seperated(splitted_by_sec_spike_dict, animal, area)
    return 'Done'



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
        directory = "/home/dekorvyb/Documents/gov_CA3/"
        
        file_name = f'{directory}_{animal}_{area}_{state}_{day}_{epoch}_{time_chunk}.parquet'
        df.to_parquet(file_name, index=True)
                            

def compute_rk_and_tau_from_splitted_by_sec_spike_dict_trial_seperated(spike_dict, animal, area): # this function may need updating
    states = spike_dict.keys()
    days = list(set(day for state in states for day in spike_dict[state].keys()))
    epochs = list(set(epoch for state in states for day in days if day in spike_dict[state] for epoch in spike_dict[state][day].keys()))
    time_chunks = list(set(time_chunk for state in states for day in days if day in spike_dict[state] for epoch in epochs if epoch in spike_dict[state][day] for time_chunk in spike_dict[state][day][epoch].keys()))
    
    total_time_chunks = len(time_chunks)
    #progress_bar = tqdm(total=total_time_chunks, desc="Processing", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    
    for state, day, epoch, time_chunk in product(states, days, epochs, time_chunks):
        if state in spike_dict and day in spike_dict[state] and epoch in spike_dict[state][day] and time_chunk in spike_dict[state][day][epoch]:
            try:
                process_time_chunk(spike_dict = spike_dict, state = state, day = day, epoch = epoch, time_chunk = time_chunk, animal = animal, area = area, method = 'ts')
            except Exception as e:
                print(f"Error processing for state: {state}, day: {day}, epoch; {epoch}. Error: {e}")
                continue
                
            gc.collect()
            
    #progress_bar.close()
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


