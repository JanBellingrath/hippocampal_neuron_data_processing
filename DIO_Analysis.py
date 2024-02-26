#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pickle
import data_analysis_utilitites as da_utilities

def load_object_from_pickle(file_path: str):
    """
    Load and return an object from a pickle file.

    Parameters:
    - file_path (str): The path to the pickle file.

    Returns:
    - object: The Python object loaded from the file.
    """
    with open(file_path, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object

my_object = load_object_from_pickle('criticality_analysis_states/target_dav.pkl')

data = da_utilities.load_all_criticality_data_no_duplicate_files('/home/bellijjy/criticality_analysis', ['_con', '_dav', '_dud', '_Cor', '_cha'], area=None, state=None, day=None, epoch=None, time_chunk=None)


# In[ ]:


my_second_object = load_object_from_pickle('criticality_analysis_states/target_cha.pkl')


# In[ ]:


my_second_object


# In[ ]:


data


# In[5]:


def filter_non_empty_rows(df):
    """
    Filters the DataFrame to only include rows with at least one non-empty cell in columns other than the index.
    Also returns the list of index values for these rows.
    
    Parameters:
    - df (DataFrame): The input DataFrame
    
    Returns:
    - filtered_df (DataFrame): The filtered DataFrame
    - index_list (list of tuples): List of index values for non-empty rows
    """
    # Drop rows where all columns have None or NaN
    filtered_df = df.dropna(how='all')
    
    # Create a list of index values for these rows
    index_list = filtered_df.index.tolist()
    
    return filtered_df, index_list
# Apply the function
filtered_df, day_epoch_list = filter_non_empty_rows(my_object)
#filtered_df_2, day_epoch_list_2 = filter_non_empty_rows(my_second_object)


# In[11]:


filtered_df_2
def count_non_empty_pulse_times(df):
    return df['pulse_times'].apply(lambda x: x != []).sum()

num_non_empty = count_non_empty_pulse_times(filtered_df_2)
num_non_empty


# In[7]:


import pandas as pd
import numpy as np

def find_matching_rows(target_df, data_df, animal, areas):
    all_matched_records = []
    all_partial_matched_records = []
    all_exception_records = []
    
    for area in areas:
        matched_records = []
        partial_matched_records = []
        exception_records = []
        keys = {}
        
        # Filter by animal and area
        filtered_data = data_df[
            (data_df.index.get_level_values('animal') == animal) & 
            (data_df['area'] == area)
        ]
        
        # Populate min_max_keys as biggest and smallest time-indices for each time_chunk
        for (day, epoch, time_chunk), sub_row in filtered_data.groupby(['day', 'epoch', 'time_chunk']):
            original_data = eval(sub_row.iloc[0]['original_data'])  # Assuming only one row per group
            key_list = list(original_data.keys())
            min_key = float(key_list[0]) / 1000
            max_key = float(key_list[-1]) / 1000 # min and max key are the first and last time-point within each time-chunk
            print(min_key)
            print(max_key)
            keys[(day, epoch, time_chunk)] = [min_key, max_key]
        
        for index, row in target_df.iterrows():
            day, epoch = index[0], index[1]
            pulse_times_series = row.get('pulse_times', None)
            
            if isinstance(pulse_times_series, np.ndarray):
                pulse_times_series_list = (pulse_times_series / 10000).tolist()  # Convert to seconds
                
                for event in pulse_times_series_list:
                    start, stop = event[0], event[1]
                    if stop - start > 1: # if the time at the target is not at least 1 second, we throw the data away
                        time_chunks_found = []
                        
                        for (d, e, t), [min_key, max_key] in keys.items():
                            
                            if start >= min_key and stop <= max_key:
                                matched_records.append({
                                    'day': d, 'epoch': e, 'time_chunk': t,
                                    'start': start, 'stop': stop, 'area': area
                                })
                                break

                            elif (start >= min_key and start <= max_key) or (stop >= min_key and stop <= max_key):
                                time_chunks_found.append(t)

                        if len(time_chunks_found) == 2:
                            partial_matched_records.append({
                                'day': day, 'epoch': epoch,
                                'time_chunk_start': time_chunks_found[0], 'time_chunk_stop': time_chunks_found[1],
                                'start': start, 'stop': stop, 'area': area
                            })

                        elif len(time_chunks_found) > 2:
                            print(f"{len(time_chunks_found)} is a strange number for time_chunks found..something is wrong with this instance (for a small number thereof it might be expected)")


                        else:
                            closest_key_value = min(
                                keys.values(), 
                                key=lambda k: min(abs(k[0] - start), abs(k[1] - stop))
                            )
                            closest_diff = min(abs(closest_key_value[0] - start), abs(closest_key_value[1] - stop))

                            # Get the time_chunk associated with the closest_key
                            closest_time_chunk = [t for (d, e, t), [min_k, max_k] in keys.items() if [min_k, max_k] == closest_key_value][0]
                            
                            if closest_diff <1:
                                exception_records.append({
                                    'day': day, 'epoch': epoch, 'closest_time_chunk': closest_time_chunk,
                                    'closest_diff': closest_diff, 'area': area
                            })

        all_matched_records += matched_records
        all_partial_matched_records += partial_matched_records
        all_exception_records += exception_records
    
    matched_df = pd.DataFrame(all_matched_records)
    partial_matched_df = pd.DataFrame(all_partial_matched_records)
    exception_df = pd.DataFrame(all_exception_records)
    
    return matched_df, partial_matched_df, exception_df

# Note: This function needs actual data frames 'target_df' and 'data_df' to be run.
matched_df, partial_matched_df, exception_df = find_matching_rows(filtered_df, data, '_dav', ['CA1', 'CA3'])


# In[8]:


matched_df


# In[37]:


exception_df


# In[38]:


partial_matched_df


# In[15]:


def get_parameters(df, data_dict, animal_index):
    # Initialize an empty list to collect the records
    records = []

    # Retrieve the animal's data from the data_dict using its index
    animal_data = data_dict.loc[data_dict.index.get_level_values('animal') == animal_index]
    
    if animal_data.empty:
        print(f"Warning: {animal_index} not found in data_dict.")
        return pd.DataFrame()
    
    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        day = row['day']
        epoch = row['epoch']
        time_chunk = row['time_chunk']
        area = row['area']

        # Filter data_dict for matching records
        filtered_data = animal_data.query(
            'day == @day and epoch == @epoch and time_chunk == @time_chunk and area == @area'
        )

        if not filtered_data.empty:
            tau = filtered_data.iloc[0]['tau']
            branching_parameter = filtered_data.iloc[0]['branching_factor']

            records.append({
                'day': day,
                'epoch': epoch,
                'time_chunk': time_chunk,
                'area': area,
                'tau': tau,
                'branching_parameter': branching_parameter
            })
        else:
            print(f"Warning: No matching data for day={day}, epoch={epoch}, time_chunk={time_chunk}, area={area}.")
            
    # Convert the records to a DataFrame
    result_df = pd.DataFrame(records)
    
    return result_df

params = get_parameters(matched_df, data, '_dav')


# In[10]:


params


# In[16]:


def remove_duplicate_rows(df):
    """
    Remove rows with duplicate values in ['day', 'epoch', 'time_chunk', 'area']
    """
    return df.drop_duplicates(subset=['day', 'epoch', 'time_chunk', 'area'])

# Remove duplicate rows
non_duplicate_params = remove_duplicate_rows(params)
non_duplicate_params


# In[18]:


from scipy.stats import wilcoxon, skew

def compare_tau(target_dict, data_dict, animal_index):
    results = {}
    # Retrieve the animal's data from the data_dict using its index
    animal_data = data_dict.loc[data_dict.index.get_level_values('animal') == animal_index]
    
    # Find common rows between target_dict and animal_data
    common_rows = pd.merge(target_dict.reset_index(), animal_data.reset_index(), on=['day', 'epoch', 'time_chunk', 'area'])

    # Drop common rows from animal_data based on unique identifiers ('day', 'epoch', 'time_chunk', 'area')
    filtered_animal_data = animal_data.reset_index()
    filtered_animal_data = filtered_animal_data.loc[
        ~filtered_animal_data.set_index(['day', 'epoch', 'time_chunk', 'area']).index.isin(
            common_rows.set_index(['day', 'epoch', 'time_chunk', 'area']).index
        )
    ]

    # Downsample the larger dataset to match the size of the smaller dataset
    min_size = min(len(target_dict), len(filtered_animal_data))
    data_sample = filtered_animal_data.sample(min_size)
    target_sample = target_dict.sample(min_size)
    
    # Check for symmetry in the distribution of differences
    differences = target_sample['tau'].values - data_sample['tau'].values 
    skewness_value = skew(differences)
    
    # Perform the Wilcoxon Signed-Rank Test
    w_statistic, p_value = wilcoxon(data_sample['tau'].values, target_sample['tau'].values, alternative='two-sided')
    
    # Calculate the median and mean differences
    median_difference = np.median(differences)
    mean_difference = np.mean(differences)
    
    # Populate the results dictionary
    results['skewness_value'] = skewness_value
    results['median_difference'] = median_difference
    results['mean_difference'] = mean_difference
    results['w_statistic'] = w_statistic
    results['p_value'] = p_value
    
    return results

# Test the function with your data
compare_tau(non_duplicate_params, data, '_dav')


# In[19]:


# Defining a bootstrapping function that repeatedly calls the `compare_tau` function
def bootstrap_compare_tau(target_dict, data_dict, animal_index, num_iterations=1000):
    p_values = []
    w_statistics = []
    median_differences = []
    mean_differences = []
    skewness_values = []
    
    for i in range(num_iterations):
        # Calling the original `compare_tau` function
        results = compare_tau(target_dict, data_dict, animal_index)
        
        # Storing the metrics
        p_values.append(results['p_value'])
        w_statistics.append(results['w_statistic'])
        median_differences.append(results['median_difference'])
        mean_differences.append(results['mean_difference'])
        skewness_values.append(results['skewness_value'])
    
    # Calculating the 95% confidence intervals
    ci_95_p_value = np.percentile(p_values, [2.5, 97.5])
    ci_95_w_statistic = np.percentile(w_statistics, [2.5, 97.5])
    ci_95_median_difference = np.percentile(median_differences, [2.5, 97.5])
    ci_95_mean_difference = np.percentile(mean_differences, [2.5, 97.5])
    ci_95_skewness = np.percentile(skewness_values, [2.5, 97.5])
    
    # Consolidating the results
    results_summary = {
        'p_value': {'mean': np.mean(p_values), '95% CI': ci_95_p_value},
        'w_statistic': {'mean': np.mean(w_statistics), '95% CI': ci_95_w_statistic},
        'median_difference': {'mean': np.mean(median_differences), '95% CI': ci_95_median_difference},
        'mean_difference': {'mean': np.mean(mean_differences), '95% CI': ci_95_mean_difference},
        'skewness': {'mean': np.mean(skewness_values), '95% CI': ci_95_skewness}
    }
    
    return results_summary

# Note: The original `compare_tau` function would need to be modified to return a dictionary of results instead of printing them.
# You would then call `bootstrap_compare_tau` like this:
bootstrap_results = bootstrap_compare_tau(non_duplicate_params, data, '_dav', num_iterations=10000)


# In[20]:


bootstrap_results


# In[ ]:




