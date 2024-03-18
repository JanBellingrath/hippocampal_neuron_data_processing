#!/usr/bin/env python
# coding: utf-8

# In[28]:




from collections import namedtuple
import pandas as pd
import numpy as np
from os.path import join
from scipy.io import loadmat


# In[29]:


Animal = namedtuple('Animal', {'short_name', 'directory'})

conley = Animal('/home/bellijjy/Conley.tar/Conley', 'con')
dave = Animal('/home/bellijjy/Dave.tar/Dave/Dave', 'dav')
chapati = Animal('/home/bellijjy/Chapati.tar/Chapati/Chapati', 'cha')
#frank = Animal('/home/bellijjy/Frank.tar/Frank', 'fra')
#egypt = Animal('/home/bellijjy/Egypt.tar/Egypt', 'egy')
#remi = Animal('/home/bellijjy/Remi.tar/Remi', 'rem')
corriander = Animal('/home/bellijjy/Corriander.tar/Corriander', 'Cor')
#dudley = Animal('/home/bellijjy/Dudley.tar/Dudley', 'dud')
#bond = Animal('/home/bellijjy/Bond', 'bon')
#government = Animal('/home/bellijjy/Government.tar/Government', 'gov')
five = Animal("/home/dekorvyb/Downloads/Fiv", "Fiv")
bon = Animal("/home/dekorvyb/Downloads/Bon", "bon")


animals = {'con': Animal('con','/home/bellijjy/Conley.tar/Conley'),
           'Cor': Animal('Cor','/home/bellijjy/Corriander.tar/Corriander'),
            #'fra': Animal('fra','/home/bellijjy/Frank.tar/Frank/fra'),
            'cha': Animal('cha','/home/bellijjy/Chapati.tar/Chapati/Chapati'),
          #'egy': Animal('egy','/home/bellijjy/Egypt.tar/Egypt/egy'),
          'rem': Animal('rem','/home/bellijjy/Remi.tar/Remi/rem'),
          'dav': Animal('dav','/home/bellijjy/Dave.tar/Dave/Dave'),
           'dud': Animal('dud','/home/bellijjy/Dudley.tar/Dudley/dud'),
        # 'gov' : Animal('gov','/home/bellijjy/Government.tar/Government/gov'),
        #'bon' : Animal('bon', '/home/bellijjy/Bond'),
           "Fiv" : Animal("Fiv", "/home/dekorvyb/Downloads/Fiv"),
          "bon" : Animal("bon", "/home/dekorvyb/Downloads/Bon/bon")}
          #} 


# In[30]:


EDGE_ORDER = [0, 2, 4, 1, 3]
EDGE_SPACING = [15, 0, 15, 0]

def get_data_filename_modified(animal, day, file_type):
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
    filename = '{animal}{file_type}{day:02d}.mat'.format(
        animal=animal.directory,
        file_type=file_type,
        day=day)
    
    return join(animal.short_name, filename)

def get_data_structure_modified(animal, day, file_type, variable):
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
        file = loadmat(get_data_filename_modified(animal, day, file_type))
        n_epochs = file[variable][0, -1].size
        return [file[variable][0, -1][0, ind]
                for ind in np.arange(n_epochs)]
    except (IOError, TypeError):
        logger.warn('Failed to load file: {0}'.format(
            get_data_filename_modified(animal, day, file_type)))
        return None


def _get_pos_dataframe_modified(epoch_key, animals):
    animal, day, epoch = epoch_key
    struct = get_data_structure_modified(animals[animal.short_name], day, 'pos', 'pos')[epoch - 1]
    position_data = struct['data'][0, 0]
    FIELD_NAMES = ['time', 'x_position', 'y_position', 'head_direction',
                   'speed', 'smoothed_x_position', 'smoothed_y_position',
                   'smoothed_head_direction', 'smoothed_speed']
    time = pd.TimedeltaIndex(
        position_data[:, 0], unit='s', name='time')
    n_cols = position_data.shape[1]

    if n_cols > 5:
        # Use the smoothed data if available
        NEW_NAMES = {'smoothed_x_position': 'x_position',
                     'smoothed_y_position': 'y_position',
                     'smoothed_head_direction': 'head_direction',
                     'smoothed_speed': 'speed'}
        return (pd.DataFrame(
            position_data[:, 5:], columns=FIELD_NAMES[5:], index=time)
            .rename(columns=NEW_NAMES))
    else:
        return pd.DataFrame(position_data[:, 1:5], columns=FIELD_NAMES[1:5],
                            index=time)

def get_position_dataframe_modified(epoch_key, animals, use_hmm=True,
                           max_distance_from_well=5,
                           route_euclidean_distance_scaling=1,
                           min_distance_traveled=50,
                           sensor_std_dev=5,
                           diagonal_bias=1E-1,
                           edge_spacing=EDGE_SPACING,
                           edge_order=EDGE_ORDER,
                           skip_linearization=False):
    '''Returns a list of position dataframes with a length corresponding
     to the number of epochs in the epoch key -- either a tuple or a
    list of tuples with the format (animal, day, epoch_number)

    Parameters
    ----------
    epoch_key : tuple
        Unique key identifying a recording epoch. Elements are
        (animal, day, epoch)
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    position : pandas dataframe
        Contains information about the animal's position, head direction,
        and speed.

    '''
    position_df = _get_pos_dataframe_modified(epoch_key, animals)
    if not skip_linearization:
        if use_hmm:
            position_df = _get_linear_position_hmm(
                epoch_key, animals, position_df,
                max_distance_from_well, route_euclidean_distance_scaling,
                min_distance_traveled, sensor_std_dev, diagonal_bias,
                edge_order=edge_order, edge_spacing=edge_spacing)
        else:
            linear_position_df = _get_linpos_dataframe(
                epoch_key, animals, edge_order=edge_order,
                edge_spacing=edge_spacing)
            position_df = position_df.join(linear_position_df)

    return position_df

def get_track_segments_modified(epoch_key, animals):
    '''

    Parameters
    ----------
    epoch_key : tuple
    animals : dict of namedtuples

    Returns
    -------
    track_segments : ndarray, shape (n_segments, n_nodes, n_space)
    center_well_position : ndarray, shape (n_space,)

    '''
    animal, day, epoch = epoch_key
    task_file = get_data_structure(animals[animal], day, 'task', 'task')
    print(task_file)
    linearcoord = task_file[epoch - 1]['linearcoord'][0, 0].squeeze(axis=0)
    track_segments = [np.stack(((arm[:-1, :, 0], arm[1:, :, 0])), axis=1)
                      for arm in linearcoord]
    center_well_position = track_segments[0][0][0]
    track_segments = np.concatenate(track_segments)
    _, unique_ind = np.unique(track_segments, return_index=True, axis=0)
    return track_segments[np.sort(unique_ind)], center_well_position

def make_track_graph_modified(epoch_key, animals):
    '''

    Parameters
    ----------
    epoch_key : tuple, (animal, day, epoch)
    animals : dict of namedtuples

    Returns
    -------
    track_graph : networkx Graph
    center_well_id : int

    '''
    track_segments, center_well_position = get_track_segments_modified(
        epoch_key, animals)
    nodes = track_segments.copy().reshape((-1, 2))
    _, unique_ind = np.unique(nodes, return_index=True, axis=0)
    nodes = nodes[np.sort(unique_ind)]

    edges = np.zeros(track_segments.shape[:2], dtype=np.int)
    for node_id, node in enumerate(nodes):
        edge_ind = np.nonzero(np.isin(track_segments, node).sum(axis=2) > 1)
        edges[edge_ind] = node_id

    edge_distances = np.linalg.norm(
        np.diff(track_segments, axis=-2).squeeze(axis=-2), axis=1)

    track_graph = nx.Graph()

    for node_id, node_position in enumerate(nodes):
        track_graph.add_node(node_id, pos=tuple(node_position))

    for edge, distance in zip(edges, edge_distances):
        nx.add_path(track_graph, edge, distance=distance)

    center_well_id = np.unique(
        np.nonzero(np.isin(nodes, center_well_position).sum(axis=1) > 1)[0])[0]

    return track_graph, center_well_id

def _get_linear_position_hmm(epoch_key, animals, position_df,
                             max_distance_from_well=5,
                             route_euclidean_distance_scaling=1,
                             min_distance_traveled=50,
                             sensor_std_dev=5,
                             diagonal_bias=1E-1,
                             edge_order=EDGE_ORDER, edge_spacing=EDGE_SPACING,
                             position_sampling_frequency=33):
    animal, day, epoch = epoch_key
    track_graph, center_well_id = make_track_graph_modified(epoch_key, animals)
    position = position_df.loc[:, ['x_position', 'y_position']].values
    track_segment_id = classify_track_segments(
        track_graph, position,
        route_euclidean_distance_scaling=route_euclidean_distance_scaling,
        sensor_std_dev=sensor_std_dev,
        diagonal_bias=diagonal_bias)
    (position_df['linear_distance'],
     position_df['projected_x_position'],
     position_df['projected_y_position']) = calculate_linear_distance(
        track_graph, track_segment_id, center_well_id, position)
    position_df['track_segment_id'] = track_segment_id
    SEGMENT_ID_TO_ARM_NAME = {0.0: 'Center Arm',
                              1.0: 'Left Arm',
                              2.0: 'Right Arm',
                              3.0: 'Left Arm',
                              4.0: 'Right Arm'}
    position_df = position_df.assign(
        arm_name=lambda df: df.track_segment_id.map(SEGMENT_ID_TO_ARM_NAME)
    )

    segments_df, labeled_segments = get_segments_df(
        epoch_key, animals, position_df, max_distance_from_well,
        min_distance_traveled)

    segments_df = pd.merge(
        labeled_segments, segments_df, right_index=True,
        left_on='labeled_segments', how='outer')
    position_df = pd.concat((position_df, segments_df), axis=1)
    position_df['linear_position'] = _calulcate_linear_position(
        position_df.linear_distance.values,
        position_df.track_segment_id.values, track_graph, center_well_id,
        edge_order=edge_order, edge_spacing=edge_spacing)
    position_df['linear_velocity'] = calculate_linear_velocity(
        position_df.linear_distance, smooth_duration=0.500,
        sampling_frequency=position_sampling_frequency)
    position_df['linear_speed'] = np.abs(position_df.linear_velocity)
    position_df['is_correct'] = position_df.is_correct.fillna(False)

    return position_df
 


# In[ ]:


#motor periods by 2 s buffer intervals (preceding and following) and excluding SWR periods. Thus brief interruptions in locomotion did not qualify as formally detected periods of immobility.

#I may have to add this above. otherwise the data get taken from running to resting too soon, in incoherence with the paper


# In[32]:


head_direction_series = _get_pos_dataframe_modified((conley,1,1), animals)['head_direction']
#The head direction can be positive or negative, indicating left or right direction

def head_speed_from_head_direction_dict(head_direction_series):
    '''
    Returns: Dict with Head Direction, Time Delta and Head Speed
    Input: Head Direction Series
    '''
    # Create a DataFrame with the time deltas and head directions
    df = pd.DataFrame({'Time Delta': head_direction_series.index, 'Head Direction': head_direction_series.values})
    
    # Convert the time deltas to seconds for easier calculations
    df['Time Delta'] = df['Time Delta'].dt.total_seconds()
    
    # Calculate the head speed (rate of change over time)
    df['Head Speed'] = df['Head Direction'].diff() / df['Time Delta'].diff()
    
    return df
    
def head_speed_series_from_head_direction(head_direction_series, head_speed_dict):
    head_speed_series = pd.Series(head_speed_dict['Head Speed'].values, index=head_direction_series.index)
    return head_speed_series


head_speed_over_time = head_speed_series_from_head_direction(head_direction_series, head_speed_from_head_direction_dict(head_direction_series))
