#!/usr/bin/env python
# coding: utf-8

# # This is the analysis pipeline for the calculation of tau and branching factor of neuronal data from the hippocampus. 
# 
# The data are take from: https://datadryad.org/stash/dataset/doi:10.7272/Q61N7ZC3

# ### Importing the branching factor/tau estimator
# 
# For documentation regards the estimator see https://mrestimator.readthedocs.io/en/latest/doc/gettingstarted.html, see also Spitzner, F. P., Dehning, J., Wilting, J., Hagemann, A., P. Neto, J., Zierenberg, J., & Priesemann, V. (2021).

# In[ ]:


import mrestimator as mre


# ### Importing relevant non-standard modules
# All of which -- as well as additional submodules which are accessed from the modules imported here -- can be found on GitHub https://github.com/JanBellingrath/Hippocampal_Neuron_Data_Processing

# In[ ]:


import utilities as u
import criticality_analysis as can
import compactifying_functions as compact


# ### Defining each animal via its short name and its directory

# In[ ]:


from collections import namedtuple

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


# ### Getting the neuron ids for a specific animal and a specific area
# The input is the subarea of interest (for example, 'CA1') and the short_name of the animal (for example, 'con').
# This function returns the neuron ids for each day/epoch combination, relative to the behavioral state of the epoch (awake/asleep). These neuron ids will be used to retrieve the data. Sometimes the function throws an error (FileNotFoundError: [Errno 2] No such file or directory: '/home/bellijjy/Conley.tar/Conleycellinfo.mat'), this solves itself with (usually not more than one or two, sometimes more) restarts of the kernel.

# In[ ]:


neuron_ids = compact.neuron_ids_for_specific_animal_and_subarea('CA1', 'con')


# ### Getting the neuronal data per state/day/epoch/time_chunk as values with the spike time as index.
# Takes the neuron ids from above and gets the data, subdividing the data into chunks of n seconds. 

# In[ ]:


splitted_by_sec_spike_dict = compact.get_spike_data(neuron_ids, 30, 'CA1', 'con')


# ### Running the estimator on the data, to get the branching parameter and tau values, relative to the defined time_chunks
# Takes the behav_state/day/epoch/time_chunk dict generated above, as well as parameters relevant for the estimator. For the documentation of the estimator and detailed explanations see https://mrestimator.readthedocs.io/en/latest/doc/gettingstarted.html.
# As (with a second interval of 30s) about 1000 (depending on animal) indiviudal points are estimated, the function may take a couple of minutes to run. The 'dt' parameter refers to the temporal resolution of the recording.

# In[ ]:


output_handler_dict = can.run_analysis(splitted_by_sec_spike_dict, numboot=0, coefficientmethod='ts', dt = 0.000666, dtunit = 's', targetdir='./output', title='My Analysis', tmin=0, tmax=8000, fitfuncs=['complex'])


# ### Saving the data into a file 
# Store the data in a file with a given name (such as con_CA1). Optimal input is directory, default value is current working directory.

# In[ ]:


u.store_data(output_handler_dict, cha_CA1)

