# Hippocampal_Neuron_Data_Processing

Data-Processing pipeline for neuronal time series data.

Extensive adoption, as well as slight modifications, of the data-processing pipeline developed by the Loren Frank Lab:
https://github.com/Eden-Kramer-Lab/loren_frank_data_processing/tree/v0.9.13.dev0/loren_frank_data_processing

At the moment, this repository contains (work-in-progress):
1. main (criticality-analysis (for animal Corriander) that calls the following modules:
2. subdivision by theme - divides epochs by task, etc.
3. get speed - gets speed position, head-speed etc.
4. lfp data to dataframe
5. spike data to dataframe
6. task information
7. general utilities
   
The resulting output of the data-processing pipeline is intended for estimating the neuronal branching factor (criticality-analysis; https://doi.org/10.1371/journal.pone.0249447; https://mrestimator.readthedocs.io/en/latest/doc/gettingstarted.html; toolbox developed by Spitzner et. al (2021)).
   
