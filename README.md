# Hippocampal_Neuron_Data_Processing

Data-Processing pipeline for neuronal time series data.

Extensive adoption of, and slight modifications on, the data-processing pipeline developed by the Loren Frank Lab:
https://github.com/Eden-Kramer-Lab/loren_frank_data_processing/tree/v0.9.13.dev0/loren_frank_data_processing

At the moment, this repository contains:
1. A file adopting (partially modified) functions of the Loren_Frank_Dataprocessing Pipeline
2. A file, taking the processed time-series data from (1.) as input, and estimating the neuronal branching factor (criticality-analysis). The determination
   of the branching factor is done via the "MR. Estimator" (https://doi.org/10.1371/journal.pone.0249447; https://mrestimator.readthedocs.io/en/latest/doc/gettingstarted.html)
   toolbox developed by Spitzner et. al (2021).
   
