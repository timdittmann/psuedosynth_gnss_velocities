# pseudo (they are real seismic waveforms combined with noise generated from real GNSS noise distributions) 
# synthetic (they aren't actually observed timeseries)
# GNSS velocities (see SNIVEL)

## This respository presents the analysis used to: 
1. characterize real world 5Hz GNSS velocity noise using probabalistic power spectral density estimation
2. From these noise distributions generate synthetic GNSS velocity noise
3. Superpose these synthetic noise timeseries on transferred strong motion waveforms
4. Train a random forest classifier to seperate signal from noise using exclusively this dataset
5. Validate against a real world GNSS strong motion dataset


## Getting started: 
This analysis can be run using a series of notebooks.  These drive python scripts in the `/bin` directory.

