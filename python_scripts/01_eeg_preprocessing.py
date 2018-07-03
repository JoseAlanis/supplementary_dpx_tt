# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
os.chdir('/Users/Josealanis/Documents/Experiments/dpx_tt/eeg/')

#import numpy as np
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs

#%% READ IN THE DATA

# EEG montage
montage = mne.channels.read_montage(kind='biosemi64')

#cd '~/Documents/Experiments/dpx_tt/eeg/bdfs/'
data_path = './bdfs/data2.bdf'

# Import raw data
raw = mne.io.read_raw_edf(data_path, 
                          montage = montage, 
                          preload = True, 
                          stim_channel = -1, 
                          exclude = ['EOGH_rechts', 'EOGH_links', 
                                     'EXG5', 'EXG6', 'EXG7', 'EXG8'])

# Get data information
raw.info

#%% EDIT INFORMATION

# Note the samlping rate of your recording
sfreq = raw.info['sfreq']
sfreq = int(sfreq)
# and Buffer size ???
bsize = raw.info['buffer_size_sec']

# Channel names
chans = raw.info['ch_names'][0:64]
chans.extend(['EXG1', 'EXG2', 'Stim'])


# Write a list of channel types (e.g., eeg, eog, ecg)
chan_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
              'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
              'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
              'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
              'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
              'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
              'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
              'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
              'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
              'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
              'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
              'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
              'eeg', 'eeg', 'eeg', 'eeg',
              'eog', 'eog', 'stim']



# Bring it all together with MNE.function for creating custom EEG info files
info_custom = mne.create_info(chans, sfreq,  chan_types, montage)

# You also my add a short description of the data set
info_custom['description'] = 'DPX Baseline'

# Replace the mne info structure with the customized one 
# which has the correct labels, channel types and positions.
raw.info = info_custom
raw.info['buffer_size_sec'] = bsize

# check data information 
raw.info

#%% GET EVENT INFORMATION
# Next, define the type of data you have provided in 'picks'
picks = mne.pick_types(raw.info, 
                       meg = False, 
                       eeg = True, 
                       eog = True,
                       stim = True)


# EVENTS
events = mne.find_events(raw, 
                         stim_channel = 'Stim', 
                         output = 'onset', 
                         min_duration = 0.002)

# EVENTS THAT ARE CUE STIMULI
evs = events[(events[:,2] >= 70) & (events[:,2] <= 75), ]
# LATENCIES
latencies = events[(events[:,2] >= 70) & (events[:,2] <= 75), 0]
# DIFFERENCE BETWEEN TWO CONSEQUITIVE CUES
diffs = [x-y for x, y in zip(latencies, latencies[1:])]
# GET FIRST EVENT AFTER BREAKS (pauses between blocks,
# time difference between two events is > 10 seconds)
diffs  = [abs(number)/sfreq for number in diffs]
breaks = [i+1 for i in range(len(diffs)) if diffs[i] > 10]

# start first block
b1s = ( latencies[ breaks[0] ] - (2*sfreq) ) / sfreq
# end of frist block
b1e = ( latencies[ (breaks[1]-1) ] + (6*sfreq) ) / sfreq

# start second block
b2s = ( latencies[ breaks[1] ] - (2*sfreq) ) / sfreq
# end of frist block
b2e = ( latencies[ (breaks[2]-1) ] + (6*sfreq) ) / sfreq


#%%
# Block 1
raw_bl1 = raw.copy().crop(tmin = b1s, tmax = b1e)
# Block 2
raw_bl2 = raw.copy().crop(tmin = b2s, tmax = b2e)

# Bind them together
raw_blocks = mne.concatenate_raws([raw_bl1, raw_bl2])
#%%


keeps = mne.find_events(raw_blocks, stim_channel='Stim', output='onset', 
                         min_duration=0.002)

len(keeps[(keeps[:,2] >= 70) & (keeps[:,2] <= 75), ])


#%%

#eog_events = mne.preprocessing.find_eog_events(raw_blocks)
#n_blinks = len(eog_events)
#
#raw_blocks.get_data(picks=picks, start=0, stop=None, return_times=False)
#
#onset = eog_events[:, 0] / raw_blocks.info['sfreq'] - 0.25
#duration = np.repeat(0.5, n_blinks)
#raw_blocks.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks, orig_time=raw_blocks.info['meas_date'])
#print(raw_blocks.annotations)  # to get information about what annotations we have
#raw_blocks.plot(events=eog_events, scalings = dict(eeg = 5e-5), n_channels = 66)  # To see the annotated segments.



#%%
#raw_blocks.save('./raw_blocks-raw.fif', 
#                buffer_size_sec = 1., 
#                overwrite = True)

#%%

# Filter and rereference the data to reduce noise and remove artefact frequencies.
raw_blocks.filter(0.1, 50., n_jobs=1, fir_design='firwin') 
raw_blocks.set_eeg_reference(ref_channels='average', projection=False) 

#%%
# Next, define the type of data you have provided in 'picks'
picks = mne.pick_types(raw_blocks.info, 
                       meg = False, 
                       eeg = True, 
                       eog = False,
                       stim = False)
n_components = 25
method = 'extended-infomax'
#decim = 3
reject = None
ica = ICA(n_components=n_components, method=method)
ica.fit(raw_blocks.copy().filter(1,50), picks=picks, reject = dict(eeg = 3e-4))
#ica.fit(raw.copy().filter(1,50), picks=picks, reject=reject) 

#%%
ica.plot_components()

ica.plot_properties(raw_blocks, picks=[0, 1], psd_args={'fmax': 50.})

#%%
ica.plot_properties(raw_blocks, picks=[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], psd_args={'fmax': 45.})

#%%
eog_average = create_eog_epochs(raw_blocks, reject = dict(eeg = 3e-4),
                                picks = picks).average()

eog_epochs = create_eog_epochs(raw_blocks, reject = dict(eeg = 3e-4))  # get single EOG trials
eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation

ica.plot_scores(scores, exclude = eog_inds)  # look at r scores of components
# we can see that only one component is highly correlated and that this
# component got detected by our correlation analysis (red).

ica.plot_sources(eog_average, exclude = eog_inds)  # look at source time course


#%%
ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 35.},
                    image_args={'sigma': 1.})


raw_blocks.plot(scalings = dict(eeg = 5e-5), n_channels = 66, events = events)


#%%
ica.apply(raw_blocks, exclude=[0, 1, 10, 11, 13, 14])

#%%

raw_blocks.info