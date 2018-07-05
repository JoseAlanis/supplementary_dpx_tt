# header

# utf

# contact

# env

# =============================================================

# --- Import relevant extensions ---
import os
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs

# --- 1) READ IN THE DATA ----------------------------
# Set working directory
os.chdir('/Users/Josealanis/Documents/Experiments/dpx_tt/eeg/')

# EEG set up and file path
montage = mne.channels.read_montage(kind='biosemi64')
data_path = './dpx_tt_bdfs/data2.bdf'

# Import raw data
raw = mne.io.read_raw_edf(data_path,
                          montage=montage,
                          preload=True,
                          stim_channel=-1,
                          exclude=['EOGH_rechts', 'EOGH_links',
                                   'EXG3', 'EXG4', 'EXG5', 'EXG6',
                                   'EXG7', 'EXG8'])

# Check data information
raw.info

# --- 2) EDIT DATA SET INFORMATION -------------------
# Note the sampling rate of recording
sfreq = raw.info['sfreq']
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

# Bring it all together with
# MNE.function for creating custom EEG info files
info_custom = mne.create_info(chans, sfreq, chan_types, montage)

# You also my add a short description of the data set
info_custom['description'] = 'Iowa Gambling Task'

# Replace the mne info structure with the customized one
# which has the correct labels, channel types and positions.
raw.info = info_custom
raw.info['buffer_size_sec'] = bsize

# check data information
raw.info

# --- 3) GET EVENT INFORMATION -----------------------
# Next, define the type of data you want to work with
picks = mne.pick_types(raw.info,
                       meg=False,
                       eeg=True,
                       eog=True,
                       stim=True)

# Get events
events = mne.find_events(raw,
                         stim_channel='Stim',
                         output='onset',
                         min_duration=0.002)

# --- 4) GET EVENTS THAT REPRESENT CUE STIMULI -------
# Cue events
evs = events[(events[:, 2] >= 70) & (events[:, 2] <= 75),]
print('There are', len(evs), 'events.')

# --- 5) GET EVENT LATENCIES -------------------------
# Latency of cues
latencies = events[(events[:, 2] >= 70) & (events[:, 2] <= 75), 0]
print('Got', len(latencies), 'latencies.')

# Difference between two consequtive cues
diffs = [x - y for x, y in zip(latencies, latencies[1:])]

# Get first event after a long break (i.e., pauses between blocks),
# Time difference in between blocks should be  > 10 seconds)
diffs = [abs(number) / sfreq for number in diffs]
breaks = [i + 1 for i in range(len(diffs)) if diffs[i] > 10]
print('Identified breaks at positions', breaks)

# --- 5) SAVE START AND END OF BLOCKS ----------------
# start first block
b1s = (latencies[breaks[0]] - (2 * sfreq)) / sfreq
# end of first block
b1e = (latencies[(breaks[1] - 1)] + (6 * sfreq)) / sfreq

# start second block
b2s = (latencies[breaks[1]] - (2 * sfreq)) / sfreq
# end of first block
b2e = (latencies[(breaks[2] - 1)] + (6 * sfreq)) / sfreq

# Block durations
print('Block 1 from', round(b1s, 3), 'to', round(b1e, 3), '    Block length ', round(b1e - b1s, 3))
print('Block 2 from', round(b2s, 3), 'to', round(b2e, 3), '    Block length ', round(b2e - b2s, 3))

# --- 6) EXTRACT BLOCK DATA --------------------------
# Block 1
raw_bl1 = raw.copy().crop(tmin=b1s, tmax=b1e)
# Block 2
raw_bl2 = raw.copy().crop(tmin=b2s, tmax=b2e)

# --- 7) CONCATENATE DATA ----------------------------
# Concatenate block data
raw_blocks = mne.concatenate_raws([raw_bl1, raw_bl2])

# Find events in the concatenated dataset
evs_blocks = mne.find_events(raw_blocks,
                             stim_channel='Stim',
                             output='onset',
                             min_duration=0.002)

# --- 8) CHECK FOR INCONSISTENCIES ------------------------
# Run this part in terminal
raw_blocks.plot(n_channels=66,
                scalings=dict(eeg=1e-4),
                events=evs_blocks)

# --- 9) FILTER THE DATA ----------------------------------
raw.filter(0.1, 50, fir_design='firwin')

# --- 11) RE-REFERENCE TO AVERAGE OF 64 ELECTRODES  -------
raw.set_eeg_reference(ref_channels='average',
                      projection=False)


# --- 12) ICA DECOMPOSITION -------------------------------
n_components = 25
method = 'extended-infomax'
decim = None
reject = None

# Pick electrodes to use
picks = mne.pick_types(raw.info,
                       meg=False,
                       eeg=True,
                       eog=False,
                       stim=False)

# ICA parameters
ica = ICA(n_components=n_components,
          method=method)
# Fit ICA
ica.fit(raw.copy().filter(1, 50),
        picks=picks,
        reject=reject)

# Plot components
ica.plot_components()
