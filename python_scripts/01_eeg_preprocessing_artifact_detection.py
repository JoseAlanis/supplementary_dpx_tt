# --- Jose C. Garcia Alanis
# --- utf-8
# --- Python 3.6.2
#
# --- EEG prepossessing - DPX TT
# --- Artifact detection, interpolate bad channels, extract block data
# --- Version Jul 2018

# ==================================================================================================
# --------------------------------- Import relevant extensions -------------------------------------

import os
import pandas as pd
import mne
# from mne.preprocessing import ICA
# from mne.preprocessing import create_eog_epochs


# --- 1) READ IN THE DATA ---------------------------------
# Set working directory
os.chdir('/Users/Josealanis/Documents/Experiments/dpx_tt/eeg/')

# EEG set up and file path
montage = mne.channels.read_montage(kind='biosemi64')
data_path = './dpx_tt_bdfs/data3.bdf'

# Import raw data
raw = mne.io.read_raw_edf(data_path,
                          montage=montage,
                          preload=True,
                          stim_channel=-1,
                          exclude=['EOGH_rechts', 'EOGH_links',
                                   'EXG3', 'EXG4', 'EXG5', 'EXG6',
                                   'EXG7', 'EXG8'])

# Check data information
print(raw.info)


# --- 2) EDIT DATA SET INFORMATION ------------------------
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
print(raw.info)


# --- 3) GET EVENT INFORMATION ----------------------------
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


# --- 4) GET EVENTS THAT REPRESENT CUE STIMULI ------------
# Cue events
evs = events[(events[:, 2] >= 70) & (events[:, 2] <= 75), ]
print('There are', len(evs), 'events.')


# --- 5) GET EVENT LATENCIES ------------------------------
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


# --- 5) SAVE START AND END OF BLOCKS ---------------------
# start first block
b1s = (latencies[breaks[0]] - (2 * sfreq)) / sfreq
# end of first block
b1e = (latencies[(breaks[1] - 1)] + (6 * sfreq)) / sfreq

# start second block
b2s = (latencies[breaks[1]] - (2 * sfreq)) / sfreq
# end of first block
if len(breaks) > 2:
    b2e = (latencies[(breaks[2] - 1)] + (6 * sfreq)) / sfreq
else:
    b2e = (latencies[-1] + (6 * sfreq)) / sfreq

# Block durations
print('Block 1 from', round(b1s, 3), 'to', round(b1e, 3), '    Block length ', round(b1e - b1s, 3))
print('Block 2 from', round(b2s, 3), 'to', round(b2e, 3), '    Block length ', round(b2e - b2s, 3))


# --------------------------------------------------------------------------------------------------
# ============================ RERUN FROM HERE IF BAD CHANNELS FOUND ===============================

# --- 6) EXTRACT BLOCK DATA --------------------------
# Block 1
raw_bl1 = raw.copy().crop(tmin=b1s, tmax=b1e)
# Block 2
raw_bl2 = raw.copy().crop(tmin=b2s, tmax=b2e)


# --- 7) CONCATENATE DATA ----------------------------
# Concatenate block data
raw_blocks = mne.concatenate_raws([raw_bl1, raw_bl2])

# Find events in the concatenated data set
evs_blocks = mne.find_events(raw_blocks,
                             stim_channel='Stim',
                             output='onset',
                             min_duration=0.002)

# Number of events in B1 and B2
print(len(evs_blocks[(evs_blocks[:, 2] >= 70) &
                     (evs_blocks[:, 2] <= 75), 0]), 'events found.')

# --- 8) REJECT BAD CHANNELS IF FOUND --------------------- UNCOMMENT SECTION
# # Mark as bad
# raw_blocks.info['bads'] = ['F5']
# # Interpolate bads
# raw_blocks.interpolate_bads(reset_bads=True,
#                             verbose=False,
#                             mode='accurate')


# --- 9) APPLY FILTER TO DATA -----------------------------
raw_blocks.filter(0.1, 50, fir_design='firwin')


# --- 10) FIND DISTORTED SEGMENTS IN DATA -----------------
# Copy of data
x = raw_blocks.get_data()

# Detect artifacts (i.e., absolute amplitude > 500 microV)
times = []
for j in range(0, len(x[0])):
    if len(times) > 0:
        if j <= (times[-1] + int(2*sfreq)):
            continue
    t = []
    for i in list(range(0, 64)):
        t.append(abs(x[i][j]))
    if max(t) >= 5e-4:
        times.append(float(j))
# If artifact found create annotations for raw data
if len(times) > 0:
    # Save onsets
    annotations_df = pd.DataFrame(times)
    annotations_df.columns = ['Onsets']
    # Include one second before artifact onset
    onsets = (annotations_df['Onsets'].values / sfreq) - 1
    # Merge with previous annotations
    duration = [2] * len(onsets) + list(raw_blocks.annotations.duration)
    labels = ['Bad'] * len(onsets) + list(raw_blocks.annotations.description)
    onsets = list(onsets)
    # Append onsets of previous annotations
    for i in range(0, len(list(raw_blocks.annotations.onset))):
        onsets.append(list(raw_blocks.annotations.onset)[i])
    # Create new annotation info
    annotations = mne.Annotations(onsets, duration, labels)
    raw_blocks.annotations = annotations


# --- 11) CHECK FOR INCONSISTENCIES -----------------------
# Threshold for plotting
clip = None
# Plot
raw_blocks.plot(n_channels=66,
                scalings=dict(eeg=100e-6),
                events=evs_blocks,
                bad_color='red',
                clipping=clip)


# --------------------------------------------------------------------------------------------------
# ========================= EVERYTHING OK? --- CONTINUE PREPROCESSING ==============================

# --- 12) RE-REFERENCE TO AVERAGE OF 64 ELECTRODES  -------
raw_blocks.set_eeg_reference(ref_channels='average',
                             projection=False)


# --- 13) SAVE RAW FILE -----------------------------------
# Pick electrode to use
picks = mne.pick_types(raw_blocks.info,
                       meg=False,
                       eeg=True,
                       eog=True,
                       stim=True)

# Check info
print(raw_blocks.info)

# Check data
raw_blocks.plot(n_channels=66,
                scalings=dict(eeg=100e-6),
                events=evs_blocks,
                bad_color='red',
                clipping=clip)

# Save segmented data
raw_blocks.save('./dpx_tt_mne_raws/data3-raw.fif',
                picks=picks,
                overwrite=True)
