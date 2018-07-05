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
                scalings=dict(eeg=5e-5),
                events=evs_blocks)

# --- 9) FILTER THE DATA ----------------------------------
raw_blocks.filter(0.1, 50, fir_design='firwin')

# --- 11) RE-REFERENCE TO AVERAGE OF 64 ELECTRODES  -------
raw_blocks.set_eeg_reference(ref_channels='average',
                      projection=False)


# --- 12) ICA DECOMPOSITION -------------------------------
n_components = 25
method = 'extended-infomax'
decim = None
reject = dict(eeg=3e-4)

# Pick electrodes to use
picks = mne.pick_types(raw_blocks.info,
                       meg=False,
                       eeg=True,
                       eog=False,
                       stim=False)

# ICA parameters
ica = ICA(n_components=n_components,
          method=method)
# Fit ICA
ica.fit(raw_blocks.copy().filter(1, 50),
        picks=picks,
        reject=reject)

# Plot components
ica.plot_components()


# # --- 13 ADVANCED ARTIFACT REJECTION ----------------------
# # Create EOG epochs
# eog_average = create_eog_epochs(raw_blocks,
#                                 reject=reject,
#                                 picks=picks).average()
# # Get single EOG trials
# eog_epochs = create_eog_epochs(raw_blocks,
#                                reject=reject)
#
# # Find via correlation
# eog_inds, scores = ica.find_bads_eog(eog_epochs)
#
# # look at r scores of components
# ica.plot_scores(scores, exclude=eog_inds)
#
# # We can see that only one component is highly correlated and that this
# # component got detected by our correlation analysis (red).
# ica.plot_sources(eog_average,
#                  exclude=eog_inds)

new_events = evs_blocks.copy()
temp_cue = 0
valid = True


for i in range(new_events[:, 2].size):
    # FIRST STEP: temp_cue == 0 indicates we are looking for 'cue stimuli'.
    # (i.e., events in {70, 71, 72, 73, 74, 75}.
    if temp_cue == 0:
        # event 70 is an 'A cue'
        if new_events[:, 2][i] == 70:
            # 'A cue' was found; move on
            temp_cue = 1
        # events 71, 72, 73, 74, 75 are 'B cues'.
        elif new_events[:, 2][i] in {71, 72, 73, 74, 75}:
            # 'B cue' was found; move on
            temp_cue = 2
        continue
    # SECOND STEP: look for 'probe stimuli' (temp_cue > 0)
    elif temp_cue == 1:
        # cues followed by wrong key presses (events 112 & 113)
        # should be marked as invalid
        if new_events[:, 2][i] in {112, 113}:
            valid = False
            continue
        if valid is True:
            if new_events[:, 2][i] == 76:
                new_events[:, 2][i] = 1
                # Set the temp_cue back to 0.
            elif new_events[:, 2][i] in {77, 78, 79, 80, 81}:
                new_events[:, 2][i] = 3
                # Set the temp_cue back to 0.
            temp_cue = 0
        elif valid is False:
            if new_events[:, 2][i] == 76:
                new_events[:, 2][i] = 11
                # Set the temp_cue back to 0.
            elif new_events[:, 2][i] in {77, 78, 79, 80, 81}:
                new_events[:, 2][i] = 31
                # Set the temp_cue back to 0.
            temp_cue = 0
            valid = True
    elif temp_cue == 2:
        # cues followed by wrong key presses (events 112 & 113)
        # should be marked as invalid
        if new_events[:, 2][i] in {112, 113}:
            valid = False
            continue
        if valid is True:
            if new_events[:, 2][i] == 76:
                new_events[:, 2][i] = 2
            elif new_events[:, 2][i] in {77, 78, 79, 80, 81}:\
                new_events[:, 2][i] = 4
            temp_cue = 0
        elif valid is False:
            if new_events[:, 2][i] == 76:
                new_events[:, 2][i] = 21
            elif new_events[:, 2][i] in {77, 78, 79, 80, 81}:
                new_events[:, 2][i] = 41
            temp_cue = 0
            valid = True