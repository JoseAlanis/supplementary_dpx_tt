# --- Jose C. Garcia Alanis
# --- utf-8
# --- Python 3.6.2
#
# --- EEG prepossessing - DPX TT
# --- ICA decomposition,
# --- Version Jul 2018

# ==================================================================================================
# --------------------------------- Import relevant extensions -------------------------------------
import os
import mne
from mne import io
from mne.preprocessing import ICA
# from mne.preprocessing import create_eog_epochs, create_ecg_epochs


# --- 1) READ IN THE DATA ---------------------------------
# Set working directory.
os.chdir('/Users/Josealanis/Documents/Experiments/dpx_tt/')
# EEG set up and file path.
montage = mne.channels.read_montage(kind='biosemi64')
data_path = './eeg/dpx_tt_mne_raws/data4-raw.fif'
# Import preprocessed data.
raw = io.read_raw_fif(data_path, preload=True)
# Check info
print(raw.info)

# Get events
evs = mne.find_events(raw,
                      stim_channel='Stim',
                      output='onset',
                      min_duration=0.002)

# Plot to check data
clip = None
raw.plot(n_channels=66,
         scalings=dict(eeg=100e-6),
         events=evs,
         bad_color='red',
         clipping=clip)


# --- 2) ICA DECOMPOSITION --------------------------------
# ICA parameters
n_components = 25
method = 'extended-infomax'
# decim = None
reject = dict(eeg=4e-4)

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


# --- 2) PLOT RESULTING COMPONENTS ------------------------
# Plot components
ica.plot_components(picks=range(0, 25))


# =========================================================
# EDIT INFORMATION BASED ON PARTICIPANTS RESULTS
# Inspect component properties
ica.plot_properties(raw,
                    picks=[0, 2, 6, 7],
                    psd_args={'fmax': 50.})

# --- 3) REMOVE COMPONENTS --------------------------------
# Set components to zero
ica.apply(raw,
          exclude=[0, 1])

# --- 4) CHECK PRUNNED DATA -------------------------------
# Plot to check
clip = None
raw.plot(n_channels=66,
         scalings=dict(eeg=100e-6),
         events=evs,
         bad_color='red',
         clipping=clip)

# --- 5) SAVE DATA ----------------------------------------
# Pick electrodes to use
picks = mne.pick_types(raw.info,
                       meg=False,
                       eeg=True,
                       eog=True,
                       stim=True)

# Save segmented and prunned data
raw.save('./eeg/dpx_tt_mne_prunned/data4_prunned-raw.fif',
         picks=picks,
         overwrite=True)


# ==================================================================================================
# --------------------------------- Import relevant extensions -------------------------------------

# --- 6) EXTRACT EPOCHS ----------------------------------------
# Copy of events
new_evs = evs.copy()
temp_cue = 0
valid = True

# Recode cue stimuli
for i in range(new_evs[:, 2].size):
    # Check if event is a cue
    if new_evs[:, 2][i] in {112, 113, 12, 13, 76, 77, 78, 79, 80, 81}:
        continue
        # If not go on to next trial
    # If event is a cue 'A': look for the next event and evaluate trial in question
    elif new_evs[:, 2][i] == 70:
        if new_evs[:, 2][i+1] in {112, 113}:
            new_evs[:, 2][i] = 111
            # Response too soon
            continue
            # Go on to next trial
        elif new_evs[:, 2][i+1] in {76, 77, 78, 79, 80, 81} and new_evs[:, 2][i+2] in {12, 13}:
            new_evs[:, 2][i] = 1
            # Correct response
            continue
            # Go on to next trial
        elif new_evs[:, 2][i+1] in {76, 77, 78, 79, 80, 81} and new_evs[:, 2][i+2] in {112, 113}:
            new_evs[:, 2][i] = 11
            # Incorrect response
            continue
            # Go on to next trial
    # If event is a cue 'B': look for the next event and evaluate trial in question
    elif new_evs[:, 2][i] in {71, 72, 73, 74, 75}:
        if new_evs[:, 2][i + 1] in {112, 113}:
            new_evs[:, 2][i] = 222
            # Response too soon
            continue
            # Go on to next trial
        elif new_evs[:, 2][i+1] in {76, 77, 78, 79, 80, 81} and new_evs[:, 2][i+2] in {12, 13}:
            new_evs[:, 2][i] = 2
            # Correct response
            continue
            # Go on to next trial
        elif new_evs[:, 2][i+1] in {76, 77, 78, 79, 80, 81} and new_evs[:, 2][i+2] in {112, 113}:
            new_evs[:, 2][i] = 22
            # Incorrect response
            continue


# if any(t == 11 for t in new_evs[:, 2]):
#     event_id = {'Correct A': 1, 'Correct B': 2,
#                 'Incorrect A': 11, 'Incorrect B': 22,
#                 'Invalid A': 111, 'Invalid B': 222}

event_id = {'Correct A': 1, 'Correct B': 2,
            'Incorrect A': 11, 'Incorrect B': 22,
            'Invalid A': 111, 'Invalid B': 222}

epochs = mne.Epochs(raw, new_evs, event_id,
                    on_missing='ignore',
                    tmin=-2.0,
                    tmax=2.5,
                    baseline=(-.25, 0),
                    preload=True,
                    reject_by_annotation=False)
print(epochs)

evoked_a_cue = epochs['Cue A'].average()
evoked_b_cue = epochs['Cue B'].average()

picks = mne.pick_types(raw.info,
                       meg=False,
                       eeg=True,
                       eog=False,
                       stim=False)

evoked_a_cue.plot_image(picks=picks, time_unit='s')

# e_ef = epochs.to_data_frame()
# e_ef.to_csv('./blabla.csv')
