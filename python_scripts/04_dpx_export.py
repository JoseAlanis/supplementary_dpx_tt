# --- Jose C. Garcia Alanis
# --- utf-8
# --- Python 3.6.2
#
# --- EEG prepossessing - DPX TT
# --- Version Sep 2018
#
# --- Apply baseline, crop for smaller file, and
# --- Export epochs to .txt

# =================================================================================================
# ------------------------------ Import relevant extensions ---------------------------------------
import mne
import glob
import os

# ========================================================================
# --- GLOBAL SETTINGS
# --- SET PATH TO .epoch-files and output
input_path = '/Users/Josealanis/Documents/Experiments/dpx_tt/eeg/dpx_tt_mne_epochs/'
output_path = '/Users/Josealanis/Documents/Experiments/dpx_tt/eeg/dpx_epochs_no_base/'

# === LOOP THROUGH FILES AND EXPORT EPOCHS ===============================
# --- Baseline correction for ERP analyses
for file in glob.glob(os.path.join(input_path, '*-epo.fif')):

    filepath, filename = os.path.split(file)
    filename, ext = os.path.splitext(filename)
    name = filename.split('_')[0] + '_dpx_epochs'

    # Read epochs
    epochs = mne.read_epochs(file, preload=True)
    # Apply baseline
    epochs.apply_baseline(baseline=(-0.3, -0.05))
    # Only keep time window fro -.3 to .99 sec. around motor response
    small = epochs.copy().crop(tmin=-1., tmax=2.49)

    # Transform to data frame
    epo = small.to_data_frame()
    # Round values
    epo = epo.round(3)
    # Export data frame
    epo.to_csv(output_path + name + '.txt', index=True)

# === LOOP THROUGH FILES AND EXPORT EPOCHS ===============================
# --- No baseline correction for time-frequency analyses
for file in glob.glob(os.path.join(input_path, '*-epo.fif')):

    filepath, filename = os.path.split(file)
    filename, ext = os.path.splitext(filename)
    name = filename.split('_')[0] + '_dpx_epochs'

    # Read epochs
    epochs = mne.read_epochs(file, preload=True)
    # Transform to data frame
    epo_nb = epochs.to_data_frame()
    # Round values
    epo_nb = epo_nb.round(3)
    # Export data frame
    epo_nb.to_csv(output_path + name + '.txt', index=True)
