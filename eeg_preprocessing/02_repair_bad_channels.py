# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7 / mne 0.20
#
# --- eeg pre-processing for DPX TT
# --- version: june 2019
#
# --- detect and repair bad channels using ransac,
# --- average reference

# ========================================================================
# ------------------- import relevant extensions -------------------------
import os.path as op
from os import mkdir
from glob import glob

from re import findall

from mne.io import read_raw_fif
from mne import pick_types, make_fixed_length_events, Epochs, Report

from autoreject import Ransac

# ========================================================================
# --- global settings
# --- prompt user to set project path
root_path = input("Type path to project directory: ")

# look for directory
if op.isdir(root_path):
    print("Setting 'root_path' to ", root_path)
else:
    raise NameError('Directory not found!')

# derivatives path
derivatives_path = op.join(root_path, 'derivatives')

# path to eeg files
data_path = op.join(derivatives_path, 'extract_blocks')

# create directory for output
if not op.isdir(op.join(derivatives_path, 'interpolate_bads')):
    mkdir(op.join(derivatives_path, 'interpolate_bads'))

# path for saving output
output_path = op.join(derivatives_path, 'interpolate_bads')

# files to be analysed
files = sorted(glob(op.join(data_path, 'sub-*', '*-raw.fif')))

# ========================================================================
# ----------- loop through files and detect artifacts --------------------
for file in files:

    # --- 1) set up paths and file names -----------------------
    file_path, filename = op.split(file)

    # subject in question
    subj = findall(r'\d+', filename)[0].rjust(3, '0')

    # --- 2) import the preprocessed data ----------------------
    raw = read_raw_fif(file, preload=True)

    # index of eogs and stim channels
    picks = pick_types(raw.info,
                       eeg=True)

    # --- 3) detect and interpolate bad sensors using RANSAC ---
    # create copy of raw and set reference
    raw_copy = raw.copy()
    # set eeg reference
    raw_copy.set_eeg_reference(ref_channels='average',
                               projection=True)
    raw_copy.apply_proj()
    ch_data = raw_copy.get_data(picks)

    import numpy as np
    ch_corr = np.abs(np.corrcoef(ch_data))

    neigh_max_distance = 0.04
    ch_locs = np.asarray([x['loc'][:3] for x in raw.info['chs'][0:64]])
    chns_dist = np.linalg.norm(ch_locs - ch_locs[:, None], axis=-1)
    chns_dist[chns_dist > neigh_max_distance] = 0

    weig = np.array(chns_dist, dtype=bool)
    chn_nei_corr = np.average(ch_corr, axis=1, weights=weig)

    # define length for epochs extraction
    tstep = 1.0

    # create epochs for detection procedure
    events = make_fixed_length_events(raw, duration=tstep)
    epochs = Epochs(raw_copy, events,
                    tmin=0.0, tmax=tstep,
                    baseline=None,
                    reject=dict(eeg=300e-6),
                    preload=True)

    # channels that should be ignored during the artifact detection procedure
    ignore_ch = {'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8'}
    # update dict
    picks = [raw.info['ch_names'].index(chan) for chan in raw.info['ch_names'] if chan not in ignore_ch]

    # define RANSAC parameters
    ransac = Ransac(verbose='progressbar', n_jobs=1)

    # fit RANSAC algorithm to find bad sensors
    epochs_clean = ransac.fit_transform(epochs)

    # --- 4) interpolate bad channels --------------------------
    raw.info['bads'] = ransac.bad_chs_
    # create figure of sensors
    fig = raw.plot_sensors(show=False,
                           show_names=True,
                           title='Sensor positions, (RED = interpolated)')

    # interpolate bads
    raw.interpolate_bads(reset_bads=True,
                         verbose=False,
                         mode='accurate')

    # --- 5) apply reference to clean data ---------------------
    # set eeg reference
    raw.set_eeg_reference(ref_channels='average',
                          projection=True)
    raw.apply_proj()

    # apply filter
    raw.filter(l_freq=0.1, h_freq=50, picks=['eeg', 'eog'],
               filter_length='auto',
               l_trans_bandwidth='auto',
               h_trans_bandwidth='auto',
               method='fir',
               phase='zero',
               fir_window='hamming',
               fir_design='firwin',
               n_jobs=2)

    # --- 7) save data -----------------------------------------
    subj_path = op.join(output_path, 'sub-%s' % subj)
    # create directory for save
    if not op.exists(subj_path):
        mkdir(subj_path)

    # save file
    raw.save(op.join(subj_path, 'sub-%s_interpolation-raw.fif' % subj),
             overwrite=True)

    # --- 8) save html report -----------------------------------------
    # create raw data report
    pattern = 'sub-%s_interpolation-raw.fif' % subj
    report = Report(verbose=True, raw_psd=dict(fmax=60))
    report.parse_folder(subj_path, pattern=pattern, render_bem=False)

    # add figure showing bad sensors (if any)
    report.add_figs_to_section(fig,
                               captions='sub-%s channels' % subj,
                               section='channels')
    # save to disk
    report.save(op.join(subj_path, 'sub-%s_interpolation-report.html' % subj),
                overwrite=True)
