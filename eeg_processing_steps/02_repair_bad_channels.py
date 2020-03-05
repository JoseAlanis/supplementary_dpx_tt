# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7 / mne 0.19.2
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
from itertools import compress

from re import findall
import numpy as np

from mne.io import read_raw_fif
from mne import pick_types, Report

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


def z_score_nan(x):
    """
        z scoring after removing nas

    """
    from scipy import nanmean, nanstd

    z_score = (x - nanmean(x)) / nanstd(x)

    return z_score


#
raw = read_raw_fif(files[0], preload=True)

# index of eeg channels
picks = pick_types(raw.info,
                   eeg=True)


chan_info = raw.info['chs']
chan_locs = np.asarray([chan_info[ch]['loc'][:3] for ch in picks])
chan_dist = np.linalg.norm(chan_locs - chan_locs[:, None], axis=-1)

chan_dist[chan_dist > 0.04] = 0
weig = np.array(chan_dist, dtype=bool)

# ========================================================================
# ----------- loop through files and detect artifacts --------------------
for file in files:

    # --- 1) set up paths and file names -----------------------
    file_path, filename = op.split(file)

    # subject in question
    subj = findall(r'\d+', filename)[0].rjust(3, '0')

    # --- 2) import the preprocessed data ----------------------
    raw = read_raw_fif(file, preload=True)

    # index of eeg channels
    picks = pick_types(raw.info,
                       eeg=True)

    # --- 3) detect bad sensors via correlation ----------------

    # --- 3.1) create copy of raw and set reference
    raw_copy = raw.copy()

    # --- 3.2) apply filter to data
    raw_copy.filter(l_freq=0.1, h_freq=40., picks=['eeg', 'eog'],
                    filter_length='auto',
                    l_trans_bandwidth='auto',
                    h_trans_bandwidth='auto',
                    method='fir',
                    phase='zero',
                    fir_window='hamming',
                    fir_design='firwin',
                    n_jobs=2)

    # --- 3.4) set eeg reference
    raw_copy.set_eeg_reference(ref_channels='average',
                               projection=True)
    raw_copy.apply_proj()

    # --- 3.5) extract channel data for inspection
    chan_data = raw_copy.get_data(picks)

    # --- 3.6) compute correlation
    chan_corr = np.corrcoef(chan_data)

    # absolute values
    chan_corr = np.abs(chan_corr)

    # check for nan-values and mask them
    masked_ch_corr = np.ma.masked_array(chan_corr, np.isnan(chan_corr))

    # --- 3.7) compute average correlation based on neighbors
    chn_nei_corr = np.average(masked_ch_corr, axis=1, weights=weig)
    chn_nei_corr = chn_nei_corr.filled(np.nan)

    chn_nei_uncorr_z = z_score_nan(1 - chn_nei_corr)

    max_pow = np.sqrt(np.sum(chan_data ** 2, axis=1))
    max_Z = z_score_nan(max_pow)

    feat_vec = (np.abs(chn_nei_uncorr_z) + np.abs(max_Z)) / 2
    max_th = feat_vec > 3

    raw_copy.info['bads'] = list(compress(raw_copy.info['ch_names'], max_th))

    raw_copy.plot(n_channels=68, scalings=dict(eeg=100e-6), bad_color='red',
                  block=True)

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
