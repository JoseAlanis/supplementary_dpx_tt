# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7.3 / mne 0.18.1
#
# --- eeg pre-processing for DPX TT
# --- version: june 2019
#
# --- import data, drop flat channels,
# --- extract task blocks

# ========================================================================
# ------------------- import relevant extensions -------------------------
import glob
import os.path as op
from os import mkdir

import re
import numpy as np

from mne import create_info, find_events, concatenate_raws
from mne.io import read_raw_bdf
from mne.channels import read_montage

# ========================================================================
# --- global settings
# --- prompt user to set project path
root_path = input("Type path to project directory: ")

# look for directory
if op.isdir(root_path):
    print("Setting 'root_path' to ", root_path)
else:
    raise NameError('Directory not found!')

# path to eeg files
data_path = op.join(root_path, 'sub-*')

# path for saving output
derivatives_path = op.join(root_path, 'derivatives')

# create directory for derivatives
if not op.isdir(derivatives_path):
    mkdir(derivatives_path)
    mkdir(op.join(derivatives_path, 'extract_blocks'))

# path for saving script output
output_path = op.join(derivatives_path, 'extract_blocks')

# files to be analysed
files = sorted(glob.glob(op.join(data_path, 'eeg/*.bdf')))

# ========================================================================
# -- define further variables that apply to all files in the data set
task_description = 'DPX, effects of time on task'
# eeg channel names and locations
montage = read_montage(kind='biosemi64')

# channels to be exclude from import
exclude = ['EXG5', 'EXG6', 'EXG7', 'EXG8']

# ========================================================================
# ------------ loop through files and extract blocks  --------------------
for file in files:

    # --- 1) set up paths and file names -----------------------
    filepath, filename = op.split(file)
    # subject in question
    subj = re.findall(r'\d+', filename)[0].rjust(3, '0')

    # --- 2) import the data -----------------------------------
    raw = read_raw_bdf(file,
                       montage=montage,
                       preload=True,
                       exclude=exclude)
    # reset `orig_time` in annotations
    raw.annotations.orig_time = None

    # --- 2) check channels variance  ------------------------
    # look for channels with zero (or near zero variance; i.e., flat-lines)
    chans_to_drop = []
    for chan in raw.info['ch_names']:
        if (np.std(raw.get_data(raw.info['ch_names'].index(chan))) * 1e6) < 10.:
            chans_to_drop.append(chan)
    # print summary
    print('Following channels were dropped as variance ~ 0:', chans_to_drop)
    # remove them from data set
    raw.drop_channels(chans_to_drop)

    # --- 3) save data set information  ------------------------
    # Note the sampling rate of recording
    sfreq = raw.info['sfreq']
    # all channels in raw
    chans = raw.info['ch_names']
    # channels in montage
    montage_chans = montage.ch_names
    # nr of eeg channels
    n_eeg = len([chan for chan in chans if chan in montage_chans])
    # channel types
    types = []
    for chan in chans:
        if chan in montage_chans:
            types.append('eeg')
        elif re.match('EOG|EXG', chan):
            types.append('eog')
        else:
            types.append('stim')

    # create custom info for subj file
    info_custom = create_info(chans, sfreq, types, montage)
    # description / name of experiment
    info_custom['description'] = task_description
    # overwrite file info
    raw.info = info_custom

    # --- 4) set reference to remove residual line noise  ------
    raw.set_eeg_reference(['Cz'], projection=False)

    # --- 5) get events in data --------------------------------
    # Get events
    events = find_events(raw,
                         stim_channel='Status',
                         output='onset',
                         min_duration=0.002)
    # Cue events
    cue_evs = events[(events[:, 2] >= 70) & (events[:, 2] <= 75), ]
    print('\n There are', len(cue_evs), 'events.')

    # latencies and difference between two consecutive cues
    latencies = cue_evs[:, 0]
    diffs = [y - x for x, y in zip(latencies, latencies[1:])]

    # Get first event after a long break (i.e., pauses between blocks),
    # Time difference in between blocks should be  > 10 seconds)
    diffs = [diff / sfreq for diff in diffs]
    breaks = [diff for diff in range(len(diffs)) if diffs[diff] > 10]
    print('\n Identified breaks at positions', breaks)

    # --- 7) save start and end points of task blocks  ---------
    # start first block
    b1s = (latencies[breaks[0] + 1] - (2 * sfreq)) / sfreq
    # end of first block
    b1e = (latencies[(breaks[1])] + (6 * sfreq)) / sfreq

    # start second block
    b2s = (latencies[breaks[1] + 1] - (2 * sfreq)) / sfreq
    # end of first block
    if len(breaks) > 2:
        b2e = (latencies[breaks[2]] + (6 * sfreq)) / sfreq
    else:
        b2e = (latencies[-1] + (6 * sfreq)) / sfreq

    # block durations
    print('Block 1 from', round(b1s, 3), 'to', round(b1e, 3), '\nBlock length ',
          round(b1e - b1s, 3))
    print('Block 2 from', round(b2s, 3), 'to', round(b2e, 3), '\nBlock length ',
          round(b2e - b2s, 3))

    # --- 8) extract block data --------------------------------
    # Block 1
    raw_bl1 = raw.copy().crop(tmin=b1s, tmax=b1e)
    # Block 2
    raw_bl2 = raw.copy().crop(tmin=b2s, tmax=b2e)

    # --- 9) concatenate data ----------------------------------
    raw_blocks = concatenate_raws([raw_bl1, raw_bl2])

    # --- 10) lower the sample rate  ---------------------------
    raw_blocks.resample(sfreq=256.)

    # --- 11) save segmented data  -----------------------------
    # create directory for save
    if not op.exists(op.join(output_path, 'sub-%s' % subj)):
        mkdir(op.join(output_path, 'sub-%s' % subj))

    # save file
    raw_blocks.save(op.join(output_path, 'sub-' + str(subj),
                            'sub-%s_task_blocks-raw.fif' % subj),
                    overwrite=True)

    # --- 12) save script summary  ------------------------------
    # get events in segmented data
    events = find_events(raw_blocks,
                         stim_channel='Status',
                         output='onset',
                         min_duration=0.002)
    # number of trials
    nr_trials = len(events[(events[:, 2] >= 70) & (events[:, 2] <= 75), ])

    # write summary
    name = 'sub-%s_task_blocks_summary.txt' % subj
    sfile = open(op.join(output_path, 'sub-%s', name) % subj, 'w')
    #     # block info
    sfile.write('Block_1_from_' + str(round(b1s, 2)) + '_to_' +
                str(round(b1e, 2)) + '\n')
    sfile.write('Block 2 from ' + str(round(b2s, 2)) + '_to_' +
                str(round(b2e, 2)) + '\n')
    sfile.write('Block_1_length:\n%s\n' % round(b1e - b1s, 2))
    sfile.write('Block_2_length:\n%s\n' % round(b2e - b2s, 2))
    # number of trials in file
    sfile.write('number_of_trials_found:\n%s\n' % nr_trials)
    # channels dropped
    sfile.write('channels_with_zero_variance:\n')
    for ch in chans_to_drop:
        sfile.write('%s\n' % ch)
    sfile.close()

    del raw, raw_bl1, raw_bl2, raw_blocks