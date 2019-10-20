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
import pandas as pd

from mne import create_info, find_events, Annotations, \
    events_from_annotations, concatenate_raws
from mne.io import read_raw_bdf
from mne.channels import make_standard_montage

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
montage = make_standard_montage(kind='biosemi64')
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
                       preload=True,
                       exclude=exclude)
    # apply montage to data
    raw.set_montage(montage)
    # reset orig_time
    date_of_record = raw.annotations.orig_time

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
    # date of measurement
    raw.info['meas_date'] = (date_of_record, 0)

    # --- 4) set reference to remove residual line noise  ------
    raw.set_eeg_reference(['Cz'], projection=False)

    # --- 5) find cue events in data ---------------------------
    # get events
    events = find_events(raw,
                         stim_channel='Status',
                         output='onset',
                         min_duration=0.002)

    # cue events
    cue_evs = events[(events[:, 2] >= 70) & (events[:, 2] <= 75)]

    # latencies and difference between two consecutive cues
    latencies = cue_evs[:, 0] / sfreq
    diffs = [(y - x) for x, y in zip(latencies, latencies[1:])]

    # Get first event after a long break (i.e., pauses between blocks),
    # Time difference in between blocks should be  > 10 seconds)
    breaks = [diff for diff in range(len(diffs)) if diffs[diff] > 10]
    print('\n Identified breaks at positions', breaks)

    # --- 7) save start and end points of task blocks  ---------
    # subject '041' has more practice trials
    if subj == '041':
        # start first block
        b1s = latencies[breaks[2] + 1] - 2
        # end of first block
        b1e = latencies[breaks[3]] + 6

        # start second block
        b2s = latencies[breaks[3] + 1] - 2
        # end of second block
        b2e = latencies[breaks[4]] + 6

    # all other subjects have the same structure
    else:
        # start first block
        b1s = latencies[breaks[0] + 1] - 2
        # end of first block
        b1e = latencies[breaks[1]] + 6

        # start second block
        b2s = latencies[breaks[1] + 1] - 2
        # end of second block
        if len(breaks) > 2:
            b2e = latencies[breaks[2]] + 6
        else:
            b2e = latencies[-1] + 6

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

    # --- 11) extract events and save them in annotations ------
    annot_infos = ['onset', 'duration', 'description']
    annotations = pd.DataFrame(raw_blocks.annotations)
    annotations = annotations[annot_infos]

    # path to events .tsv
    events = find_events(raw_blocks,
                         stim_channel='Status',
                         output='onset',
                         min_duration=0.002)
    # import events
    events = pd.DataFrame(events, columns=annot_infos)
    events.onset = events.onset / raw_blocks.info['sfreq']

    # merge with annotations
    events = events.append(annotations, ignore_index=True)
    # sort by onset´
    events = events.sort_values(by=['onset'])

    # crate annotations object
    annotations = Annotations(events['onset'],
                              events['duration'],
                              events['description'],
                              orig_time=date_of_record)
    # apply to raw data
    raw_blocks.set_annotations(annotations)

    # drop stimulus channel
    raw_blocks.drop_channels('Status')

    # --- 12) save segmented data  -----------------------------
    # create directory for save
    if not op.exists(op.join(output_path, 'sub-%s' % subj)):
        mkdir(op.join(output_path, 'sub-%s' % subj))

    # save file
    raw_blocks.save(op.join(output_path, 'sub-' + str(subj),
                            'sub-%s_task_blocks-raw.fif' % subj),
                    overwrite=True)

    # --- 13) save script summary  ------------------------------
    # get cue events in segmented data
    events = events_from_annotations(raw_blocks, regexp='^[7][0-5]')[0]

    # number of trials
    nr_trials = len(events)

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
