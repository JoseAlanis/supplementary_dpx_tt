# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7 / mne 0.19.2
#
# --- eeg pre-processing for DPX TT
# --- version: february 2019
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
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mne import create_info, find_events, \
    Annotations, events_from_annotations, concatenate_raws
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
montage = make_standard_montage(kind='standard_1020')
# channels in montage
montage_chans = montage.ch_names
# channels to be exclude from import
exclude = ['EXG5', 'EXG6', 'EXG7', 'EXG8']

# colormap to use
cmap = cm.get_cmap('plasma')

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

    # --- 3) get info about the dataset ------------------------
    # measurement date
    date_of_record = raw.info['meas_date']
    # sampling rate
    sfreq = raw.info['sfreq']
    # channels
    chans = raw.info['ch_names']
    # channel types
    types = []
    for chan in chans:
        if chan in montage_chans:
            types.append('eeg')
        elif re.match('EOG|EXG', chan):
            types.append('eog')
        else:
            types.append('stim')
    # nr of eeg channels
    n_eeg = len([chan for chan in chans if chan in montage_chans])

    # create custom info for subj file
    info_custom = create_info(chans, sfreq, types, montage)

    # add improved dataset info
    raw.info = info_custom
    raw.info['meas_date'] = date_of_record

    # --- 3) check channels variance ----------------------------
    # check if dataset contains channels with zero or near zero variance
    # (i.e., flat lines)
    eeg_channel_variance = np.var(raw.get_data(picks='eeg'), axis=1) * 100e6
    norm_var = normalize(eeg_channel_variance[:, np.newaxis], axis=0).ravel()

    fig, ax = plt.subplots(figsize=(5, 10))
    for i in range(norm_var.shape[0]):
        ax.barh(i, norm_var[i], 0.9, color=cmap(norm_var[i]))
        ax.text(norm_var[i] + 0.05, i, raw.info['ch_names'][i],
                ha='center', va='center', fontsize=9)

    ax.set_xlim(0, 1)
    plt.title('eeg channel variance')
    plt.xlabel('normalized variance')
    plt.ylabel('channels')
    plt.show()

    eog_channel_variance = np.var(raw.get_data(picks='eog'), axis=1) * 100e6
    norm_eog_var = normalize(eog_channel_variance[:, np.newaxis], axis=0).ravel()

    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(norm_eog_var.shape[0]):
        ax.barh(i, norm_eog_var[i], 0.9, color=cmap(norm_var[i]))
        ax.text(norm_eog_var[i] + 0.05, i, raw.info['ch_names'][n_eeg + i],
                ha='center', va='center', fontsize=9)

    ax.set_xlim(0, 1)
    plt.title('eog channel variance')
    plt.xlabel('normalized variance')
    plt.ylabel('channels')
    plt.show()



    # indices
    flats = np.where(np.var(raw.get_data(), axis=1) * 1e6 < 10.)[0]
    # names
    flats = [raw.ch_names[ch] for ch in flats]
    # print summary
    print('Following channels were dropped as variance ~ 0:', flats)

    fig, ax = plt.subplots()
    for i in range(data.shape[1]):
        ax.bar(x + dx[i], data[:, i], width=d, label="label {}".format(i))

    plt.legend(framealpha=1).draggable()
    plt.show()


    # remove them from data set
    raw.drop_channels(flats)

    # --- 3) modify data set information  ----------------------
    # keep the sampling rate
    sfreq = raw.info['sfreq']
    # and date of measurement
    date_of_record = raw.annotations.orig_time



    # create custom info for subj file
    info_custom = create_info(chans, sfreq, types, montage)
    # description / name of experiment
    info_custom['description'] = task_description
    # overwrite file info
    raw.info = info_custom
    # replace date info
    raw.info['meas_date'] = (date_of_record, 0)

    # --- 3) set reference to remove residual line noise  ------
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

    raw_blocks.plot(n_channels=len(raw_blocks.ch_names),
                    scalings=dict(eeg=100e-6),
                    block=True)

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
    sfile.write('Block_1_from:\n%s to %s\n' % (str(round(b1s, 2)),
                                               str(round(b1e, 2))))
    sfile.write('Block_2_from:\n%s to %s\n' % (str(round(b2s, 2)),
                                               str(round(b2e, 2))))
    sfile.write('Block_1_length:\n%s\n' % round(b1e - b1s, 2))
    sfile.write('Block_2_length:\n%s\n' % round(b2e - b2s, 2))
    # number of trials in file
    sfile.write('number_of_trials_found:\n%s\n' % nr_trials)
    # channels dropped
    sfile.write('channels_with_zero_variance:\n')
    for ch in flats:
        sfile.write('%s\n' % ch)
    sfile.close()

    del raw, raw_bl1, raw_bl2, raw_blocks
