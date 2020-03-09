"""
=============================================================
Extract segments of the data recorded during task performance
=============================================================

Segments that were recorded during the self-paced breaks (in between
experimental blocks) will be dropped.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
from mne.io import read_raw_fif
from mne import find_events, Annotations, events_from_annotations, \
    concatenate_raws

# All parameters are defined in config.py
from config import fname, parser

# Handle command line arguments
args = parser.parse_args()
subject = args.subject

print('Converting subject %s to BIDS' % subject)

###############################################################################
# 2) import the output from previous processing step
input_file = fname.output(subject=subject,
                          processing_step='raw_files',
                          file_type='raw')

raw = read_raw_fif(input_file, preload=True)

# get events
events = find_events(raw,
                     stim_channel='Status',
                     output='onset',
                     min_duration=0.002)

# cue events
cue_evs = events[(events[:, 2] >= 70) & (events[:, 2] <= 75)]

# latencies and difference between two consecutive cues
latencies = cue_evs[:, 0] / raw.info['sfreq']
diffs = [(y - x) for x, y in zip(latencies, latencies[1:])]

# Get first event after a long break (i.e., pauses between blocks),
# Time difference in between blocks should be  > 10 seconds)
breaks = [diff for diff in range(len(diffs)) if diffs[diff] > 10]
print('\n Identified breaks at positions', breaks)

# --- 7) save start and end points of task blocks  ---------
# subject '041' has more practice trials
if subject == 41:
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

# --- 8) extract block data --------------------------------
# Block 1
raw_bl1 = raw.copy().crop(tmin=b1s, tmax=b1e)
# Block 2
raw_bl2 = raw.copy().crop(tmin=b2s, tmax=b2e)

ref_raw_bl1 = raw_bl1.copy().set_eeg_reference(ref_channels='average', projection=True)

raw_bl1_filt = raw_bl1.copy().filter(l_freq=0.1, h_freq=40., picks=['eeg', 'eog'],
                                     filter_length='auto',
                                     l_trans_bandwidth='auto',
                                     h_trans_bandwidth='auto',
                                     method='fir',
                                     phase='zero',
                                     fir_window='hamming',
                                     fir_design='firwin',
                                     n_jobs=2)

ref_raw_bl1.plot(scalings=dict(eeg=100e-6, eog=100e-6), n_channels=len(raw.info['ch_names']))
raw_bl1.plot(scalings=dict(eeg=100e-6, eog=100e-6), n_channels=len(raw.info['ch_names']))
raw_bl1_filt.plot(scalings=dict(eeg=100e-6, eog=100e-6), n_channels=len(raw.info['ch_names']))


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
# sort by onset
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




# block durations
print('Block 1 from', round(b1s, 3), 'to', round(b1e, 3), '\nBlock length ',
      round(b1e - b1s, 3))
print('Block 2 from', round(b2s, 3), 'to', round(b2e, 3), '\nBlock length ',
      round(b2e - b2s, 3))