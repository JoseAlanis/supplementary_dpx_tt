# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7.3 / mne 0.18.1
#
# --- eeg pre-processing for DPX TT
# --- version: june 2019
#
# --- remove bad components,
# --- export pruned data

# ========================================================================
# ------------------- import relevant extensions -------------------------
import os.path as op
from os import mkdir
from glob import glob

from re import findall

from mne import pick_types
from mne.io import read_raw_fif
from mne.preprocessing import read_ica, create_eog_epochs

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
data_path = op.join(derivatives_path, 'artifact_detection')
# path to ica files
ica_path = op.join(derivatives_path, 'ica')

# create directory for output
if not op.isdir(op.join(derivatives_path, 'pruned_with_ica')):
    mkdir(op.join(derivatives_path, 'pruned_with_ica'))

# path for saving output
output_path = op.join(derivatives_path, 'pruned_with_ica')

# files to be analysed
files = sorted(glob(op.join(data_path, 'sub-*', '*-raw.fif')))

# ========================================================================
# ---------------- loop through files and fit ICA ------------------------
for file in files:

    # --- 1) Set up paths and file names -----------------------
    filepath, filename = op.split(file)
    # subject in question
    subj = findall(r'\d+', filename)[0].rjust(3, '0')

    # create directory for save
    if not op.exists(op.join(output_path, 'sub-%s' % subj)):
        mkdir(op.join(output_path, 'sub-%s' % subj))

    # --- 2) import the data -----------------------------------
    raw = read_raw_fif(file, preload=True)

    # get eogs
    eogs = pick_types(raw.info, eog=True)

    if len(eogs) > 2:
        raw.drop_channels(raw.info['ch_names'][-3])
        raw.drop_channels(raw.info['ch_names'][-1])

    # --- 4) import ICA weights --------------------------------
    ica = read_ica(op.join(ica_path,
                           'sub-%s' % subj,
                           'sub-%s_ica_weights-ica.fif' % subj))

    # --- 5) find "eog" components via correlation -------------
    # create "blink ERP"
    eog_average = create_eog_epochs(raw,
                                    reject=dict(eeg=3e-4),
                                    picks='eeg').average()
    # get single blink trials
    eog_epochs = create_eog_epochs(raw,
                                   reject=dict(eeg=3e-4))
    # find matching components via correlation
    eog_inds, scores = ica.find_bads_eog(eog_epochs,
                                         reject_by_annotation=True)

    # --- 6) inspect component time series  --------------------
    ica.plot_sources(raw, exclude=eog_inds, block=True)

    # --- 7) look at correlation scores of components ----------
    fig = ica.plot_scores(scores, exclude=ica.exclude)
    fig.savefig(op.join(output_path, 'sub-%s' % subj,
                        'sub-%s_r_scores.pdf' % subj))
    del fig

    # look at source time course
    fig = ica.plot_sources(eog_average, exclude=ica.exclude)
    fig.savefig(op.join(output_path, 'sub-%s' % subj,
                        'sub-%s_sources.pdf' % subj))
    del fig

    # save component properties
    for ind in ica.exclude:
        fig = ica.plot_properties(eog_epochs,
                                  picks=ind,
                                  psd_args={'fmax': 35.},
                                  image_args={'sigma': 1.})[0]
        fig.savefig(op.join(output_path, 'sub-%s' % subj,
                            'sub-%s_comp_%d.pdf' % (subj, ind)))
        del fig

    # --- 6) remove bad components --------------------------------
    # apply ica weights to raw data
    ica.apply(raw)

    # --- 7) check results -------------------------------------
    # plot pruned data
    raw.plot(n_channels=67, title=str(filename),
             scalings=dict(eeg=50e-6),
             bad_color='red',
             block=True)

    # --- 8) save pruned data -----------------------------------
    # pick electrodes to save
    picks = pick_types(raw.info,
                       eeg=True,
                       eog=False)

    # save file
    raw.save(op.join(output_path, 'sub-%s' % subj,
                     'sub-%s_pruned-raw.fif' % subj),
             overwrite=True)
