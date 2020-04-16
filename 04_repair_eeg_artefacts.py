"""
===============================================
Repair EEG artefacts caused by ocular movements
===============================================

Identify "bad" components in ICA solution (e.g., components which are highly
correlated the time course of the electrooculogram).

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import matplotlib.pyplot as plt

import numpy as np

from mne import pick_types, open_report
from mne.io import read_raw_fif
from mne.preprocessing import read_ica, create_eog_epochs

# All parameters are defined in config.py
from config import subjects, fname, parser, LoggingFormat

# Handle command line arguments
args = parser.parse_args()
subject = args.subject

print(LoggingFormat.PURPLE +
      LoggingFormat.BOLD +
      'Finding and removing bad components for subject %s' % subject +
      LoggingFormat.END)

###############################################################################
# 1) Import the output from previous processing step
input_file = fname.output(subject=subject,
                          processing_step='repair_bads',
                          file_type='raw.fif')
raw = read_raw_fif(input_file, preload=True)

###############################################################################
# 2) Import ICA weights from precious processing step
ica_file = fname.output(subject=subject,
                        processing_step='fit_ica',
                        file_type='ica.fif')
ica = read_ica(ica_file)

###############################################################################
# 3) Find blink components via correlation with EOG-channels
# get eogs indices and names
eogs = pick_types(raw.info, eog=True)
eog_names = [raw.ch_names[ch] for ch in eogs]

# place holder for blink components
blink_components = []

for n, eog in enumerate(eog_names):
    eog_epochs = create_eog_epochs(raw,
                                   ch_name=eog,
                                   reject_by_annotation=True)

    # find components that correlate with activity recorded at eog
    # channel in question
    eog_indices, eog_scores = ica.find_bads_eog(eog_epochs,
                                                ch_name=eog,
                                                reject_by_annotation=True)
    # if
    if eog_indices and eog_indices not in blink_components:
        for eog_i in eog_indices:
            fig = ica.plot_properties(eog_epochs,
                                      picks=eog_i,
                                      psd_args={'fmax': 35.},
                                      image_args={'sigma': 1.})[0]
            plt.close(fig)
            # 5) Create HTML report
            with open_report(fname.report(subject=subject)[0]) as report:
                report.add_figs_to_section(fig, 'Bad components identified '
                                                'by %s electrode' % eog,
                                           section='ICA',
                                           replace=True)
                report.save(fname.report(subject=subject)[1], overwrite=True,
                            open_browser=False)

    # component maps to use as templates for ocular artefacts
    blink_components.extend(eog_indices)

###############################################################################
# 4) Remove components identified as bad
ica.exclude = np.unique(blink_components)

# check if any others should be removed
ica.plot_sources(raw)

# apply ica weights to raw data
ica.apply(raw)

