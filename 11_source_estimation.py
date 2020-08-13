"""
========================================
Source estimation using LCMV beamforming
========================================

Fit Linearly Constrained Minimum Variance (LCMV) beamformer for source
estimation of ERP signals

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import os.path as op

import pickle

from mne.beamformer import make_lcmv, apply_lcmv
from mne.datasets import fetch_fsaverage
from mne.epochs import equalize_epoch_counts

from mne import read_epochs, make_forward_solution, compute_covariance

# All parameters are defined in config.py
from config import subjects, fname, LoggingFormat, n_jobs

# load MNE's built-in fsaverage transformation
trans = 'fsaverage'
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# volume source space
src = op.join(fs_dir, 'bem', 'fsaverage-vol-5-src.fif')
# boundary-element model
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# baseline period (e.g., for computation of noise covariance)
baseline = (-0.300, -0.050)

# placeholders for results
stcs_a = []
stcs_b = []

###############################################################################
# 1) loop through subjects and LCMV spatial filters for condition specific
# (i.e., cue A and cue B) ERPs

for subj in subjects:

    # log progress
    print(LoggingFormat.PURPLE +
          LoggingFormat.BOLD +
          'Creating LCMV spacial filter for subject %s' % subj +
          LoggingFormat.END)

    # import the output from previous processing step
    input_file = fname.output(subject=subj,
                              processing_step='cue_epochs',
                              file_type='epo.fif')
    cue_epo = read_epochs(input_file, preload=True)

    # extract epochs relevant for analysis apply baseline correction
    a_epo = cue_epo['Correct A']
    a_epo.apply_baseline(baseline=baseline).crop(tmin=-0.3, tmax=2.45)
    a_epochs_info = a_epo.info

    b_epo = cue_epo['Correct B']
    b_epo.apply_baseline(baseline=baseline).crop(tmin=-0.3, tmax=2.45)
    b_epochs_info = b_epo.info

    # number of epochs should be equal between conditions
    equalize_epoch_counts([a_epo, b_epo])

    # compute covariance for analysis time window
    # cue A epochs
    data_cov_a = compute_covariance(a_epo,
                                    tmin=0.01, tmax=2.45,
                                    method='shrunk')
    # cue B epochs
    data_cov_b = compute_covariance(b_epo,
                                    tmin=0.01, tmax=2.45,
                                    method='shrunk')

    # compute covariance for the baseline period (i.e., noise)
    # cue A epochs
    noise_cov_a = compute_covariance(a_epo, tmin=-0.3, tmax=-0.01,
                                     method='shrunk')
    # cue B epochs
    noise_cov_b = compute_covariance(b_epo, tmin=-0.3, tmax=-0.01,
                                     method='shrunk')

    # compute ERP
    evoked_a = a_epo.average()
    evoked_b = b_epo.average()

    # create forward solution
    fwd_a = make_forward_solution(a_epo.info,
                                  trans=trans, src=src, bem=bem,
                                  meg=False, eeg=True,
                                  mindist=5.0, n_jobs=n_jobs)
    fwd_b = make_forward_solution(b_epo.info,
                                  trans=trans, src=src, bem=bem,
                                  meg=False, eeg=True,
                                  mindist=5.0, n_jobs=n_jobs)

    # compute LCMV spatial filters
    filters_a = make_lcmv(evoked_a.info,
                          fwd_a, data_cov_a, reg=0.05,
                          noise_cov=noise_cov_a, pick_ori='max-power',
                          weight_norm='nai', rank=None)
    filters_b = make_lcmv(evoked_b.info, fwd_b, data_cov_b, reg=0.05,
                          noise_cov=noise_cov_b, pick_ori='max-power',
                          weight_norm='nai', rank=None)

    # delete forward solutions to free memory space
    del fwd_a, fwd_b

    # apply the spacial filters to the ERPs
    stc_a = apply_lcmv(evoked_a, filters_a, max_ori_out='signed')
    stc_b = apply_lcmv(evoked_b, filters_b, max_ori_out='signed')

    # store results in list
    stcs_a.append(stc_a)
    stcs_b.append(stc_b)

###############################################################################
# 2) save results

# create dictionary containing LCMV beamforming results
stcs = dict()
stcs['cue_a'] = stcs_a
stcs['cue_b'] = stcs_b

# save to disk
with open(fname.results + '/stcs_lcmv.pkl', 'wb') as f:
    pickle.dump(stcs, f, protocol=pickle.HIGHEST_PROTOCOL)
