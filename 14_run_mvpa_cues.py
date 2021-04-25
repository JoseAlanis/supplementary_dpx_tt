"""
==================================================
Run multivariate pattern analyses for cue activity
==================================================

Fits a classifier to test how well the scalp-topography (i.e., the
multivariate pattern) evoked by the presentation of a stimulus can
be used to discriminate among classes of stimuli across at a given
time point of the EEG epoch.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import pickle

import numpy as np

from mvpa_stats import run_gat, get_p_scores

from config import subjects, fname

# exclude subject 51
subjects = subjects[subjects != 51]

###############################################################################
# 1) choose decoder.

# In principle, it can be one or multiple of:
# "ridge", "log_reg", "svm-lin", or "svm-nonlin"
# Here we use a Ridge regression classifier (i.e., least-squares with
# Tikhonov regularisation)
decoders = ["ridge"]

###############################################################################
# 2) run MVPA with chosen decoder

# initialise place holders for results
scores = np.zeros((len(subjects), 384, 384))
pred = dict()
patterns = np.zeros((len(subjects), 64*384))

# compute classification scores for each participant
for d in decoders:
    for s, subj in enumerate(subjects):
        score, predict, pattern = run_gat(subj, decoder=d, n_jobs=16)
        scores[s, :] = score
        pred[subj] = predict
        patterns[s, :] = pattern

# save classification scores
np.save(fname.results + '/gat_scores_ridge.npy', scores)
# save topographical patterns
np.save(fname.results + '/gat_patterns_ridge.npy', patterns)

# save classifier predictions
with open(fname.results + '/gat_predictions_ridge.pkl', 'wb') as f:
    pickle.dump(pred, f)
# clean up / free space
del f, pred

###############################################################################
# 3) run permutations cluster test on classification scores
# to assesses significance

# compute p values
p_values = get_p_scores(scores, chance=0.5, tfce=True, n_jobs=2)

# save p values
np.save(fname.results + '/p_vals_gat_ridge.npy', p_values)
