"""
==============================
Run generalisation across time
==============================

Multivariate pattern analysis of evoked eeg activity

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import pickle

import numpy as np

from mvpa_stats import run_gat, get_stats_lines, get_p_scores

from config import subjects, fname
# exclude subjects 51
subjects = subjects[subjects != 51]

# choose decoder, can be one or multiple  of:
# "svm-lin", "ridge", "log_reg", or "svm-nonlin"
scores = np.zeros((len(subjects), 384, 384))
pred = dict()
# scores_dict = dict()
decoders = ["ridge"]
for d in decoders:
    for s, subj in enumerate(subjects):
        print(s, subj)
        score, predict = run_gat(subj, decoder=d)
        scores[s, :] = score
        pred[subj] = predict
        # scores_dict[d]["scores"].append(score)
        # scores_dict[d]["predicts"].append(predict)

# save scores
np.save(fname.results + '/gat_scores_ridge.npy', scores)

# load GAT scores
scores = np.load(fname.results + '/gat_scores_ridge.npy')

# run TFCE on decoder scores
p_ = get_p_scores(scores, chance=.5, tfce=True)

# dec = "ridge"
# scores = scores_dict[dec]["scores"].copy()
#
# del scores_dict
#
# scores = np.asarray(scores)
# stats_dict = get_stats_lines(scores)
#
#
# # save prediction confidence
# with open(fname.results + '/gat_pred_conf_ridge.pkl', 'wb') as f:
#     pickle.dump(pred, f)
#
# # execute this cell to load previously calculated(and saved) scores
# with open(fname.results + '/gat_pred_conf_ridge.pkl', 'rb') as f:
#     scores_dict = pickle.load(f)
#
# del f
