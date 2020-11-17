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

# choose decoder, can be one or multiple  of:
# "svm-lin", "ridge", "log_reg", or "svm-nonlin"
decoders = ["ridge"]

scores_dict = dict()
for d in decoders:
    scores_dict[d]=dict()
    scores_dict[d]["scores"] = []
    scores_dict[d]["predicts"] = []
    for subj in subjects:
        score, predict = run_gat(subj, decoder='ridge')
        scores_dict[d]["scores"].append(score)
        scores_dict[d]["predicts"].append(predict)

with open(fname.results + '/gat_scores_ridge.pkl', 'wb') as f:
    pickle.dump(scores_dict, f)

# execute this cell to load previously calculated(and saved) scores
with open(fname.results + '/gat_scores_ridge.pkl', 'rb') as f:
    scores_dict = pickle.load(f)

del f

# thick lines index p<0.01
# thin lines dec = "ridge"index p<0.05
dec = "ridge"
scores = scores_dict[dec]["scores"].copy()

stats_dict = get_stats_lines(np.asarray(scores))

# to run TFCE on decoder scores
p_ = get_p_scores(np.asarray(scores), chance=.5, tfce=True)
