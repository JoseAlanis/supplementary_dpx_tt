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

from mvpa_stats import run_gat, get_stats_lines, get_p_scores, plot_results

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

# save GAT scores
np.save(fname.results + '/gat_scores_ridge.npy', scores)
# load GAT scores
scores = np.load(fname.results + '/gat_scores_ridge.npy')

# run TFCE on decoder scores
p_ = get_p_scores(scores, chance=.5, tfce=True)

# save p values
np.save(fname.results + '/p_vals_gat_ridge.npy', p_)
# load p values
p_vals = np.load(fname.results + '/p_vals_gat_ridge.npy')

# classifier performance
stats_dict = get_stats_lines(scores)

# save classifier performance
with open(fname.results + '/stat_lines_dict_ridge.pkl', 'wb') as f:
    pickle.dump(stats_dict, f)

# execute this cell to load previously calculated(and saved) scores
with open(fname.results + '/stat_lines_dict_ridge.pkl', 'rb') as f:
    stats_dict = pickle.load(f)
del f

# plot GAT results
fg1 = plot_results(stats_dict, scores,
                   decoder='ridge', p_values=p_vals)

fg1.savefig(fname.figures + '/gat_fig.pdf', dpi=300)



# dec = "ridge"
# scores = scores_dict[dec]["scores"].copy()
#
# del scores_dict
#
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
