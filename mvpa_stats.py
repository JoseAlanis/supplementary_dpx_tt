"""
==============
MVPA functions
==============

Utility functions for estimation and statistical analysis of MVPA parameters.

Authors: Functions retrieved and adapted from
         https://github.com/heikele/GAT_n4-p6, also see:
         Heikel, E., Sassenhagen, J., & Fiebach, C. J. (2018).
         Time-generalized multivariate analysis of EEG responses reveals a
         cascading architecture of semantic mismatch processing.
         Brain and language, 184, 43-53.
         Changes made by José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from scipy.stats import wilcoxon

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mne.decoding import cross_val_multiscore, GeneralizingEstimator, \
    get_coef, Vectorizer
from mne.parallel import parallel_func
from mne.stats import fdr_correction, spatio_temporal_cluster_1samp_test, \
    ttest_1samp_no_p
from mne import read_epochs

from config import fname


# signed rank test
def _my_wilcoxon(X):
    out = wilcoxon(X)
    return out[1]


# loop function
def _loop(x, function):
    out = list()
    for ii in range(x.shape[1]):
        out.append(function(x[:, ii]))
    return out


# correct p values for multiple testing
def parallel_stats(X, function=_my_wilcoxon, correction='FDR', n_jobs=2):

    # check if correction method was provided
    if correction not in [False, None, 'FDR']:
        raise ValueError('Unknown correction')

    # reshape to 2D
    X = np.array(X)
    dims = X.shape
    X.resize([dims[0], np.prod(dims[1:])])

    # prepare parallel
    n_cols = X.shape[1]
    parallel, pfunc, n_jobs = parallel_func(_loop, n_jobs)
    n_chunks = min(n_cols, n_jobs)
    chunks = np.array_split(range(n_cols), n_chunks)
    p_values = parallel(pfunc(X[:, chunk], function) for chunk in chunks)
    p_values = np.reshape(np.hstack(p_values), dims[1:])
    X.resize(dims)

    # apply correction
    if correction == 'FDR':
        dims = p_values.shape
        _, p_values = fdr_correction(p_values)
        p_values = np.reshape(p_values, dims)

    return p_values


# one sample t-test
def _stat_fun(x, sigma=0, method='relative'):
    t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
    t_values[np.isnan(t_values)] = 0
    return t_values


# threshold free cluster permutation test
def stats_tfce(X, n_permutations=2**10,
               threshold=dict(start=0.2, step=0.2),
               n_jobs=2):
    X = np.array(X)
    T_obs_, clusters, p_values, _ = \
        spatio_temporal_cluster_1samp_test(
            X,
            out_type='mask',
            stat_fun=_stat_fun,
            n_permutations=n_permutations,
            threshold=threshold,
            n_jobs=n_jobs)

    p_values = p_values.reshape(X.shape[1:])

    return p_values


# function to import mne-epochs for participant
def get_epochs(subj):
    """
    Loads the single trial data for a participant (name)
    """

    input_file = fname.output(subject=subj,
                              processing_step='cue_epochs',
                              file_type='epo.fif')
    epoch = read_epochs(input_file)
    epoch.crop(tmin=-0.5, tmax=epoch.tmax, include_tmax=False)
    epoch.apply_baseline((-0.300, -0.050))

    return epoch


# run generalisation across time and condition
def run_gat(subj, decoder="ridge", n_jobs=2):
    """
    Function to run Generalization Across Time (GAT).

    Parameters
    ----------
    subj: int
    name: str
        Name (pseudonym) of individual subject.
    decoder: str
        Specify type of classifier -'ridge' for Ridge Regression (default),
        'lin-svm' for linear SVM 'svm' for nonlinear (RBF) SVM and 'log_reg'
        for Logistic Regression
    n_jobs: int
        The number of jobs to run in parallel.
    """
    # load cue A and cue B epochs
    epochs = get_epochs(subj)['Correct A', 'Correct B']

    # specify whether to use a linear or nonlinear SVM if SVM is used
    lin = ''  # if not svm it doesn't matter, both log_reg and ridge are linear
    if "svm" in decoder:
        decoder, lin = decoder.split("-")

    # build classifier pipeline #
    # pick a machine learning algorithm to use (ridge/SVM/logistic regression)
    decoder_dict = {
        "ridge": RidgeClassifier(class_weight='balanced',
                                 random_state=42,
                                 solver="sag"),
        "svm": SVC(class_weight='balanced',
                   kernel=("rbf" if "non" in lin else "linear"),
                   random_state=42),
        "log_reg": LogisticRegression(class_weight='balanced',
                                      random_state=42)}

    # get data and targets
    data = epochs.get_data()
    labels = epochs.events[:, -1]

    # create classifier pipeline
    clf = make_pipeline(StandardScaler(),
                        decoder_dict[decoder])
    gen_clf = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=n_jobs)

    # compute cross validated performance scores
    scores = cross_val_multiscore(gen_clf, data,
                                  labels,
                                  cv=5,
                                  n_jobs=n_jobs).mean(0)

    # calculate prediction confidence scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.empty((len(labels), data.shape[2], data.shape[2]))
    for train, test in cv.split(data, labels):
        gen_clf.fit(data[train], labels[train])
        d = gen_clf.decision_function(data[test])
        preds[test] = d

    # compute topographical patterns
    dat = Vectorizer().fit_transform(data)
    clf.fit(dat, labels)
    dat = dat - dat.mean(0, keepdims=True)

    # look for the type of classifier and get the weights
    if decoder == 'ridge':
        filt_ = clf.named_steps.ridgeclassifier.coef_.copy()
    elif decoder == 'svm':
        filt_ = clf.named_steps.svc.coef_.copy()
    elif decoder == 'log_reg':
        filt_ = clf.named_steps.logisticregression.coef_.copy()

    # Compute patterns using Haufe's trick: A = Cov_X . W . Precision_Y
    # cf.Haufe, et al., 2014, NeuroImage,
    # doi:10.1016/j.neuroimage.2013.10.067)
    inv_y = 1.
    patt_ = np.cov(dat.T).dot(filt_.T.dot(inv_y)).T

    # store the patterns accordingly
    if decoder == 'ridge':
        clf.named_steps.ridgeclassifier.patterns_ = patt_
    elif decoder == 'svm':
        clf.named_steps.svc.patterns_ = patt_
    elif decoder == 'log_reg':
        clf.named_steps.logisticregression.patterns_ = patt_

    # back transform using steps in pipeline
    patterns = get_coef(clf, 'patterns_', inverse_transform=True)

    # return subject scores,  prediction confidence and topographical patterns
    return scores, preds, patterns


def get_p_scores(scores, chance=.5, tfce=False, n_jobs=2):
    """
    Calculate p_values from scores for significance masking using either
    TFCE or FDR.

    Parameters
    ----------
    scores: numpy array
        Calculated scores from decoder
    chance: float
        Indicate chance level
    tfce: True | False
        Specify whether to Threshold Free Cluster Enhancement (True)
        or FDR (False)
    """
    p_values = (parallel_stats(scores - chance, n_jobs=n_jobs) if tfce is False
                else stats_tfce(scores - chance, n_jobs=n_jobs))
    return p_values


def grouper(iterable):
    """
    List of time points of significance, identifies neighbouring time points.
    """
    prev = None
    group = []
    for item in iterable:
        if not prev or round(item - prev, 2) <= .01:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def find_clus(sig_times):
    """
    Identify time points of significance from FDR correction, results in
    lists of ranges and individual time points of significance and
    creates a dictionary for later use.

    Parameters
    ----------
    sig_times: list
        List of significant time points
    """
    group = dict(enumerate(grouper(sig_times)))
    clus = []
    for key in group.keys():
        ls = group[key]
        clus.append((([ls[0], ls[-1]] if round((ls[1] - ls[0]), 2) <= 0.01
                      else [ls[1], ls[-1]]) if len(group[key]) > 1
                     else group[key]))
    return clus


def get_stats_lines(scores, times, test_times, alphas=[.05, .01]):
    """
    Calculate subject level decoder performances for each of the times series
    plots and perform FDR correction (p<0.05 and p<0.01).
    Creates a dictionary of relevant stats.

    Parameters
    ----------

    scores: array
    times: array
    test_times: dict
    alphas: list
        List of alphas for significance masking default masks for p<0.05
        and p<0.01
    """

    # get alpha levels
    alpha1, alpha2 = alphas

    # get the diagonal (training_t==testing_t) for each subject,
    # FDR correction, and mask time points of significance
    diag = np.asarray([sc.diagonal() for sc in scores])
    diag_pvalues = parallel_stats(list(diag - 0.5))
    diag1, diag2 = times[diag_pvalues < alpha1], times[diag_pvalues < alpha2]

    # get component boundaries for perfomance analysis
    min_max = {k: [(np.abs(v[0] - times)).argmin(),
                   (np.abs(v[1] - times)).argmin()]
               for (k, v) in test_times.items()}

    # average classifier performance over time for time window of interest
    # for each subject
    class_performance = {k: scores[:, v[0]:v[1], :].mean(1)
                         for (k, v) in min_max.items()}

    # FDR correction and significance testing
    p_vals = {k: parallel_stats(list(v - 0.5))
              for (k, v) in class_performance.items()}

    # mask time points of significance that are p<0.05 (alpha1) and
    # p<0.01 (alpha2)
    masks = {k: [times[v < alpha] for alpha in alphas]
             for (k, v) in p_vals.items()}

    # # *** keep this just in case we neeed it later ***
    # # average difference between classifier performance over time for
    # time window of interest for each subject
    # diff = np.array([(sc[p6_min:p6_max].mean(0) -
    #                   sc[n4_min:n4_max].mean(0))
    #                  for sc in scores])
    # # FDR correction and significance masking
    # diff_pvalues = parallel_stats(list(diff))
    # diff1, diff2 = xx[diff_pvalues < alpha1], xx[diff_pvalues < alpha2]

    # create dict of diagonal stats
    diag_stats = {'diag': [diag, diag1, diag2]}

    # create dict of classifier stats
    class_stats = {k: [v, m[0], m[1]]
                   for ((k, v), (a, m)) in
                   zip(class_performance.items(), masks.items())}

    # object for return
    stats_dict = {**diag_stats, **class_stats}

    return stats_dict


def plot_image(data, times, mask=None, ax=None, vmax=None, vmin=None,
               draw_mask=None, draw_contour=None, colorbar=True,
               draw_diag=True, draw_zerolines=True,
               xlabel="Time (s)", ylabel="Time (s)",
               cbar_unit="%", cmap="RdBu_r",
               mask_alpha=.75, mask_cmap="RdBu_r"):
    """Return fig and ax for further styling of GAT matrix, e.g., titles

    Parameters
    ----------
    data: array of scores
    times: list of epoched time points
    mask: None | array
    ...
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    if vmax is None:
        vmax = np.abs(data).max()
    if vmin is None:
        vmax = np.abs(data).max()
        vmin = -vmax
    tmin, tmax = xlim = times[0], times[-1]
    extent = [tmin, tmax, tmin, tmax]
    im_args = dict(interpolation='nearest', origin='lower',
                   extent=extent, aspect='auto', vmin=vmin, vmax=vmax)

    if mask is not None:
        draw_mask = True if draw_mask is None else draw_mask
        draw_contour = True if draw_contour is None else draw_contour
    if any((draw_mask, draw_contour,)):
        if mask is None:
            raise ValueError("No mask to show!")

    if draw_mask:
        ax.imshow(data, alpha=mask_alpha, cmap=mask_cmap, **im_args)
        im = ax.imshow(np.ma.masked_where(~mask, data), cmap=cmap, **im_args)
    else:
        im = ax.imshow(data, cmap=cmap, **im_args)
    if draw_contour and np.unique(mask).size == 2:
        big_mask = np.kron(mask, np.ones((10, 10)))
        ax.contour(big_mask, colors=["k"], extent=extent, linewidths=[1],
                   aspect=1,
                   corner_mask=False, antialiased=False, levels=[.5])
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)

    if draw_diag:
        ax.plot((tmin, tmax), (tmin, tmax), color="k", linestyle=":")
    if draw_zerolines:
        ax.axhline(0, color="k", linestyle=":")
        ax.axvline(0, color="k", linestyle=":")

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel, labelpad=10.0)

    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_title(cbar_unit)
    ax.set_aspect(1. / ax.get_data_ratio())
    ax.set_title("GAT Matrix", pad=10.0)
    # ax.title.set_position([.5, 1.025])

    return fig if ax is None else ax


def get_dfs(stats_dict, df_type=False):
    """Create DataFrames for time series plotting"""
    from config import subjects
    # get times
    times = get_epochs(subjects[0]).times

    if not df_type:
        # create dataframe for N400 and P600 decoders
        df = pd.DataFrame()
        sub, time, accuracy, comp = [], [], [], []
        comps = list(stats_dict.keys())
        comps = [i for i in comps if i != 'diag']
        for c in comps:
            for ii, s in enumerate(stats_dict[c][0]):
                for t, a in enumerate(s):
                    sub.append(ii)
                    accuracy.append(a)
                    time.append(times[t])
                    comp.append(c)
        df["Time (s)"] = time
        df["Subject"] = sub
        df["Accuracy (%)"] = accuracy
        df["Component"] = comp

    else:
        # create dataframe for diagonal or difference between components
        sub, time, ac = [], [], []
        df = pd.DataFrame()
        for ii, s in enumerate(stats_dict[df_type][0]):
            for t, a in enumerate(s):
                sub.append(ii), ac.append(a), time.append(times[t])
        df["Time (s)"], df["Subject"] = time, sub
        df["{}".format(("Accuracy (%)" if df_type == "diag"
                        else "Difference in Accuracy (%)"))] = ac

    return df
