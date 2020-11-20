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
import seaborn as sns

from scipy.stats import wilcoxon

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mne.decoding import cross_val_multiscore
from mne.decoding import GeneralizingEstimator
from mne.parallel import parallel_func
from mne.stats import fdr_correction, spatio_temporal_cluster_1samp_test, \
    ttest_1samp_no_p
from mne import read_epochs

from config import fname, n_jobs


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
def parallel_stats(X, function=_my_wilcoxon, correction='FDR', n_jobs=n_jobs):

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
def stats_tfce(X, n_permutations=2**10, threshold=dict(start=.2, step=.2),
               n_jobs=n_jobs):
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
def run_gat(subj, decoder="ridge", n_jobs=n_jobs):
    """
    Function to run Generalization Across Time (GAT).

    Parameters
    ----------
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

    clf = make_pipeline(StandardScaler(),
                        decoder_dict[decoder])
    gen_clf = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=n_jobs)
    scores = cross_val_multiscore(gen_clf, epochs.get_data(),
                                  epochs.events[:, -1],
                                  cv=5,
                                  n_jobs=n_jobs).mean(0)

    data = epochs.get_data()
    labels = epochs.events[:, -1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # calculate prediction confidence scores
    preds = np.empty((len(labels), data.shape[2], data.shape[2]))
    for train, test in cv.split(data, labels):
        gen_clf.fit(data[train], labels[train])
        d = gen_clf.decision_function(data[test])
        preds[test] = d

    return scores, preds  # return subject scores and prediction confidence


def get_p_scores(scores, chance=.5, tfce=False):
    """
    Calculate p_values from scores for significance masking using either
    TFCE or FDR.

    Parameters
    ----------
    scores: numpy array
        Calulated scores from decoder
    chance: float
        Indicate chance level
    tfce: True | False
        Specify whether to Threshold Free Cluster Enhancement (True)
        or FDR (False)
    """
    p_values = (parallel_stats(scores - chance) if tfce is False else stats_tfce(scores - chance))
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
        clus.append((([ls[0], ls[-1]] if round((ls[1] - ls[0]), 2) <= 0.01 else [ls[1], ls[-1]])
                     if len(group[key]) > 1 else group[key]))
    return clus


def get_stats_lines(scores, n4_times=[.15, .25], p6_times=[.35, .45], alphas=[.05, .01]):
    """
    Calculate subject level decoder performances for each of the times series
    plots and perform FDR correction (p<0.05 and p<0.01).
    Creates a dictionary of relevant stats.

    Parameters
    ----------

    scores: array
    n4_times: list
        List of tmin and tmax for N400 time window
    p6_times: list
        List of tmin and tmax for P600 time window
    alphas: list
        List of alphas for significance masking default masks for p<0.05 and p<0.01
    """
    from config import subjects

    times = get_epochs(subjects[0]).times
    l_times, xx = list(times), np.meshgrid(times)[0]
    n4_min, n4_max, p6_min, p6_max = [(np.abs(t-np.asarray(l_times))).argmin() for t in n4_times + p6_times]
    alpha1, alpha2 = alphas

    # average classifier performance over time for time window of interest for each subject
    n4 = scores[:, n4_min:n4_max, :].mean(1)
    p6 = scores[:, p6_min:p6_max, :].mean(1)

    # FDR correction and significance testing
    n4_pvalues = parallel_stats(list(n4 - 0.5))
    p6_pvalues = parallel_stats(list(p6 - 0.5))

    # mask time points of significance that are p<0.05 (alpha1) and p<0.01 (alpha2)
    n1, n2 = xx[n4_pvalues < alpha1], xx[n4_pvalues < alpha2]
    p1, p2 = xx[p6_pvalues < alpha1], xx[p6_pvalues < alpha2]

    # get the diagonal (training_t==testing_t) for each subject, FDR correction, and mask time points of significance
    diag = np.asarray([sc.diagonal() for sc in scores])
    diag_pvalues = parallel_stats(list(diag - 0.5))
    diag1, diag2 = xx[diag_pvalues < alpha1], xx[diag_pvalues < alpha2]

    # average difference between classifier performance over time for time window of interest for each subject
    diff = np.array([(sc[p6_min:p6_max].mean(0) -
                      sc[n4_min:n4_max].mean(0))  # subtract N400 from P600 decoders
                     for sc in scores])
    # FDR correction and significance masking
    diff_pvalues = parallel_stats(list(diff))
    diff1, diff2 = xx[diff_pvalues < alpha1], xx[diff_pvalues < alpha2]

    # create dict of stats
    stats_dict = {"N170": [n4, n1, n2], "LPC": [p6, p1, p2], "diag": [diag, diag1, diag2],
                  "diff": [diff, diff1, diff2]}
    return stats_dict


def plot_image(data, times, mask=None, ax=None, vmax=None, vmin=None,
               draw_mask=None, draw_contour=None, colorbar=True,
               draw_diag=True, draw_zerolines=True, xlabel="Time (s)", ylabel="Time (s)",
               cbar_unit="%", cmap="RdBu_r", mask_alpha=.75, mask_cmap="RdBu_r"):
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
    ax.set_xlabel(xlabel)

    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_title(cbar_unit)
    ax.set_aspect(1. / ax.get_data_ratio())
    ax.set_title("GAT Matrix")
    ax.title.set_position([.5, 1.025])

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
        for c in ["N170", "LPC"]:
            for ii, s in enumerate(stats_dict[c][0]):
                for t, a in enumerate(s):
                    sub.append(ii), accuracy.append(a), time.append(times[t]), comp.append(c)
        df["Time (s)"], df["Subject"], df["Accuracy (%)"], df["Component"] = time, sub, accuracy, comp

    else:
        # create dataframe for diagonal or difference between N400&P600
        sub, time, ac = [], [], []
        df = pd.DataFrame()
        for ii, s in enumerate(stats_dict[df_type][0]):
            for t, a in enumerate(s):
                sub.append(ii), ac.append(a), time.append(times[t])
        df["Time (s)"], df["Subject"] = time, sub
        df["{}".format(("Accuracy (%)" if df_type == "diag" else "Difference in Accuracy (%)"))] = ac

    return df


def over_plot(stats_dict, fill, axes=None, df_type=False):
    """
    Plot line plots (component generalization, diagonal, component difference)
    """

    if df_type == "diff":
        # plot difference between decoders
        # plt_ = sns.lineplot(data=get_dfs(stats_dict, df_type=df_type),
        #                     # time="Time (s)",
        #                     ax=axes, ci=[95], color="k",
        #                     units="Subject",
        #                     # condition=None,
        #                     value="Difference in Accuracy (%)")

        plt_ = sns.lineplot(data=get_dfs(stats_dict, df_type=df_type),
                            color='k',
                            y='Difference in Accuracy (%)',
                            x="Time (s)",
                            ci=95,
                            ax=axes)

        plt_.title.set_text("Difference between N170 and LPC decoding performance across time")
        plt_.title.set_position([.5, 1.025])
        plt_.axvline(0, color="k", lw=0.5)
        plt_.axhline(0, color="k", lw=0.5)
        plt_.set_ylim(-0.1, 0.15)
        plt_.axvline(0.25, color="k", linestyle="--", lw=0.5)
        plt_.axvline(0.35, color="k", linestyle="--", lw=0.5)
        plt_.tick_params("both", labelsize=10)

        if fill:
            plt_.fill_betweenx([-0.8, 0.8], 0.15, 0.25,
                               alpha=0.15,
                               color=plt.cm.viridis(np.linspace(0., 1., 10))[0])
            plt_.fill_betweenx([-0.8, 0.8], 0.35, 0.45,
                               alpha=0.15,
                               color=plt.cm.viridis(np.linspace(0., 1., 10))[5])

    else:

        if df_type == "diag":  # plot diagonal
            # plt_ = sns.lineplot(data=get_dfs(stats_dict, df_type=df_type),
            #                     # time="Time (s)",
            #                     ci=[95],
            #                     units="Subject",
            #                     # condition=None,
            #                     linestyle="--", value="Accuracy (%)",
            #                     color="k", ax=axes)

            plt_ = sns.lineplot(data=get_dfs(stats_dict, df_type=df_type),
                                color='k',
                                linestyle="--",
                                y='Accuracy (%)',
                                x="Time (s)",
                                ci=95,
                                ax=axes)

        else:  # component generalization
            # plt_ = sns.lineplot(data=get_dfs(stats_dict, df_type),
            #                     # time="Time (s)",
            #                     ax=axes,
            #                     ci=[95],
            #                     color=(plt.cm.viridis(np.linspace(0., 1., 10))[0],
            #                            plt.cm.viridis(np.linspace(0., 1., 10))[5]),
            #                     units="Subject",
            #                     legend=True,
            #                     # condition="Component",
            #                     value="Accuracy (%)")

            plt_ = sns.lineplot(data=get_dfs(stats_dict, df_type=df_type),
                                hue='Component',
                                color=(plt.cm.viridis(np.linspace(0., 1., 10))[0],
                                       plt.cm.viridis(np.linspace(0., 1., 10))[5]),
                                y='Accuracy (%)',
                                x="Time (s)",
                                ci=95,
                                ax=axes)

        plt_.title.set_text(("Component generalization across time"
                             if not df_type else "Diagonal decoding performance"))
        plt_.title.set_position([.5, 1.025])
        plt_.axvline(0, color="k", lw=0.5)
        plt_.axhline(0.5, color="k", lw=0.5)
        plt_.axvline(0.25, color="k", linestyle="--", lw=0.5)
        plt_.axvline(0.35, color="k", linestyle="--", lw=0.5)
        plt_.tick_params("both", labelsize=10)
        plt_.set_ylim(0.43, 0.72)
        if fill:
            plt_.fill_betweenx([0.40, 0.75], 0.15, 0.25,
                               alpha=0.15,
                               color=plt.cm.viridis(np.linspace(0., 1., 10))[0])
            plt_.fill_betweenx([0.40, 0.75], 0.35, 0.45,
                               alpha=0.15,
                               color=plt.cm.viridis(np.linspace(0., 1., 10))[5])

    return plt_

def calc_xval(clus):
    """calculate xmin and xmax for axvlines to indicate time points of significance"""
    from config import subjects
    # get times
    times = get_epochs(subjects[0]).times
    # put the in a list
    l_times = list(times)

    return {"xmin": (clus[0] + abs(min(l_times))) / (abs(min(l_times)) + max(l_times)),
            "xmax": (clus[-1] + (abs(min(l_times)) if len(clus) != 1
                                 else abs(min(l_times))+.001)) / (abs(min(l_times)) + max(l_times))}

# def calc_xval(clus):
#     """calculate xmin and xmax for axvlines to indicate time points of significance"""
#     return {"xmin": (clus[0] + 0.3) / 1.6, "xmax": (clus[-1] + (0.3 if len(clus) != 1 else 0.301)) / 1.6}


def plot_results(stats_dict, scores, decoder="ridge",
                 p_values=None, fill=True):
    """
    Plotting results and masking for significance, thick lines index p<0.01
    and thin lines p<0.05 on time series plots, GAT is masked with TFCE
    and p<0.01
    """
    from config import subjects

    fg, axes = plt.subplots(nrows=2, ncols=2)
    fg.set_size_inches(14, 8)
    fg.set_tight_layout("tight")
    times = get_epochs(subjects[0]).times

    ### Plot GAT matrix ###
    data = np.array(scores)
    plot_image(data.mean(0), times, mask=p_values < 0.01, ax=axes[0, 0], vmax=.7, vmin=.3,
               draw_mask=True, draw_contour=True, colorbar=True,
               draw_diag=True, draw_zerolines=True, xlabel="Time (s)", ylabel="Time (s)",
               cbar_unit="%", cmap="RdBu_r", mask_alpha=.75, mask_cmap="RdBu_r");

    ### plot decoder performances trained at respective time windows (N4/P6) ##
    plt_gen = over_plot(stats_dict, fill, axes[1, 0])

    params = {"N170": {"val": 0.45, "color": plt.cm.viridis(np.linspace(0., 1., 10))[0]},
              "LPC": {"val": 0.44, "color": plt.cm.viridis(np.linspace(0., 1., 10))[5]}}
    # plot time points of significance according to FDR
    for key in params.keys():
        for it in [1, 2]:  # Indicates whether p<0.01 or 0.05, respectively. Influence line thickness
            for clus in find_clus(stats_dict[key][it]):
                if type(clus) != np.float64:
                    plt_gen.axhline(params[key]["val"], color=params[key]["color"],
                                    lw=(1 if it == 1 else 3), **calc_xval(clus))

    ### plot diagonal decoding performance ###
    plt_diag = over_plot(stats_dict, fill, axes[0, 1], "diag")
    for it in [1, 2]:  # FDR
        for clus in find_clus(stats_dict["diag"][it]):
            if type(clus) != np.float64:
                plt_diag.axhline(0.45, color="k", linestyle="-", lw=(1 if it == 1 else 3),
                                 **calc_xval(clus))

    ### plot difference between decoder performance in time windows ###
    plt_diff = over_plot(stats_dict, fill, axes[1, 1], "diff")
    for it in [1, 2]:  # FDR
        for clus in find_clus(stats_dict["diff"][it]):
            if type(clus) != np.float64:
                plt_diff.axhline(-0.08, color=(plt.cm.viridis(np.linspace(0., 1., 10))[0] if clus[-1] < .5

                                               else
                                               plt.cm.viridis(np.linspace(0., 1., 10))[5]),
                                 lw=(1 if it == 1 else 3), **calc_xval(clus))
    return fg
