import numpy as np
import pandas as pd
import scipy.io
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import normaltest

def normality_fun(emg1, emg2, quantiles):

    # plot raw data
    T = 0.001
    N = emg1.shape[0]
    t = np.linspace(0, N*T, N)
    fig, axs = plt.subplots(1,2)
    axs[0].plot(t, emg1)
    axs[0].set_xlabel("Time (s)", fontsize=20)
    axs[0].set_ylabel("raw EMG (mV)", fontsize=20)
    axs[1].set_plot(t, emg2)
    axs[1].set_xlabel("Time (s)", fontsize=20)
    axs[1].set_ylabel("raw EMG (mV)", fontsize=20)
    fig.suptitle("Raw and Normalized EMG from biceps and triceps", fontsize=22)
    fig.tight_layout()
    fig.close()


    # Plot histogramm and best fit Gaussian
    fig, axs = plt.subplots(1,2)
    n = quantiles
    _, bins, _ = axs[0].hist(emg1, n, density=1)

    mu, std = scipy.stats.norm.fit(emg1)
    y = scipy.stats.norm.pdf(bins, mu, std)
    axs[0].plot(bins, y, linewidth=4)

    _, bins, _ = axs[1].hist(emg2, n, density=1)

    mu, std = scipy.stats.norm.fit(emg2)
    y = scipy.stats.norm.pdf(bins, mu, std)
    axs[1].plot(bins, y, linewidth=4)
    fig.suptitle("EMG biceps and triceps histogram vs best fit Gaussian", fontsize=22)
    fig.tight_layout()
    fig.close()

    # Plot qq-plot
    fig, axs = plt.subplots(1,2)
    qqplot(emg1, line='s', ax=axs[0])
    qqplot(emg2, line='s', ax=axs[1])
    fig.suptitle("QQ-Plots for normality checking", fontsize=22)
    fi.tight_layout()
    fig.close()

    # D' agostino normality test
    stat, p = normaltest(emg1)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

    stat, p = normaltest(emg2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')