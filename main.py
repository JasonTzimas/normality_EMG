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
from normality_function import normality_fun
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
import statsmodels
from scipy.interpolate import interp1d
from scipy import signal
from scipy.fft import fftshift

# Import mat files
mat_angle = scipy.io.loadmat('data/angle_40bpm.mat')
mat_emg1 = scipy.io.loadmat('data/emg1_40bpm.mat')
mat_emg2 = scipy.io.loadmat('data/emg2_40bpm.mat')

# Convert them to numpy arrays
angle = np.array(mat_angle['angle1'])
emg1 = np.array(mat_emg1['emg1'])
emg2 = np.array(mat_emg2['emg2'])
T = 0.001
N = emg1.shape[0]

# Normalize data
scaler = preprocessing.StandardScaler().fit(emg1)
emg1 = scaler.transform(emg1)
scaler = preprocessing.StandardScaler().fit(emg2)
emg2 = scaler.transform(emg2)

# Test normality
n = 1000
#normality_fun(emg1, emg2, n)

# Perform tests in a smaller quantity
emg1_small = emg1[int(0.2*N):int(0.24*N)]
emg2_small = emg2[int(0.2*N):int(0.24*N)]

# Test normality on smaller sample
n = 50
#normality_fun(emg1_small, emg2_small, n)

# Plot acf
lag = 100
sm.graphics.tsa.plot_acf(emg1, lags=lag)
plt.show()
sm.graphics.tsa.plot_acf(emg2, lags=lag)
plt.show()
sm.graphics.tsa.plot_acf(emg1_small, lags=lag)
plt.show()
sm.graphics.tsa.plot_acf(emg2_small, lags=lag)
plt.show()

# Plot moments
mom1 = np.empty((20))
mom2 = np.empty((20))
mom1_small = np.empty((20))
mom2_small = np.empty((20))
for i in range(20):
    mom1[i] = stats.moment(emg1, i)
    mom2[i] = stats.moment(emg2, i)
    mom1_small[i] = stats.moment(emg1_small, i)
    mom2_small[i] = stats.moment(emg2_small, i)

print(mom1)
fig, axs = plt.subplots(4, 1)
axs[0].bar(np.arange(20), np.log(mom1))
axs[1].bar(np.arange(20), np.log(mom2))
axs[2].bar(np.arange(20), np.log(mom1_small))
axs[3].bar(np.arange(20), np.log(mom2_small))
plt.show()

# Augmented Dickey-Fuller stationarity test
#statistic = statsmodels.tsa.stattools.adfuller(emg1,
           #maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)
#print(" Dickey Fuller test for emg biceps: ", statistic)
#statistic = statsmodels.tsa.stattools.adfuller(emg2,
           #maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)
#print(" Dickey Fuller test for emg biceps: ", statistic)

# Spectrogram
# Create Interpolated version of emg1
old_indices = np.arange(0, emg1.shape[0])
new_length = 1 * emg1.shape[0]
new_indices = np.linspace(0, emg1.shape[0] - 1, new_length)
inter = interp1d(old_indices, emg1, axis=0)
emg1_interp = np.array(inter(new_indices))
inter = interp1d(old_indices, emg2, axis=0)
emg2_interp = np.array(inter(new_indices))
fs = 1 * 1000
print(emg1_interp.shape)
[f, t, Sxx] = signal.spectrogram(emg1_interp.T, fs, nperseg=int(0.1 * emg1_interp.shape[0]), noverlap=int(0.98*0.1 * emg1_interp.shape[0]))
plt.pcolormesh(t, f, Sxx[0,:,:], shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()