import numpy as np
import xarray as xr

def classify_threshold(traces, threshold, rolling=None, window_size=1):
    classification = traces > threshold
    if rolling is not None and rolling != '':
        classification = classification.astype(int).rolling(frame=window_size, center=True, min_periods=1)
        classification = getattr(classification, rolling)(classification).astype(bool)
    classification = xr.DataArray(classification, dims=('molecule', 'frame'), name='classification')
    return classification

#TODO: Add usage of the functions below to File

def trace_selection_threshold(traces, threshold):
    classification = classify_threshold(traces, threshold, name='')
    return classification.all(dim='frame')



def rolling_correlation(traces, rolling_dim='frame', correlation_dim='channel', window=10):
    windows = traces.rolling(dim={rolling_dim: window}, center=True, min_periods=1).construct(window_dim='section', stride=1, keep_attrs=None)

    mean_windows = windows.mean('section')
    windows_minus_mean = windows-mean_windows

    a = windows_minus_mean.prod(correlation_dim, skipna=False).sum('section')
    b = (windows_minus_mean**2).sum('section').prod(correlation_dim)**(1/2)
    p = a/b

    return p

def classify_correlation(traces, rolling_dim='frame', correlation_dim='channel', window=10, rolling_mean_window=10, threshold=0.75):
    rc = rolling_correlation(traces, rolling_dim=rolling_dim, correlation_dim=correlation_dim, window=window)
    rcm = rc.rolling(dim={rolling_dim: rolling_mean_window}, center=True, min_periods=1).mean()
    classification = (rcm > threshold).astype(int).rolling(dim={rolling_dim: rolling_mean_window}, center=True, min_periods=1).max()
    classification.name = 'classification'
    return classification


def classify_anticorrelation(traces, rolling_dim='frame', correlation_dim='channel', window=10, rolling_mean_window=10, threshold=-0.75):
    rc = rolling_correlation(traces, rolling_dim=rolling_dim, correlation_dim=correlation_dim, window=window)
    rcm = rc.rolling(dim={rolling_dim: rolling_mean_window}, center=True, min_periods=1).mean() # To smooth out variations
    classification = (rcm < threshold).astype(int).rolling(dim={rolling_dim: rolling_mean_window}, center=True, min_periods=1).max() # To widen the window
    classification.name = 'classification'
    return classification