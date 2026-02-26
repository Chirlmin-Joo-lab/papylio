"""Simple trace classification utilities.

Provides threshold, correlation-based, and rolling-window classifiers for
labeling trace segments or entire traces (per-molecule) according to various
criteria used in single-molecule analysis.
"""

import numpy as np
import xarray as xr


def classify_threshold(traces, threshold, rolling=None, window_size=1):
    """Classify traces by intensity threshold with optional rolling aggregation.

    Parameters
    ----------
    traces : xr.DataArray or np.ndarray
        Trace data with dimensions (molecule, frame) or equivalent
    threshold : float
        Threshold value for classification (trace > threshold -> True)
    rolling : str or None
        Name of a pandas-like rolling aggregation method (e.g., 'mean', 'max');
        if provided, apply rolling aggregation over time
    window_size : int, optional
        Window size for rolling aggregation (default: 1)

    Returns
    -------
    xr.DataArray
        Boolean DataArray named 'classification' with dims ('molecule', 'frame')
    """
    classification = traces > threshold
    if rolling is not None and rolling != '':
        classification = classification.astype(int).rolling(frame=window_size, center=True, min_periods=1)
        classification = getattr(classification, rolling)(classification).astype(bool)
    classification = xr.DataArray(classification, dims=('molecule', 'frame'), name='classification')
    return classification

#TODO: Add usage of the functions below to File

def trace_selection_threshold(traces, threshold):
    """Select traces where the entire trace exceeds a threshold.

    Parameters
    ----------
    traces : xr.DataArray
        Traces with dims ('molecule', 'frame')
    threshold : float
        Threshold value

    Returns
    -------
    xr.DataArray
        Boolean selection per molecule (True if the full trace meets threshold)
    """
    classification = classify_threshold(traces, threshold, name='')
    return classification.all(dim='frame')



def rolling_correlation(traces, rolling_dim='frame', correlation_dim='channel', window=10):
    """Compute rolling (windowed) correlation across a specified dimension.

    Parameters
    ----------
    traces : xr.DataArray
        Multi-dimensional traces, must contain rolling_dim and correlation_dim
    rolling_dim : str, optional
        Dimension name to roll over (default: 'frame')
    correlation_dim : str, optional
        Dimension name over which to compute correlation (default: 'channel')
    window : int, optional
        Window size for rolling computation (default: 10)

    Returns
    -------
    xr.DataArray
        Rolling correlation values with same leading coordinates as input
    """
    windows = traces.rolling(dim={rolling_dim: window}, center=True, min_periods=1).construct(window_dim='section', stride=1, keep_attrs=None)

    mean_windows = windows.mean('section')
    windows_minus_mean = windows-mean_windows

    a = windows_minus_mean.prod(correlation_dim, skipna=False).sum('section')
    b = (windows_minus_mean**2).sum('section').prod(correlation_dim)**(1/2)
    p = a/b

    return p

def classify_correlation(traces, rolling_dim='frame', correlation_dim='channel', window=10, rolling_mean_window=10, threshold=0.75):
    """Classify periods with high positive correlation across channels.

    Parameters
    ----------
    traces : xr.DataArray
        Input traces
    rolling_dim : str
        Dimension to roll over (default 'frame')
    correlation_dim : str
        Dimension to compute correlation across (default 'channel')
    window : int
        Window size for local correlation (default 10)
    rolling_mean_window : int
        Window size for smoothing correlation (default 10)
    threshold : float
        Correlation threshold for classification (default 0.75)

    Returns
    -------
    xr.DataArray
        Integer DataArray named 'classification' marking correlated segments
    """
    rc = rolling_correlation(traces, rolling_dim=rolling_dim, correlation_dim=correlation_dim, window=window)
    rcm = rc.rolling(dim={rolling_dim: rolling_mean_window}, center=True, min_periods=1).mean()
    classification = (rcm > threshold).astype(int).rolling(dim={rolling_dim: rolling_mean_window}, center=True, min_periods=1).max()
    classification.name = 'classification'
    return classification


def classify_anticorrelation(traces, rolling_dim='frame', correlation_dim='channel', window=10, rolling_mean_window=10, threshold=-0.75):
    """Classify periods with strong negative correlation across channels.

    Parameters
    ----------
    traces : xr.DataArray
        Input traces
    rolling_dim : str
        Dimension to roll over (default 'frame')
    correlation_dim : str
        Dimension to compute correlation across (default 'channel')
    window : int
        Window size for local correlation (default 10)
    rolling_mean_window : int
        Window size for smoothing correlation (default 10)
    threshold : float
        Correlation threshold for classification (default -0.75)

    Returns
    -------
    xr.DataArray
        Integer DataArray named 'classification' marking anticorrelated segments
    """
    rc = rolling_correlation(traces, rolling_dim=rolling_dim, correlation_dim=correlation_dim, window=window)
    rcm = rc.rolling(dim={rolling_dim: rolling_mean_window}, center=True, min_periods=1).mean() # To smooth out variations
    classification = (rcm < threshold).astype(int).rolling(dim={rolling_dim: rolling_mean_window}, center=True, min_periods=1).max() # To widen the window
    classification.name = 'classification'
    return classification