import pandas as pd
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import numpy as np
from scipy.signal import correlate, stft


def load_data(path: str) -> Dataset:
    """
    Load time series data from netCDF4 file
    """
    return Dataset(path, 'r', format="NETCDF4")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _get_correlation_phase_shift(s1: np.ndarray, s2: np.ndarray) -> int:
    """
    Compute optimal phase shift between two time series
    """
    correlations = correlate(s1, s2, method='direct')
    shift = np.argmax(correlations) - (len(s1) - 1)
    return int(shift)


def get_alignment_phase_shift(s1: np.ndarray,
                              s2: np.ndarray,
                              precision: float = 0.01) -> int:
    """
    Provided by two periodic time series, function will return number of time steps required to be s1 shifted left to
    align with s2 in specified precision.
    """
    threshold = np.abs((max(s1) - min(s1)) * precision)
    alignment_value = s2[0]
    alignment_gradient_sign = np.sign(np.diff(s2[:2]))

    shift_index = 0
    for i, v in enumerate(reversed(s1)):
        difference = np.abs(alignment_value - v)
        actual_gradient_sign = np.sign(np.diff(s1[-i - 2:-i]))
        if difference <= threshold and alignment_gradient_sign == actual_gradient_sign:
            shift_index = i
            break

    return shift_index


def concatenate_time_series(s1: np.ndarray,
                            s2: np.ndarray,
                            precision: float = 0.01,
                            max_correlation_phase_shift: int = 200) -> np.ndarray:
    """
    Concatenate two time series by considering phase shift and derivation sign.
    Additional fine alignment is applied to smooth transitions between signals.
    The final time series will have different-shorter length.
    """
    # Phase correlation alignment
    shift = _get_correlation_phase_shift(s1, s2)
    if np.abs(shift) < max_correlation_phase_shift:
        if shift < 0:
            s1 = s1[:shift]
        else:
            s2 = s2[shift:]

    # Absolute value & gradient alignment
    fine_shift_index = get_alignment_phase_shift(s1, s2, precision=precision)

    # Mering and final interpolations to fill possible missing values in interfaces
    result = np.concatenate([s1[:len(s1) - fine_shift_index], s2])
    result = pd.Series(result).interpolate(method='polynomial', order=5).values

    return result


def shift_and_concatenate_time_series(s1: np.ndarray,
                                      s2: np.ndarray,
                                      shift_steps: int,
                                      flex_steps: int = 3) -> np.ndarray:
    """
    Shift ts1 left by number of shift_steps, concatenate shifted time series by creating flexible np.NaN window to be
    interpolated afterwards.
    """
    s1 = s1[:-shift_steps]
    result = np.concatenate([s1[:-flex_steps], [np.NAN for _ in range(flex_steps * 2)], s2[flex_steps:]])
    result = pd.Series(result).interpolate(method='polynomial', order=3).values

    return result


def plot_results(data: pd.DataFrame, gap_start: int, gap_end: int, fs: int):
    OVERLAP_SIZE = 1000
    SEGMENT_SIZE = 2000

    fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(15, 80),
                           gridspec_kw={'height_ratios': [1, 1.5, 1, 1.5, 1, 1.5]})

    colors = ['tab:blue', 'tab:red', 'tab:purple']
    for i, (col, color) in enumerate(zip(data.columns, colors)):
        x = data[col].values

        frequencies, times, Zxx = stft(x, fs=fs, nperseg=SEGMENT_SIZE, noverlap=OVERLAP_SIZE, boundary=None,
                                       padded=False)

        ax[i * 2].plot(x, color=color)
        ax[i * 2].axvline(x=gap_start, label='gap_start', color='purple')
        ax[i * 2].axvline(x=gap_end, label='gap_end', color='red')
        ax[i * 2].set_title(f'Channel {col}')
        ax[i * 2].set_xlim(0, len(x))
        ax[i * 2].set_xlabel('Timestep')
        ax[i * 2].set_ylabel('Amplitude')
        ax[i * 2].legend(loc='upper left')

        ax[i * 2 + 1].pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
        ax[i * 2 + 1].set_ylim([0, 10])
        ax[i * 2 + 1].axvline(x=gap_start / fs, label='gap_start', color='purple')
        ax[i * 2 + 1].axvline(x=gap_end / fs, label='gap_end', color='red')
        ax[i * 2 + 1].set_ylabel('Frequency [Hz]')
        ax[i * 2 + 1].legend(loc='upper right')
        ax[i * 2 + 1].margins(x=10)

    plt.show()
