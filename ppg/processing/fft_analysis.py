from typing import Optional

import numpy as np
from scipy.signal import stft, istft
from scipy.interpolate import interp1d

from ppg.common.utils import concatenate_time_series


def merge_by_stft_interpolation(s1: np.ndarray,
                                s2: np.ndarray,
                                segments_per_gap: int,
                                gap_size: int,
                                fs: int):
    """
    Attempt to model gap of missing values by interpolating stft frequency spectrum.

    This approach is wrong by design, since it is not trivial to interpolate Phase of STFT.
    STFT Phase can be estimated by iterative algorithms, like Griffin-Lim.
    """
    gap_start = len(s1) + 1
    gap_end = len(s1) + gap_size

    OVERLAP_SIZE = (gap_size // segments_per_gap)
    SEGMENT_SIZE = int(((gap_size - OVERLAP_SIZE) / segments_per_gap) + OVERLAP_SIZE)

    frequencies, times_before, Zxx_before = stft(s1, fs=fs, nperseg=SEGMENT_SIZE, noverlap=OVERLAP_SIZE, boundary=None, padded=False)
    _, times_after, Zxx_after = stft(s2, fs=fs, nperseg=SEGMENT_SIZE, noverlap=OVERLAP_SIZE, boundary=None, padded=False)

    # Identify dominant frequencies before and after the gap
    threshold = max(np.median(np.abs(Zxx_before[:, -1])), np.median(np.abs(Zxx_after[:, 0])))

    dominant_freq_before = frequencies[np.abs(Zxx_before[:, -1]) > threshold]
    dominant_freq_after = frequencies[np.abs(Zxx_after[:, 0]) > threshold]

    # Interpolate only the dominant frequencies
    Zxx_fill = np.zeros((len(frequencies), gap_end-gap_start), dtype=complex)
    for freq in set(dominant_freq_before).union(dominant_freq_after):
        i = np.where(frequencies == freq)[0][0]
        magnitude_before = np.abs(Zxx_before[i, -1])
        phase_before = np.angle(Zxx_before[i, -1])

        magnitude_after = np.abs(Zxx_after[i, 0])
        phase_after = np.angle(Zxx_after[i, 0])

        interp_mag = interp1d([times_before[-1], times_after[0]], [magnitude_before, magnitude_after], kind='linear')
        interp_phase = interp1d([times_before[-1], times_after[0]], [phase_before, phase_after], kind='linear')

        Zxx_fill[i, :] = interp_mag(np.linspace(times_before[-1], times_after[0], gap_end-gap_start)) * \
                         np.exp(1j * interp_phase(np.linspace(times_before[-1], times_after[0], gap_end-gap_start)))

    # Inverse STFT transform to obtain time series
    _, gap_fill = istft(Zxx_fill, fs=fs, nperseg=SEGMENT_SIZE, noverlap=OVERLAP_SIZE,
                        time_axis=-1, freq_axis=-2, boundary=None)

    x = concatenate_time_series(s1, gap_fill)
    x = concatenate_time_series(x, s2)

    return x


def griffin_lim_algorithm(magnitude_spectrogram: np.ndarray,
                          known_phase: np.ndarray,
                          known_phase_mask: np.ndarray,
                          n_iter: Optional[int] = 100,
                          nperseg: Optional[int] = 2048,
                          noverlap: Optional[int] = 512):
    """
    Griffin-Lim iterative algorithm to reconstruct time series from STFT magnitude spectrogram.

    "Signal estimation from modified short-time Fourier transform"
    https://typeset.io/pdf/signal-estimation-from-modified-short-time-fourier-transform-2wf0xylo3c.pdf

    Adjusted to follow scipy interface and argument names. Algorithm use know phase for initialization.
    """
    spectrogram_shape = magnitude_spectrogram.shape

    phase = np.random.uniform(-np.pi, np.pi, spectrogram_shape)
    phase = known_phase * known_phase_mask + phase * (1 - known_phase_mask)

    # Combine magnitude and phase to create initial complex spectrogram
    complex_spectrogram = magnitude_spectrogram * np.exp(1j * phase)

    # Initialize an array to store the signal
    y = np.zeros((spectrogram_shape[1] - 1) * (nperseg-noverlap) + nperseg)

    for i in range(n_iter):
        # Inverse STFT
        _, y = istft(complex_spectrogram, nperseg=nperseg, noverlap=noverlap)

        if i < n_iter - 1:
            # Re-calculate the STFT
            _, _, complex_spectrogram_next = stft(y, nperseg=nperseg, noverlap=noverlap)

            # Extract the phase and combine with the original magnitude
            phase_next = np.angle(complex_spectrogram_next)
            phase = known_phase * known_phase_mask + phase_next * (1 - known_phase_mask)
            complex_spectrogram = magnitude_spectrogram * np.exp(1j * phase)

    return y