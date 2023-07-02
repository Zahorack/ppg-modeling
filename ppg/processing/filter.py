import numpy as np

from scipy import signal
from typing import List, Union, Tuple, Optional

from ppg.enums import WavelengthIndex
from ppg.processing import Processor


class BandPassPaddingFilter(Processor):
    """
    Basic BandPass filter with padding to address Ringing effect (Gibbs phenomenon).

    Used to filter low-frequency DC component and high-frequency noises.
    """
    def __init__(self,
                 order: int,
                 sampling_frequency: float,
                 cutoff_frequencies: Union[Tuple, List],
                 padding_size: int,
                 custom_padding: Optional[bool] = True):
        self.order = order
        self.fs = sampling_frequency
        self.cutoffs = cutoff_frequencies
        self.padding_size = padding_size
        self.custom_padding = custom_padding

    def process(self, data: np.array) -> np.array:

        assert data.shape[0] == len(WavelengthIndex), 'Invalid data format! Expecting array with 3 channels.'

        b, a = signal.butter(self.order,
                             Wn=self.cutoffs,
                             btype='bandpass',
                             fs=self.fs)

        if self.custom_padding:
            # Later I've discovered padding settings in filtfilt itself,but this custom one still worked better
            filtered_data = []
            for channel in range(len(data)):
                channel_data = data[channel]

                pre_padding = channel_data[:self.padding_size][::-1]
                post_padding = channel_data[-self.padding_size:][::-1]

                channel_data = np.concatenate((pre_padding, channel_data, post_padding))
                channel_data = signal.filtfilt(b, a, channel_data)[self.padding_size:-self.padding_size]

                filtered_data.append(channel_data)

            return np.array(filtered_data)

        else:
            signal.filtfilt(b, a, data, method='pad', padtype='even')
