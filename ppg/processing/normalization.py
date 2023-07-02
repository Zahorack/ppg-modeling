import numpy as np

from ppg.enums import WavelengthIndex
from ppg.processing import Processor


class ZscoreNormalization(Processor):
    """
    Preform Zscore normalization by subtracting the mean and dividing by the standard deviation
    """
    def process(self, data: np.array) -> np.array:

        assert data.shape[0] == len(WavelengthIndex), 'Invalid data format! Expecting masked_array with 3 channels.'

        for i in range(data.shape[0]):
            mean = np.mean(data[i])
            std_dev = np.std(data[i])

            if std_dev != 0:
                data[i, :] = (data[i] - mean) / std_dev
            else:
                data[i, :] = data[i] - mean

        return data


class MinMaxScaler(Processor):
    """
    Perform Min-Max scaling on the data. It transforms features by scaling each feature to a given range.
    This range can be specified by the user, and default is 0 to 1.
    """
    def __init__(self, scale_min=0, scale_max=1):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def process(self, data: np.array) -> np.array:

        assert data.shape[0] == len(WavelengthIndex), 'Invalid data format! Expecting masked_array with 3 channels.'

        for i in range(data.shape[0]):
            data_min = np.min(data[i])
            data_max = np.max(data[i])

            if (data_max - data_min) != 0:
                data[i, :] = (data[i] - data_min) / (data_max - data_min) * (self.scale_max - self.scale_min) + self.scale_min
            else:
                data[i, :] = data[i] - data_min

        return data
