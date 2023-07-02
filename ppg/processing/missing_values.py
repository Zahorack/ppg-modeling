import numpy as np
import pandas as pd

from ppg.enums import WavelengthIndex
from ppg.processing import Processor


class PolynomialInterpolation(Processor):
    """
    Polynomial interpolation to handle missing values in time series,
    supposing there are no significant gaps in data.
    """
    def process(self, data: np.ma.masked_array) -> np.array:
        assert data.shape[0] == len(WavelengthIndex), 'Invalid data format! Expecting masked_array with 3 channels.'

        return pd.DataFrame(data).interpolate(method='polynomial', order=5, axis=1).values
