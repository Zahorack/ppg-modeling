import logging
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import train_test_split

from ppg.common import config
from ppg.common.utils import sigmoid
from ppg.models import PPGModel


class AutoregressiveGapModel:

    _logger = logging.getLogger(__name__)

    def __init__(self,
                 regressor1: PPGModel,
                 regressor2: PPGModel,
                 s1: np.ndarray,
                 s2: np.ndarray,
                 gap_size: int,
                 fs: Optional[int] = config.DEFAULT_SAMPLING_FREQUENCY,
                 n_lags: Optional[int] = config.DEFAULT_N_LAGS,
                 rolling_windows: Optional[List[int]] = config.DEFAULT_ROLLING_WINDOWS,
                 nth_sampling: Optional[int] = None):
        """
        Uni-variate regression model capable of modeling wide gaps of missing data between two time series.

        Model is designed to handle and model signal shape, frequency, amplitude as well as long term features.
        Considering data before and after gap differ, model use sigmoid weighting to provide smooth transition of distributions.

        TODO: Model abstraction should support several regression models, like Neural Networks, etc.

        :param s1: Time series before gap
        :param s2: Time series after gap
        :param gap_size: Number of timestamps to be modeled
        :param n_lags: number of lagged values in time series to consider in regression
        :param rolling_windows: List of rolling window sizes used for aggregated and windowing metrics
        :param fs: sampling frequency
        """
        self.s1 = s1
        self.s2 = s2
        self.gap_size = gap_size
        self.n_lags = n_lags
        self.rolling_windows = rolling_windows
        self.fs = fs
        self.nth_sampling = nth_sampling

        if nth_sampling:
            self._logger.info('Time series features are down-sampled %sx', nth_sampling)
            self.s1 = pd.Series(self.s1).iloc[::nth_sampling].values
            self.s2 = pd.Series(self.s2).iloc[::nth_sampling].values
            self.gap_size = self.gap_size // nth_sampling
            self.fs = self.fs // nth_sampling

        self._regressor1 = regressor1
        self._regressor2 = regressor2

    def generate(self) -> np.ndarray:
        self._logger.info('Uni-variate autoregressive gap modeling started. Time steps to be modeled: %s', self.gap_size)

        # Define effective and symmetric size of data used for training
        data_size = int(min([len(self.s1), len(self.s2)]))
        if data_size < self.gap_size:
            self._logger.warning('No enough training data, gap is too large, results may be unsatisfactory.')

        x_1, y_1 = self.create_regression_data(self.s1[-data_size:])
        x_2, y_2 = self.create_regression_data(self.s2[:data_size])

        X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(x_1, y_1, test_size=0.2, random_state=42)
        X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(x_2, y_2, test_size=0.2, random_state=42)

        self._regressor1.fit(X_1_train, y_1_train)
        self._regressor2.fit(X_2_train, y_2_train)

        self._logger.info('Models training is done.')
        self._logger.info('Model(1) %s - before gap metrics: %s', self._regressor1.__class__.__name__,
                          self._evaluation_metrics(self._regressor1, X_1_test, y_1_test))
        self._logger.info('Model(2) %s - after gap metrics: %s', self._regressor1.__class__.__name__,
                          self._evaluation_metrics(self._regressor2, X_2_test, y_2_test))

        x_autoreg = self.s1[-max(self.rolling_windows + [self.n_lags]) - 1:]
        weights = sigmoid(np.linspace(-4, 4, self.gap_size))

        # Autoregressive gap modeling using sigmoid weighting of modeled data before and after the gap.
        predictions = []
        for i in tqdm(range(self.gap_size)):
            # Construct lag and long-term features
            x_win, _ = self.create_regression_data(np.append(x_autoreg, [-1]))
            x_win = x_win.tail(1)

            # Weight predictions to address smooth transition between models
            prediction_before = self._regressor1.predict(x_win)[0]
            prediction_after = self._regressor2.predict(x_win)[0]
            prediction = (1 - weights[i]) * prediction_before + weights[i] * prediction_after

            # Add predicted value to autoregressive buffer
            x_autoreg = np.append(x_autoreg, [prediction])
            predictions.append(prediction)

        self._logger.info('Gap modeled successfully!')

        if self.nth_sampling:
            self._logger.info('Reconstructed gap time series is up-sampled %sx and interpolated.', self.nth_sampling)
            up_sampled = pd.Series(predictions).reset_index(drop=True)
            up_sampled.index = up_sampled.index * self.nth_sampling
            up_sampled = up_sampled.reindex(range(self.gap_size * self.nth_sampling)).interpolate(method='polynomial', order=5)
            return up_sampled.values

        return np.array(predictions)

    def _regressive_smoothing(self, ts, start, end, iterations):
        """
        Postprocessing regression smoothing over existing time series
        TODO: retrain model on different moving features in each iteration
        """
        x_smooth = ts[:end]
        weights = sigmoid(np.linspace(-4, 4, end-start))

        for i in range(iterations):
            x, _ = self.create_regression_data(x_smooth)

            predictions_before = self._regressor1.predict(x)
            predictions_after = self._regressor2.predict(x)
            predictions = (1 - weights) * predictions_before + weights * predictions_after

            x_smooth = np.concatenate([ts[:end - len(predictions)], predictions])

        result = np.concatenate([ts[:start], x_smooth[-(end-start):], ts[end:]])
        assert len(result) == len(ts)

        return result

    @staticmethod
    def _evaluation_metrics(model, x, y) -> Dict:
        pred = model.predict(x)
        mape = np.mean(np.abs((y - pred) / np.abs(y)))

        return {
            'MAE:': metrics.mean_absolute_error(y, pred),
            'MSE': metrics.mean_squared_error(y, pred),
            'RMSE': np.sqrt(metrics.mean_squared_error(y, pred)),
            'MAPE': round(mape * 100, 2)
        }

    def create_regression_data(self,
                               data: np.ndarray,
                               shuffle: Optional[bool] = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Provided by time series data, function will output dataframes X - inputs, y - targets used for regression models.
        Function creates n_lags values for short-term regression and moving window features to handle long-term dependencies

        Note: Function can be computationally improved and optimized.
        It is fast enough for bulk dataset creation, but not for autoregressive use-case...

        :param data: np.ndarray time series
        :param shuffle: whether to randomize order of data samples
        :return: X inputs values, y - target values
        """
        assert self.n_lags > 2, 'n_lags should be greater than 2'

        df = pd.DataFrame({'lag_0': data})

        # Create lagged features
        for i in range(1, self.n_lags + 1):
            df[f'lag_{i}'] = df['lag_0'].shift(i)
        df[f'lag_1_diff'] = df[f'lag_1'].diff()

        # Create long-term features, using lag_1, since lag_0 is value to be predicted in current step
        min_rolling_size = min([self.n_lags*10, min(self.rolling_windows)])
        df = self._add_moving_features(df=df, column='lag_1', rolling_windows=self.rolling_windows,
                                       min_rolling_size=min_rolling_size)

        df = df.dropna().reset_index(drop=True)
        df = self._add_envelope_features(df=df, column='lag_1', fs=self.fs, upper=True, interpolate=False)
        df = self._add_envelope_features(df=df, column='lag_1', fs=self.fs, upper=False, interpolate=False)
        # df = self._add_envelope_features(df=df, column='lag_1', fs=self.fs, upper=True, interpolate=True)
        # df = self._add_envelope_features(df=df, column='lag_1', fs=self.fs, upper=False, interpolate=True)

        if shuffle:
            df = df.sample(frac=1)

        y = df['lag_0']
        X = df.drop('lag_0', axis=1)

        return X, y

    @staticmethod
    def _add_moving_features(df: pd.DataFrame,
                             column: str,
                             rolling_windows: List[int],
                             min_rolling_size: int) -> pd.DataFrame:
        """
        Construct moving standard deviation and moving average.
        """
        for w in rolling_windows or []:
            df[f'rolling_std_{w}'] = df[column].rolling(window=w, min_periods=min_rolling_size).std()
            df[f'rolling_mean_{w}'] = df[column].rolling(window=w, min_periods=min_rolling_size).mean()
            df[f'rolling_min_{w}'] = df[column].rolling(window=w, min_periods=min_rolling_size).min()
            df[f'rolling_max_{w}'] = df[column].rolling(window=w, min_periods=min_rolling_size).max()

        return df

    @staticmethod
    def _add_envelope_features(df: pd.DataFrame,
                               column: str,
                               fs: int,
                               upper: bool,
                               interpolate: bool) -> pd.DataFrame:
        """
        Compute peak envelope and forward-fill values over all time-steps to extract robust long-term differentiation.
        If interpolate is True, peak envelope are interpolated (spline) over all time-steps to extract smooth differentiation.

        Note: Also absolute envelope values of peaks are used for regression only if interpolate is False,
                since interpolated values itself confused model
        """
        peak_indices, _ = find_peaks(df[column].values if upper else -df[column].values, distance=int(fs // 3))
        feature_name = f'{"upper" if upper else "lower"}_envelope_{"spline" if interpolate else "peak"}'

        if interpolate:
            spline = UnivariateSpline(peak_indices, df[column].values[peak_indices], s=0)
            df[f'{feature_name}_diff'] = pd.Series(spline(np.arange(len(df)))).diff().fillna(method='ffill').bfill()
        else:
            df_feature = pd.DataFrame({feature_name: df[column].values[peak_indices]})
            df_feature[f'{feature_name}_diff'] = df_feature[feature_name].diff()
            df_feature.index = peak_indices
            df_feature = df_feature.reindex(range(len(df))).fillna(method='ffill').bfill()
            df = pd.concat([df, df_feature], axis=1)

        return df
