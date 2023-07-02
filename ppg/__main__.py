from datetime import datetime, timedelta

import click
import logging
import numpy as np
import pandas as pd

from ppg.common import config
from ppg.common.utils import load_data, plot_results, get_alignment_phase_shift, shift_and_concatenate_time_series
from ppg.enums import DataColumn, WavelengthIndex
from ppg.models.regression import RFRegressor, NNRegressor
from ppg.processing import Pipeline
from ppg.processing.filter import BandPassPaddingFilter
from ppg.processing.missing_values import PolynomialInterpolation
from ppg.processing.normalization import ZscoreNormalization, MinMaxScaler

from ppg.models.autoregressive_gap_model import AutoregressiveGapModel

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@click.command()
@click.argument('ts1', type=click.Path(exists=True), required=False, nargs=1)
@click.argument('ts2', type=click.Path(exists=True), required=False, nargs=1)
@click.option('--fs', type=float, required=False, default=None, help='Sampling frequency')
@click.option('--out', type=click.Path(exists=False), required=False, default=None, help='Output parquet file')
@click.option('--gap_size', type=int, required=False, default=None, help='Number of gap time steps')
def merge(ts1: str,
          ts2: str,
          fs: int,
          out: str,
          gap_size: int):
    """
    Model gap between and merge two PPG time series

    :param fs: Sampling frequency
    :param ts1: Path to first PPG time series file
    :param ts2: Path to second PPG time series file
    :param out: Output file path where merged PPG time series have to be stored
    :param gap_size: Number of time steps considering as gap to be modeled
    """
    _logger.info('Command merge executed.')

    if not fs:
        _logger.warning(f'Using default sampling frequency: %s', config.DEFAULT_SAMPLING_FREQUENCY)
        fs = config.DEFAULT_SAMPLING_FREQUENCY

    if not ts1 and ts2:
        _logger.warning('Using default PPG time series files.')

    ts1 = load_data(ts1 or config.DATA_DIR / config.DEFAULT_TS1_FILE)
    ts2 = load_data(ts2 or config.DATA_DIR / config.DEFAULT_TS2_FILE)

    if not gap_size:
        try:
            ts2_duration_sec = max(ts2[DataColumn.TIME][:])
            ts1_start = datetime.fromisoformat(ts1[DataColumn.PPG_CURVE].meas_datetime)
            ts2_start = datetime.fromisoformat(ts2[DataColumn.PPG_CURVE].meas_datetime)
            gap_size = int((ts1_start - ts2_start - timedelta(seconds=ts2_duration_sec)).total_seconds() * fs)
            if gap_size < 1:
                raise Exception('Time series are not measured in order.')

            _logger.debug('Gap size recognized: %s time steps', gap_size)

        except Exception as e:
            gap_size = config.DEFAULT_GAP_SIZE
            _logger.warning('Can\'t parse exact gap duration, default size will be used: %s s', gap_size, exc_info=e)

    preprocessing = Pipeline([
        PolynomialInterpolation(),
        ZscoreNormalization(),
        BandPassPaddingFilter(order=config.DEFAULT_FILTER_ORDER,
                              sampling_frequency=fs,
                              cutoff_frequencies=config.DEFAULT_BANDPASS_CUTOFF_FREQUENCIES,
                              padding_size=fs * config.FILTER_PADDING_MULTIPLIER),
    ])
    ts1_data = preprocessing.process(ts1[DataColumn.PPG_CURVE][:])
    ts2_data = preprocessing.process(ts2[DataColumn.PPG_CURVE][:])

    # Uni-variate autoregressive gap modeling
    if config.DEFAULT_NTH_SAMPLING:
        _logger.info(f'Time series will be modeled down-sampled {config.DEFAULT_NTH_SAMPLING}x')

    gaps = {}
    phase_shifts = []
    for channel in WavelengthIndex:
        _logger.info('Modeling gap for %s channel.', channel._name_)
        regressor1 = RFRegressor()
        regressor2 = RFRegressor()
        model = AutoregressiveGapModel(regressor1=regressor1, regressor2=regressor2, s1=ts1_data[channel], s2=ts2_data[channel],
                                       n_lags=config.DEFAULT_N_LAGS, gap_size=gap_size, fs=fs,
                                       nth_sampling=config.DEFAULT_NTH_SAMPLING)
        gap = model.generate()
        phase_shifts.append(get_alignment_phase_shift(gap, ts2_data[channel]))
        gaps[channel] = gap

    merged = []
    phase_shift = int(np.mean(phase_shifts))
    _logger.info('To address phase shift alignment after gap, all channels are shifted left %s time steps.', phase_shift)
    for channel in WavelengthIndex:
        merged.append(shift_and_concatenate_time_series(
            s1=np.concatenate([ts1_data[channel], gaps[channel]]),
            s2=ts2_data[channel],
            shift_steps=phase_shift,
            flex_steps=5
        ))

    postprocessing = Pipeline([
        MinMaxScaler()
    ])
    df_merged = pd.DataFrame(postprocessing.process(np.array(merged)).T, columns=[c._name_ for c in WavelengthIndex])

    try:
        df_merged.to_parquet(out or (config.DATA_DIR / 'output' / 'merged.parquet'))
    except ImportError as e:
        _logger.error('Can\'t export merged time series.', exc_info=e)

    plot_results(df_merged, gap_start=ts1[DataColumn.PPG_CURVE].shape[1],
                 gap_end=(df_merged.shape[0] - ts2[DataColumn.PPG_CURVE].shape[1]), fs=fs)


cli.add_command(merge)

if __name__ == '__main__':
    cli()
