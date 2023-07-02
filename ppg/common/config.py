from pathlib import Path


class Config:
    ROOT_PATH = Path(__file__).parent.parent.absolute()
    PROJECT_PATH = ROOT_PATH.parent

    DATA_DIR = PROJECT_PATH / 'data'

    DEFAULT_TS1_FILE = 'ppg_curve_0.nc'
    DEFAULT_TS2_FILE = 'ppg_curve_1.nc'

    # Preprocessing
    FILTER_PADDING_MULTIPLIER = 10
    DEFAULT_SAMPLING_FREQUENCY = 100
    DEFAULT_FILTER_ORDER = 5
    DEFAULT_BANDPASS_CUTOFF_FREQUENCIES = (0.15, 10)

    # Regression Modeling
    DEFAULT_GAP_SIZE = 3000
    DEFAULT_N_LAGS = 3
    DEFAULT_NTH_SAMPLING = 5
    DEFAULT_ROLLING_WINDOWS = [10, 50, 150, 500, 1000]
    DEFAULT_N_ESTIMATORS = 200
    DEFAULT_MAX_DEPTH = 12


