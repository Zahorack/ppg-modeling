from typing import Optional

import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor

from ppg.common import config
from ppg.models import PPGModel


class NNRegressor(PPGModel):

    def __init__(self):
        self._model = keras.models.Sequential([
            keras.layers.Normalization(),
            keras.layers.Dense(units=32, activation='relu'),
            keras.layers.Dropout(rate=0.1),
            keras.layers.Dense(units=16, activation='relu'),
            keras.layers.Dropout(rate=0.1),
            keras.layers.Dense(units=1, activation='linear'),
        ])
        self._model.compile(loss='mean_absolute_error', optimizer='adam')

    def fit(self, x, y):
        self._model.fit(x, y,
                        epochs=100,
                        verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)])

    def predict(self, x):
        return self._model.predict(x, verbose=0).reshape(-1)


class RFRegressor(PPGModel):

    def __init__(self,
                 n_estimators: Optional[int] = config.DEFAULT_N_ESTIMATORS,
                 max_depth: Optional[int] = config.DEFAULT_MAX_DEPTH,
                 n_jobs: Optional[int] = 6):

        self._model = RandomForestRegressor(n_estimators=n_estimators,
                                            random_state=42,
                                            n_jobs=n_jobs,
                                            max_depth=max_depth)

    def fit(self, x, y):
        self._model.fit(x, y)

    def predict(self, x):
        return self._model.predict(x)
