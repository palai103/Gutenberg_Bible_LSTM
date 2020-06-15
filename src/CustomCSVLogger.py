import inspect
import time

import numpy as np
import six
from tensorflow.keras.callbacks import CSVLogger


class CustomCSVLogger(CSVLogger):

    def __init__(self, additional_columns, **kwargs):
        super(CustomCSVLogger, self).__init__(**kwargs)
        self.epoch_start = None
        self.lowest_val = 999.999
        self.epochs = 0
        self.additional_columns = additional_columns
        for key in self.additional_columns:
            if key != 'seconds':
                argspec = inspect.getargspec if six.PY2 else inspect.getfullargspec
                args = set(argspec(self.additional_columns[key]).args)
                assert 'epoch' in args, "keyword argument 'epoch' is missing"
                assert 'model' in args, "keyword argument 'model' is missing"

    def on_epoch_begin(self, epoch, logs=None):
        if 'seconds' in self.additional_columns:
            self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key in self.additional_columns:
            if key == 'seconds':
                now = time.time()
                logs[key] = np.round(now - self.epoch_start, 1)
            else:
                func = self.additional_columns[key]
                logs[key] = str(func(epoch=epoch, model=self.model))
        super(CustomCSVLogger, self).on_epoch_end(epoch, logs=logs)
