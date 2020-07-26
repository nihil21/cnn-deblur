from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from model.custom_losses_metrics import *
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, LogCosh
from typing import List, Optional


class ConvNet:
    """Abstract class representing a generic Convolutional Neural Network"""

    def __init__(self):
        self.model = None

    def compile(self,
                lr: Optional[float] = 1e-4,
                loss: Optional[str] = 'mse'):

        loss_dict = dict({
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError(),
            'logcosh': LogCosh()
        })

        metric_list = [ssim,
                       psnr,
                       'mse',
                       'mae',
                       'accuracy']

        self.model.compile(Adam(learning_rate=lr),
                           loss=loss_dict[loss],
                           metrics=metric_list)

    def fit(self,
            x: Optional = None,
            y: Optional = None,
            batch_size: Optional[int] = 32,
            epochs: Optional[int] = 1,
            steps_per_epoch: Optional[int] = None,
            validation_data: Optional = None,
            validation_steps: Optional[int] = None,
            initial_epoch: Optional[int] = 0,
            callbacks: Optional[List[Callback]] = None):
        if y is not None:
            return self.model.fit(x, y,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=validation_data,
                                  initial_epoch=initial_epoch,
                                  callbacks=callbacks)
        else:
            return self.model.fit(x,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=validation_data,
                                  validation_steps=validation_steps,
                                  initial_epoch=initial_epoch,
                                  callbacks=callbacks)

    def evaluate(self,
                 x: Optional = None,
                 y: Optional = None,
                 batch_size: Optional[int] = None,
                 steps: Optional[int] = None):
        if y is not None:
            return self.model.evaluate(x, y, batch_size=batch_size, steps=steps)
        else:
            return self.model.evaluate(x, batch_size=batch_size, steps=steps)

    def predict(self, X):
        return self.model.predict(X)

    def summary(self):
        self.model.summary()

    def plot_model(self, path):
        plot_model(self.model, to_file=path, show_shapes=True)
