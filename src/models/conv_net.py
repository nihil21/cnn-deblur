import typing

from tensorflow import keras

from ..utils.custom_metrics import *


class ConvNet:
    """Abstract class representing a generic Convolutional Neural Network"""

    def __init__(self):
        self.model = None

    def compile(
            self,
            lr: float = 1e-4,
            loss: str = 'mse',
            use_nesterov: bool = False
    ):

        loss_dict = dict({
            'mse': keras.losses.MeanSquaredError(),
            'mae': keras.losses.MeanAbsoluteError(),
            'logcosh': keras.losses.LogCosh()
        })

        metric_list = [
            ssim,
            psnr,
            'mse',
            'mae',
            'accuracy'
        ]

        if use_nesterov:
            optimizer = keras.optimizers.Nadam(learning_rate=lr)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=lr)

        self.model.compile(
            optimizer,
            loss=loss_dict[loss],
            metrics=metric_list
        )

    def fit(
            self,
            x: typing.Optional = None,
            y: typing.Optional = None,
            batch_size: int = 32,
            epochs: int = 1,
            steps_per_epoch: typing.Optional[int] = None,
            validation_data: typing.Optional = None,
            validation_steps: typing.Optional[int] = None,
            initial_epoch: int = 0,
            callbacks: typing.Optional[typing.List[keras.callbacks.Callback]] = None
    ):
        if y is not None:
            return self.model.fit(
                x,
                y,
                batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_data,
                initial_epoch=initial_epoch,
                callbacks=callbacks
            )
        else:
            return self.model.fit(
                x,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_data,
                validation_steps=validation_steps,
                initial_epoch=initial_epoch,
                callbacks=callbacks
            )

    def evaluate(
            self,
            x: typing.Optional = None,
            y: typing.Optional = None,
            batch_size: typing.Optional[int] = None,
            steps: typing.Optional[int] = None
    ):
        if y is not None:
            return self.model.evaluate(x, y, batch_size=batch_size, steps=steps)
        else:
            return self.model.evaluate(x, batch_size=batch_size, steps=steps)

    def predict(self, X):
        return self.model.predict(X)

    def summary(self):
        self.model.summary()

    def plot_model(self, path):
        keras.utils.plot_model(self.model, to_file=path, show_shapes=True)
