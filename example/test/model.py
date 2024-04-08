from pydantic_settings import BaseSettings
from pydantic import Field
from ml_pilot.proxy import SettingsProxy


class TestModelOptions(BaseSettings):
    input_shape: tuple = Field(
        (4, 4),
        description="Input shape of the model.",
    )


def create_test_model():
    import tensorflow as tf

    proxy: SettingsProxy = SettingsProxy()
    opts: TestModelOptions = proxy.get_settings("TestModelOptions")

    class Model:
        def __init__(self):
            self.model = self.get_model()

        def __getattr__(self, name):
            return getattr(self.model, name)

        def get_model(self) -> tf.keras.Model:
            shape = [None] + list(opts.input_shape)
            x = inputs = tf.keras.Input(shape)
            return tf.keras.Model(inputs, x, name="")

    return Model()
