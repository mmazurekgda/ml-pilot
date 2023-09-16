from core.config import Config


def generate_model():
    import tensorflow as tf

    class Model:
        def __init__(self):
            self.model = self.get_model()

        def __getattr__(self, name):
            return getattr(self.model, name)

        def get_model(self) -> tf.keras.Model:
            config = Config()
            shape = [None] + list(config.input_shape)
            x = inputs = tf.keras.Input(shape)
            return tf.keras.Model(inputs, x, name="")

    return Model()
