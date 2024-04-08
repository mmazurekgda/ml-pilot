import string
import random

from ml_pilot.proxy import SettingsProxy
from model import TestModelOptions


def generate_tfrecord_decoder():
    import tensorflow as tf

    proxy: SettingsProxy = SettingsProxy()
    config: TestModelOptions = proxy.get_settings("TestModelOptions")

    def decoder(dataset):
        parsed = tf.io.parse_single_example(
            dataset,
            {
                "x": tf.io.FixedLenSequenceFeature(
                    [], dtype=tf.float32, allow_missing=True
                ),
                "y": tf.io.FixedLenSequenceFeature(
                    [], dtype=tf.float32, allow_missing=True
                ),
            },
        )
        shape = [-1] + list(config.input_shape)
        return (
            tf.reshape(parsed["x"], shape),
            tf.reshape(parsed["y"], shape),
        )

    return decoder


def generate_tfrecord_encoder(files_no: int, samples_no: int):
    import tensorflow as tf

    proxy: SettingsProxy = SettingsProxy()
    config: TestModelOptions = proxy.get_settings("TestModelOptions")

    def encoder():
        counter = 0
        while counter < files_no:
            shape = [samples_no] + list(config.input_shape)
            x = tf.reshape(tf.ones(shape), -1)
            y = tf.reshape(tf.ones(shape), -1)
            feature = {
                "x": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                "y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            file_name = "".join(random.choices(string.ascii_lowercase, k=5))
            file_name += ".tf"
            yield file_name, [example]
            counter += 1

    return encoder
