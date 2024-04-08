from ml_pilot.proxy import SettingsProxy
from ml_pilot.actions.tensorflow.train import train

from dataloader import generate_tfrecord_decoder
from loss import generate_loss
from ml_pilot.actions.tensorflow.dataloader import generate_tfrecord_dataloader


def train():
    dataset = None
    val_dataset = None

    proxy: SettingsProxy = SettingsProxy()

    if config.dataloader_type == "tfrecord":
        dataset = generate_tfrecord_dataloader(
            generate_tfrecord_decoder(),
            "training",
        )()
        val_dataset = generate_tfrecord_dataloader(
            generate_tfrecord_decoder(),
            "validation",
        )()
    else:
        raise NotImplementedError(
            f"Dataloader type '{config.dataloader_type}' not implemented."
        )
    train(proxy.get_model(), generate_loss, dataset, val_dataset)