import click
import os
from core.constants import PROJECT_NAME

GENERAL_OPTIONS = {
    "experiment_name": {
        "default": PROJECT_NAME,
        "help": "Name of the experiment.",
        "type": str,
    },
    "run_number": {
        "default": None,
        "help": "Run number / id. Preferably unique."
        "If not provided, it will be generated automatically "
        "based on the current time.",
        "type": str,
    },
    "config_file": {
        "default": None,
        "help": "Preload the configuration file",
        "type": click.Path(exists=True),
    },
}

TRAINING_OPTIONS = {
    "learning_rate": {
        "default": 1e-4,
        "help": "Learning rate",
        "type": float,
    },
    "epochs": {
        "default": 1,
        "help": "Number of epochs. If not provided, "
        "the model will be trained until the "
        "validation loss stops decreasing. "
        "This must be set with callbacks.",
        "type": click.IntRange(min=1),
    },
    "batch_size": {
        "default": 1,
        "help": "Batch size",
        "type": click.IntRange(min=1),
    },
}

TRAINING_STANDARD_CALLBACKS_OPTIONS = {
    "reduce_lr": {
        "default": False,
        "help": "Reduce learning rate on plateau",
        "type": bool,
    },
    "reduce_lr_verbosity": {
        "default": 1,
        "help": "Reduce learning rate on plateau verbosity",
        "type": int,
    },
    "reduce_lr_patience": {
        "default": 10,
        "help": "Reduce learning rate on plateau patience",
        "type": click.IntRange(min=1),
    },
    "reduce_lr_cooldown": {
        "default": 10,
        "help": "Reduce learning rate on plateau cooldown",
        "type": click.IntRange(min=0),
    },
    "early_stopping": {
        "default": False,
        "help": "Early stopping",
        "type": bool,
    },
    "early_stopping_verbosity": {
        "default": 1,
        "help": "Early stopping verbosity",
        "type": int,
    },
    "early_stopping_patience": {
        "default": 30,
        "help": "Early stopping patience",
        "type": click.IntRange(min=1),
    },
    "early_stopping_min_delta": {
        "default": 1e-6,
        "help": "Early stopping min delta",
        "type": float,
    },
    "early_stopping_restore": {
        "default": True,
        "help": "Early stopping restore best weights",
        "type": bool,
    },
    "model_checkpoint": {
        "default": False,
        "help": "Model checkpoint",
        "type": bool,
    },
    "model_checkpoint_out_weight_file": {
        "default": "model_with_weights.tf",
        "help": "Model checkpoint output weight file",
        "type": str,
    },
    "model_checkpoint_verbosity": {
        "default": 1,
        "help": "Model checkpoint verbosity",
        "type": int,
    },
}

TRAINING_TENSORBOARD_OPTIONS = {
    "tensorboard": {
        "default": False,
        "help": "Tensorboard",
        "type": bool,
    },
    "tensorboard_log_dir_name": {
        "default": "tensorboard",
        "help": "Tensorboard log dir",
        "type": str,
    },
    "tensorboard_histogram_freq": {
        "default": 1,
        "help": "Tensorboard histogram freq",
        "type": click.IntRange(min=1),
    },
    "tensorboard_write_graph": {
        "default": True,
        "help": "Tensorboard write graph",
        "type": bool,
    },
    "tensorboard_write_images": {
        "default": False,
        "help": "Tensorboard write images",
        "type": bool,
    },
    "tensorboard_update_freq": {
        "default": "epoch",
        "help": "Tensorboard update freq",
        "type": click.Choice(
            [
                "batch",
                "epoch",
            ]
        ),
    },
    "tensorboard_profile_batch": {
        "default": 0,
        "help": "Tensorboard profile batch",
        "type": click.IntRange(min=0),
    },
    "tensorboard_embeddings_freq": {
        "default": 1,
        "help": "Tensorboard embeddings freq",
        "type": click.IntRange(min=0),
    },
    "tensorboard_embeddings_metadata": {
        "default": None,
        "help": "Tensorboard embeddings metadata",
        "type": str,
    },
}

DATA_OPTIONS = {
    "dataloader_type": {
        "default": "tfrecord",
        "help": "Dataloader type",
        "type": click.Choice(
            [
                "tfrecord",
                "custom",
            ]
        ),
    },
    "tfrecord_training_files": {
        "default": None,
        "help": "TFRecord training files. "
        "To be used with the TFRecord dataloader.",
        "type": click.Path(exists=True),
    },
    "tfrecord_validation_files": {
        "default": None,
        "help": "TFRecord validation files. "
        "To be used with the TFRecord dataloader.",
        "type": click.Path(exists=True),
    },
    "tfrecord_test_files": {
        "default": None,
        "help": "TFRecord testing files. "
        "To be used with the TFRecord dataloader.",
        "type": click.Path(exists=True),
    },
    "tfrecord_buffer_size": {
        "default": None,
        "help": "TFRecord buffer size. "
        "To be used with the TFRecord dataloader.",
        "type": click.IntRange(min=1),
    },
    "tfrecord_num_parallel_reads": {
        "default": os.cpu_count(),
        "help": "TFRecord number of parallel reads. "
        "To be used with the TFRecord dataloader.",
        "type": click.IntRange(min=1),
    },
    "tfrecord_compression_type": {
        "default": "GZIP",
        "help": "TFRecord compression type. "
        "To be used with the TFRecord dataloader.",
        "type": click.Choice(
            [
                "GZIP",
                "ZLIB",
                "",
            ]
        ),
    },
    "tfrecord_compression_level": {
        "default": 9,
        "help": "TFRecord compression level. "
        "To be used with the TFRecord dataloader.",
        "type": click.IntRange(min=1, max=9),
    },
}

CONVERTER_OPTIONS = {
    "converter_model_path": {
        "default": None,
        "help": "Path to the model to convert.",
        "type": click.Path(),
    },
}

EVALUATION_OPTIONS = {
    "model_path": {
        "default": None,
        "help": "Path to the model weights.",
        "type": click.Path(),
    },
}
