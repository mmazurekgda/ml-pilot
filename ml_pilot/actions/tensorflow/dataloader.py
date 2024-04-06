import os
from core.config import Config

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

def generate_tfrecord_dataloader(decoder, datatype: str):
    import tensorflow as tf

    config = Config()

    def dataloader() -> tf.data.Dataset:
        files_dir = getattr(config, f"tfrecord_{datatype}_files")
        files = []
        if not files_dir or not os.path.isdir(files_dir):
            msg = (
                f"TFRecord directory '{files_dir}' for"
                f" datatype '{datatype}' does not exist."
            )
            config.log.error(msg)
            raise FileNotFoundError(msg)
        for file_name in os.listdir(files_dir):
            if file_name.endswith(".tf"):
                files.append(os.path.join(files_dir, file_name))
        if not files:
            msg = f"No TFRecord files found in '{files_dir}'."
            config.log.error(msg)
            raise FileNotFoundError(msg)
        return tf.data.TFRecordDataset(
            files,
            buffer_size=config.tfrecord_buffer_size,
            num_parallel_reads=config.tfrecord_num_parallel_reads,
            compression_type=config.tfrecord_compression_type,
        ).map(decoder)

    return dataloader


def generate_tfrecord_datagenerator(encoder, datatype: str):
    import tensorflow as tf

    config = Config()

    def generator() -> None:
        config.log.debug(f"-> Generating tfrecord_file for '{datatype}' data.")
        tf_file_options = tf.io.TFRecordOptions(
            compression_type=config.tfrecord_compression_type,
            compression_level=config.tfrecord_compression_level,
        )
        dir = getattr(config, f"tfrecord_{datatype}_files")
        if not dir or not os.path.isdir(dir):
            msg = (
                f"TFRecord directory '{dir}' does not exist. "
                "Please create it manually."
            )
            config.log.error(msg)
            raise FileNotFoundError(msg)
        for file_name, examples in encoder():
            file_path = os.path.join(
                getattr(config, f"tfrecord_{datatype}_files"),
                file_name,
            )
            with tf.io.TFRecordWriter(
                file_path, options=tf_file_options
            ) as writer:
                for example in examples:
                    writer.write(example.SerializeToString())
            config.log.debug(f"--> Written to tfrecord_file: '{file_path}.'")

    return generator
