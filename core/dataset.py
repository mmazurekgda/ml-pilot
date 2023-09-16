import os
from core.config import Config


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
        for _ in range(getattr(config, f"generator_{datatype}_files_no")):
            file_name, example = encoder()
            file_path = os.path.join(
                getattr(config, f"tfrecord_{datatype}_files"),
                file_name,
            )
            with tf.io.TFRecordWriter(
                file_path, options=tf_file_options
            ) as writer:
                writer.write(example.SerializeToString())
            config.log.debug(f"--> Written to tfrecord_file: '{file_path}.'")

    return generator
