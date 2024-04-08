import os
from typing import Literal
import logging
from pydantic import Field, DirectoryPath
from pydantic_settings import BaseSettings
from ml_pilot.proxy import SettingsProxy


class DataloaderOptions(BaseSettings):
    dataloader_type: Literal["tfrecord", "custom"] = Field(
        "tfrecord",
        description="Dataloader type",

    )
    tfrecord_training_files: DirectoryPath = Field(
        ...,
        description="TFRecord training files",
    )
    tfrecord_validation_files: DirectoryPath = Field(
        ...,
        description="TFRecord validation files",
    )
    tfrecord_test_files: DirectoryPath = Field(
        ...,
        description="TFRecord test files",
    )
    tfrecord_buffer_size: int | None = Field(
        None,
        description="TFRecord buffer size",
    )
    tfrecord_num_parallel_reads: int = Field(
        os.cpu_count(),
        ge=1,
        description="TFRecord number of parallel reads",
    )
    tfrecord_compression_type: Literal["GZIP", "ZLIB", ""] = Field(
        "GZIP",
        description="TFRecord compression type",
    )
    tfrecord_compression_level: int = Field(
        9,
        ge=1,
        le=9,
        description="TFRecord compression level",
    )


def generate_tfrecord_dataloader(decoder, datatype: str):
    import tensorflow as tf

    proxy: SettingsProxy = SettingsProxy()
    options: DataloaderOptions = proxy.get_settings("DataloaderOptions")
    logger: logging.Logger = proxy.logger

    def dataloader() -> tf.data.Dataset:
        files_dir = getattr(options, f"tfrecord_{datatype}_files")
        files = []
        if not files_dir or not os.path.isdir(files_dir):
            msg = (
                f"TFRecord directory '{files_dir}' for"
                f" datatype '{datatype}' does not exist."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)
        for file_name in os.listdir(files_dir):
            if file_name.endswith(".tf"):
                files.append(os.path.join(files_dir, file_name))
        if not files:
            msg = f"No TFRecord files found in '{files_dir}'."
            logger.error(msg)
            raise FileNotFoundError(msg)
        return tf.data.TFRecordDataset(
            files,
            buffer_size=options.tfrecord_buffer_size,
            num_parallel_reads=options.tfrecord_num_parallel_reads,
            compression_type=options.tfrecord_compression_type,
        ).map(decoder)

    return dataloader


def generate_tfrecord_datagenerator(encoder, datatype: str):
    import tensorflow as tf

    proxy: SettingsProxy = SettingsProxy()
    options: DataloaderOptions = proxy.get_settings("DataloaderOptions")
    logger: logging.Logger = proxy.logger

    def generator() -> None:
        logger.debug(f"-> Generating tfrecord_file for '{datatype}' data.")
        tf_file_options = tf.io.TFRecordOptions(
            compression_type=options.tfrecord_compression_type,
            compression_level=options.tfrecord_compression_level,
        )
        dir = getattr(options, f"tfrecord_{datatype}_files")
        if not dir or not os.path.isdir(dir):
            msg = (
                f"TFRecord directory '{dir}' does not exist. "
                "Please create it manually."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)
        for file_name, examples in encoder():
            file_path = os.path.join(
                getattr(options, f"tfrecord_{datatype}_files"),
                file_name,
            )
            with tf.io.TFRecordWriter(
                file_path, options=tf_file_options
            ) as writer:
                for example in examples:
                    writer.write(example.SerializeToString())
            logger.debug(f"--> Written to tfrecord_file: '{file_path}.'")

    return generator
