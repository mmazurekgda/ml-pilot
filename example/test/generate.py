from pydantic_settings import BaseSettings
from pydantic import Field

from ml_pilot.proxy import SettingsProxy
from ml_pilot.actions.tensorflow.dataloader import (
    generate_tfrecord_datagenerator,
    DataloaderOptions,
)
from dataloader import generate_tfrecord_encoder


class GenerateOptions(BaseSettings):
    generator_training_files_no: int = Field(
        ...,
        description="Number of training files to generate. "
        "To be used with the data generator.",
        ge=1,
    )
    generator_validation_files_no: int = Field(
        ...,
        description="Number of validation files to generate. "
        "To be used with the data generator.",
        ge=1,
    )
    generator_test_files_no: int = Field(
        ...,
        description="Number of test files to generate. "
        "To be used with the data generator.",
        ge=1,
    )
    generator_training_samples_no_per_file: int = Field(
        ...,
        description="Number of training samples per file to generate. "
        "To be used with the data generator.",
        ge=1,
    )
    generator_validation_samples_no_per_file: int = Field(
        ...,
        description="Number of validation samples per file to generate. "
        "To be used with the data generator.",
        ge=1,
    )
    generator_test_samples_no_per_file: int = Field(
        ...,
        description="Number of test samples per file to generate. "
        "To be used with the data generator.",
        ge=1,
    )


def generate() -> callable:
    proxy: SettingsProxy = SettingsProxy()
    generate_opts: GenerateOptions = proxy.get_settings("GenerateOptions")
    dataloader_opts: DataloaderOptions = proxy.get_settings(
        "DataloaderOptions"
    )

    for datatype in ["training", "validation", "test"]:
        proxy.logger.info(f"-> Generating '{datatype}' data.")
        if getattr(generate_opts, f"generator_{datatype}_files_no"):
            if dataloader_opts.dataloader_type == "tfrecord":
                samples_no = getattr(
                    generate_opts,
                    f"generator_{datatype}_samples_no_per_file",
                )
                files_no = getattr(
                    generate_opts, f"generator_{datatype}_files_no"
                )
                generate_tfrecord_datagenerator(
                    generate_tfrecord_encoder(
                        files_no=files_no,
                        samples_no=samples_no,
                    ),
                    datatype,
                )()
            else:
                raise NotImplementedError(
                    f"Dataloader type '{dataloader_opts.dataloader_type}'"
                    " not implemented."
                )
        else:
            proxy.logger.warning(
                f"No '{datatype}' data will be generated. "
            )
