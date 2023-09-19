import os
from core.train import train
from core.generate import generate
from core.dataset import (
    generate_tfrecord_datagenerator,
    generate_tfrecord_dataloader,
)
from core.config import Config
from models.vae.model import generate_model

from models.vae.loss import generate_loss
from models.vae.dataset import (
    generate_tfrecord_encoder,
    generate_tfrecord_decoder,
)
from models.vae.callbacks import generate_callbacks
import wandb


def run() -> bool:
    config = Config()
    config._freeze()
    config.check_readiness()
    config.log.info(
        f"-> Starting the run for the '{config._model_name}' model."
    )
    if config._action == "train":
        model = generate_model()
        dataset = None
        val_dataset = None
        if config.dataloader_type == "tfrecord":
            dataset = generate_tfrecord_dataloader(
                generate_tfrecord_decoder(),
                "training",
            )()
            val_dataset = generate_tfrecord_dataloader(
                generate_tfrecord_decoder(),
                "validation",
            )()
            test_dataset = generate_tfrecord_dataloader(
                generate_tfrecord_decoder(training=False),
                "test",
            )()
        else:
            raise NotImplementedError(
                f"Dataloader type '{config.dataloader_type}' not implemented."
            )
        config._freeze()
        options_values = {
            key: (getattr(config, key)) for key in config.options()
        }
        wandb.init(
            name=config.run_number,
            project=config.experiment_name,
            entity=config.wandb_entity,
            reinit=True,
            config=options_values,
            tags=config.wandb_tags,
        )
        wandb.save("/".join([config.output_area, "config.yaml"]))
        callbacks = generate_callbacks(test_dataset)
        train(model, generate_loss, dataset, val_dataset, callbacks)
    elif config._action == "generate":

        def data_generator():
            splits = {
                "training": 1 - config.validation_split - config.test_split,
                "validation": config.validation_split,
                "test": config.test_split,
            }
            files = os.listdir(config.root_files_path)
            files = [
                f"{config.root_files_path}/{file}"
                for file in files
                if file.endswith(".root")
            ]
            if not files:
                msg = f"No root files found in '{config.root_files_path}'."
                config.log.error(msg)
                raise FileNotFoundError(msg)
            for datatype in ["training", "validation", "test"]:
                config.log.info(f"-> Generating '{datatype}' data.")
                if splits[datatype]:
                    if config.dataloader_type == "tfrecord":
                        generate_tfrecord_datagenerator(
                            generate_tfrecord_encoder(
                                datatype=datatype,
                                files=files,
                                splits=splits,
                            ),
                            datatype,
                        )()
                    else:
                        raise NotImplementedError(
                            f"Dataloader type '{config.dataloader_type}'"
                            " not implemented."
                        )
                else:
                    config.log.warning(
                        f"Requested '{splits[datatype]}' "
                        f"of the '{datatype}' data. "
                        f"No '{datatype}' data will be generated. "
                    )

        generate(data_generator)
    else:
        raise NotImplementedError(
            f"Action '{config._action}' not implemented."
        )
    return True
