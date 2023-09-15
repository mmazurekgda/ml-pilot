from datetime import datetime
import click

from core.config import Config
from core.utils import add_options


def train():
    config = Config()
    config.check_readiness()
    start_time = datetime.now()
    config.log.info(f"-> Started training the '{config._model_name}' model.")
    config.log.info(f"--> Time: {start_time.strftime('%H:%M:%S')}.")

    config.log.info(f"-> Finished training the '{config._model_name}' model.")
    config.log.info(f"--> Time: {start_time.strftime('%H:%M:%S')}.")
    config.log.info(f"--> Took: {datetime.now() - start_time} h.")


def train_cli_generator():
    @click.group(
        name="train",
        context_settings={"show_default": True},
    )
    @add_options(mode="training")
    def train_cli(*_, **kwargs):
        """
        Train the model.
        """
        config = Config()
        config.set_action("train")
        config.log.debug(
            "-> Updating the TRAINING configuration with CLI parameters."
        )
        for prop, value in kwargs.items():
            default = getattr(config, prop)
            if default != value:
                setattr(config, prop, value)
        config.log.debug("--> Done.")

    return train_cli
