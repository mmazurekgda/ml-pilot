from datetime import datetime
import click

from config import Config
from utils import update_configuration


def train(output_area, config_file, **kwargs):
    """
    Train a model.
    """
    start_time = datetime.now()
    parsed_start_time = start_time.strftime("%Y%m%d_%H%M%S%f")
    config = Config(
        output_area=output_area.format(parsed_start_time),
        load_config_file=config_file,
    )
    config.log.debug("-> Updating the configuration with CLI parameters.")
    for prop, value in kwargs.items():
        setattr(config, prop, value)
    config.log.info(f"Training a model, started at {start_time}")


@click.command(name="train")
@click.option(
    "--output_area",
    help="Directory where to keep files",
    default="./evaluations/{}",
)
@click.option(
    "--config_file",
    help="Preload the configuration file",
    type=str,
    default=None,
)
@update_configuration(mode="training")
def train_cli(output_area, config_file, **kwargs):
    """
    Train a model.
    """
    start_time = datetime.now()
    parsed_start_time = start_time.strftime("%Y%m%d_%H%M%S%f")
    config = Config(
        output_area=output_area.format(parsed_start_time),
        load_config_file=config_file,
    )
    config.log.debug("-> Updating the configuration with CLI parameters.")
    for prop, value in kwargs.items():
        setattr(config, prop, value)
    config.log.info(f"Training a model, started at {start_time}")
