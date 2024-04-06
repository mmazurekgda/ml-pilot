from datetime import datetime
import click

from core.config import Config
from core.utils import add_options


def generate(data_generator):
    config = Config()
    config.check_readiness()
    start_time = datetime.now()
    config.log.info(
        f"-> Started data generation for the '{config._model_name}' model."
    )
    config.log.info(f"--> Time: {start_time.strftime('%H:%M:%S')}.")
    data_generator()
    config.log.info(
        f"-> Finished data generation for the '{config._model_name}' model."
    )
    config.log.info(f"--> Time: {start_time.strftime('%H:%M:%S')}.")
    config.log.info(f"--> Took: {datetime.now() - start_time} h.")


def data_generator_cli_generator():
    @click.group(
        name="generate",
        context_settings={"show_default": True},
    )
    @add_options(mode="data_generator")
    def generate_cli(*_, **kwargs):
        """
        Generate the datasets.
        """
        config = Config()
        config.set_action("generate")
        config.log.debug(
            "-> Updating the DATA GENERATOR configuration with CLI parameters."
        )
        for prop, value in kwargs.items():
            default = getattr(config, prop)
            if default != value:
                setattr(config, prop, value)
        config.log.debug("--> Done.")

    return generate_cli
