import click
from core.utils import add_options
from core.config import Config
from importlib import import_module


ACTIVE_MODEL_NAMES = [
    "test",
]


def model_cli_generator(model_name: str):
    @add_options(mode="model", model_name=model_name)
    def model_cli(**kwargs):
        config = Config()
        if config._model_name and config._model_name != model_name:
            msg = (
                "Incompatible model name. "
                f"Expected: {model_name}. "
                f"Found: {config._model_name}."
            )
            config.log.error(msg)
            raise click.BadParameter(msg)
        if not config._model_name:
            config.set_model_name(model_name)
        config.log.debug(
            "-> Updating the MODEL configuration with CLI parameters."
        )
        for prop, value in kwargs.items():
            default = getattr(config, prop)
            if default != value:
                setattr(config, prop, value)
        config.log.debug("--> Done.")
        import_module(f"models.{model_name}.run").run()

    model_cli.__doc__ = f"Use the '{model_name}' model."
    model_cli = click.command(name=model_name)(model_cli)
    return model_cli
