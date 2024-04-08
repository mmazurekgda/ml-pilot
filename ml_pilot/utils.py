import click
# from core.config import Config
from importlib import import_module


# def set_default_options(verbosity: str, output_area: str, **kwargs):
#     config = Config()
#     config.configure(verbosity=verbosity, output_area=output_area, **kwargs)


def add_options(mode: str, model_name: str = None):
    def wrapper(callback):
        if not mode:
            mode_options = Config.default_options()
        else:
            mode_options = getattr(Config, f"{mode}_options")
        if not mode_options:
            if mode == "model":
                mode_options = import_module(
                    f"models.{model_name}.options"
                ).OPTIONS
            else:
                raise ValueError(f"Config has no option for mode '{mode}'.")
        for name, fields in reversed(mode_options.items()):
            callback = click.option(f"--{name}", **fields)(callback)
        return callback

    return wrapper
