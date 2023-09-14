import click
from config import Config


def update_configuration(mode: str):
    def wrapper(callback):
        mode_options = getattr(Config, f"{mode}_options")
        if not mode_options:
            raise ValueError(f"Config has no option '{mode_options}'.")
        for name, fields in mode_options.items():
            callback = click.option(f"--{name}", **fields)(callback)
        return callback

    return wrapper
