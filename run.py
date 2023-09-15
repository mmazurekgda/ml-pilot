import click
import pyfiglet as pf
from core.constants import (
    PROJECT_NAME,
    PROJECT_DESCRIPTION,
    ACTIVE_MODEL_NAMES,
)
from core.utils import set_default_options, add_options
from core.model import model_cli_generator
from core.train import train_cli_generator


def cli_generator():
    @click.option(
        "--verbosity",
        default="INFO",
        type=click.Choice(
            [
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ]
        ),
        help="verbosity of the logger",
    )
    @click.option(
        "--output_area",
        default=None,
        type=str,
        help="Output area for the run.",
    )
    @add_options(mode="general")
    def _cli(*args, **kwargs):
        set_default_options(*args, **kwargs)

    _cli.__doc__ = PROJECT_DESCRIPTION
    return _cli


cli = click.group(
    cli_generator(),
    context_settings={"show_default": True},
)

train_cli = train_cli_generator()

for model_name in ACTIVE_MODEL_NAMES:
    model_cli = model_cli_generator(model_name=model_name)
    train_cli.add_command(model_cli)
cli.add_command(train_cli)


if __name__ == "__main__":
    click.secho(
        pf.figlet_format(PROJECT_NAME),
        fg="blue",
        bg=None,
        bold=True,
    )
    cli()
