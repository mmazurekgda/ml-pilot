import click
import logging
import pyfiglet as pf
from core.train import train_cli


@click.group()
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
def cli(verbosity):
    log = logging.getLogger("metahep")
    log.setLevel(verbosity)


cli.add_command(train_cli)

if __name__ == "__main__":
    click.secho(
        pf.figlet_format("GsinoMetaHEP"),
        fg="blue",
        bg=None,
        bold=True,
    )
    cli()
