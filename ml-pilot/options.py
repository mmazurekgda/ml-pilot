import click
import os
from core.constants import PROJECT_NAME

GENERAL_OPTIONS = {
    "experiment_name": {
        "default": PROJECT_NAME,
        "help": "Name of the experiment.",
        "type": str,
    },
    "run_number": {
        "default": None,
        "help": "Run number / id. Preferably unique."
        "If not provided, it will be generated automatically "
        "based on the current time.",
        "type": str,
    },
    "config_file": {
        "default": None,
        "help": "Preload the configuration file",
        "type": click.Path(exists=True),
    },
}





