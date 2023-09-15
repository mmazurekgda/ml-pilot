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
        "type": str,
    },
}

TRAINING_OPTIONS = {
    "learning_rate": {
        "default": 1e-4,
        "help": "Learning rate",
        "type": float,
    },
}

ARCHITECTURE_OPTIONS = {
    "model_version": {
        "default": "0.0.1",
        "help": "Version of the model structure.",
        "type": str,
    },
}
