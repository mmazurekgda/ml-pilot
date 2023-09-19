import click

OPTIONS = {
    "input_shape": {
        "help": "Input shape of the model.",
        "default": (4, 4),
        "type": int,
        "multiple": True,
    },
    "generator_training_files_no": {
        "default": None,
        "help": "Number of training files to generate. "
        "To be used with the data generator.",
        "type": click.IntRange(min=1),
    },
    "generator_validation_files_no": {
        "default": None,
        "help": "Number of validation files to generate. "
        "To be used with the data generator.",
        "type": click.IntRange(min=1),
    },
    "generator_test_files_no": {
        "default": None,
        "help": "Number of test files to generate. "
        "To be used with the data generator.",
        "type": click.IntRange(min=1),
    },
    "generator_training_samples_no_per_file": {
        "default": None,
        "help": "Number of training samples per file to generate. "
        "To be used with the data generator.",
        "type": click.IntRange(min=1),
    },
    "generator_validation_samples_no_per_file": {
        "default": None,
        "help": "Number of validation samples per file to generate. "
        "To be used with the data generator.",
        "type": click.IntRange(min=1),
    },
    "generator_test_samples_no_per_file": {
        "default": None,
        "help": "Number of test samples per file to generate. "
        "To be used with the data generator.",
        "type": click.IntRange(min=1),
    },
}
