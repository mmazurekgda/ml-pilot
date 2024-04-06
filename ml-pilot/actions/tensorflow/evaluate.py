from datetime import datetime
import click

from core.config import Config
from core.utils import add_options



EVALUATION_OPTIONS = {
    "model_path": {
        "default": None,
        "help": "Path to the model weights.",
        "type": click.Path(),
    },
}


def evaluate(
    model,
    evaluate,
    test_dataset,
):
    # import tensorflow as tf

    config = Config()
    config.check_readiness()
    start_time = datetime.now()
    config.log.info(f"-> Started evaluating the '{config._model_name}' model.")
    config.log.info(f"--> Time: {start_time.strftime('%H:%M:%S')}.")
    config.log.debug("-> Model summary: ")
    model.summary(print_fn=lambda x: config.log.debug(x))
    config.log.debug("-> Loading weights: ")
    if not config.model_path:
        msg = "No model weights provided."
        config.log.error(msg)
        raise FileNotFoundError(msg)
    model.load_weights(config.model_path).expect_partial()
    config.log.debug("--> Done.")
    config.log.debug("-> Dataset preparation...")
    config.log.debug("--> Done.")
    config.log.debug("-> Evaluating...")
    evaluate(model, test_dataset)
    config.log.info(
        f"-> Finished evaluating the '{config._model_name}' model."
    )
    config.log.info(f"--> Time: {start_time.strftime('%H:%M:%S')}.")
    config.log.info(f"--> Took: {datetime.now() - start_time} h.")


def evaluate_cli_generator():
    @click.group(
        name="evaluate",
        context_settings={"show_default": True},
    )
    @add_options(mode="evaluation")
    def evaluate_cli(*_, **kwargs):
        """
        Train the model.
        """
        config = Config()
        config.set_action("evaluate")
        config.log.debug(
            "-> Updating the EVALUATION configuration with CLI parameters."
        )
        for prop, value in kwargs.items():
            default = getattr(config, prop)
            if default != value:
                setattr(config, prop, value)
        config.log.debug("--> Done.")

    return evaluate_cli
