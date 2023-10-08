from datetime import datetime
import click

from core.config import Config
from core.utils import add_options


def convert(full_model):
    import tf2onnx

    config = Config()
    config.check_readiness()
    start_time = datetime.now()
    config.log.info(
        f"-> Started model conversion for the '{config._model_name}' model."
    )
    config.log.info(f"--> Time: {start_time.strftime('%H:%M:%S')}.")
    if config.converter_model_path is None:
        msg = "No model to convert provided."
        config.log.error(msg)
        raise FileNotFoundError(msg)
    full_model.load_weights(config.converter_model_path).expect_partial()
    tf2onnx.convert.from_keras(
        full_model.decoder,  # TODO: fix this
        output_path=f"{config.output_area}/model.onnx",
        opset=18,
    )
    config.log.info(
        f"-> Finished model conversion for the '{config._model_name}' model."
    )
    config.log.info(f"--> Time: {start_time.strftime('%H:%M:%S')}.")
    config.log.info(f"--> Took: {datetime.now() - start_time} h.")


def converter_cli_generator():
    @click.group(
        name="convert",
        context_settings={"show_default": True},
    )
    @add_options(mode="converter")
    def generate_cli(*_, **kwargs):
        """
        Generate the datasets.
        """
        config = Config()
        config.set_action("convert")
        config.log.debug(
            "-> Updating the CONVERTER configuration with CLI parameters."
        )
        for prop, value in kwargs.items():
            default = getattr(config, prop)
            if default != value:
                setattr(config, prop, value)
        config.log.debug("--> Done.")

    return generate_cli
