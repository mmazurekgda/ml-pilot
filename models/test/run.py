from core.train import train
from core.config import Config


def run():
    config = Config()
    config._freeze()
    config.check_readiness()
    config.log.info(
        f"-> Starting the run for the '{config._model_name}' model."
    )
    if config._action == "train":
        train()
    else:
        raise NotImplementedError(
            f"Action '{config._action}' not implemented."
        )
