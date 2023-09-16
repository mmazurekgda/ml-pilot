import pytest
from models.test import options
from core.config import Config


def test_adding_new_model_default():
    options.OPTIONS["test_option2"] = {
        "help": "Test option2",
        "default": "test",
    }
    config = Config()
    config.configure(
        verbosity="DEBUG",
        output_area="test/tmp",
        model_name="test",
    )
    assert hasattr(config, "test_option")
    assert hasattr(config, "test_option2")
    assert config.test_option2 == "test"
    with pytest.raises(ValueError):
        config.check_readiness()
    del Config.instance
