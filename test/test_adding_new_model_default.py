import pytest
import tempfile
from core.config import Config


def test_adding_new_model_default():
    output_area = tempfile.TemporaryDirectory()
    config = Config()
    config.model_options["test_option2"] = {
        "help": "Test option2",
        "default": "test",
    }
    config.configure(
        verbosity="DEBUG",
        output_area=output_area.name,
        model_name="test",
    )
    assert hasattr(config, "test_option2")
    assert config.test_option2 == "test"
    with pytest.raises(ValueError):
        config.check_readiness()
    del Config.instance
    output_area.cleanup()
