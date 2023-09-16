from models.test.run import run
from core.config import Config
import tempfile


def test_standalone_simple_training():
    output_area = tempfile.TemporaryDirectory()
    training_dir = tempfile.TemporaryDirectory()
    validation_dir = tempfile.TemporaryDirectory()
    config = Config()
    import logging

    logging.info("test")
    config.configure(
        verbosity="DEBUG",
        output_area=output_area.name,
        action="generate",
        model_name="test",
    )
    config.generator_training_files_no = 5
    config.generator_training_samples_no_per_file = 100
    config.tfrecord_training_files = training_dir.name
    config.generator_validation_files_no = 3
    config.generator_validation_samples_no_per_file = 100
    config.tfrecord_validation_files = validation_dir.name
    assert config.check_readiness()
    assert run()
    config._unfreeze()
    config.set_action("train", ignore_already_set=True)
    assert run()
    del Config.instance
    output_area.cleanup()
    training_dir.cleanup()
    validation_dir.cleanup()
