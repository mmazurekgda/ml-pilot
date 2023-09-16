import tempfile
import subprocess


def test_cli_simple_training():
    output_area = tempfile.TemporaryDirectory()
    training_dir = tempfile.TemporaryDirectory()
    validation_dir = tempfile.TemporaryDirectory()

    ex_generate = subprocess.run(
        [
            "python",
            "run.py",
            "--verbosity",
            "DEBUG",
            "--output_area",
            str(output_area.name),
            "generate",
            "--generator_training_files_no",
            "5",
            "--generator_training_samples_no_per_file",
            "100",
            "--tfrecord_training_files",
            str(training_dir.name),
            "--generator_validation_files_no",
            "3",
            "--generator_validation_samples_no_per_file",
            "100",
            "--tfrecord_validation_files",
            str(validation_dir.name),
            "test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert ex_generate.returncode == 0
    assert (
        ex_generate.stdout.find(
            "Finished data generation for the 'test' model."
        )
        != -1
    )

    ex_train = subprocess.run(
        [
            "python",
            "run.py",
            "--verbosity",
            "DEBUG",
            "train",
            "--tfrecord_training_files",
            str(training_dir.name),
            "--tfrecord_validation_files",
            str(validation_dir.name),
            "test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert ex_train.returncode == 0
    assert ex_train.stdout.find("Finished training the 'test' model.") != -1

    output_area.cleanup()
    training_dir.cleanup()
    validation_dir.cleanup()
