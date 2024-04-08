import click
import pyfiglet as pf
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pydanclick import from_pydantic
from ml_pilot.proxy import SettingsProxy

from ml_pilot.actions.tensorflow.train import TrainingOptions
from ml_pilot.actions.tensorflow.dataloader import DataloaderOptions

from model import create_test_model, TestModelOptions
from generate import generate, GenerateOptions
from train import train



main_cli = SettingsProxy.get_main_cli()


@click.group()
@from_pydantic("test_model_opts", TestModelOptions)
def test_model_cli(
    test_model_opts: TestModelOptions,
):
    proxy = SettingsProxy()
    proxy.add_settings(test_model_opts)


@click.command()
@from_pydantic("training", TrainingOptions)
@from_pydantic("data_loader_opts", DataloaderOptions)
def training_cli(
    training_options: TrainingOptions,
    data_loader_opts: DataloaderOptions,
):
    proxy: SettingsProxy = SettingsProxy()
    proxy.set_model(create_test_model())
    proxy.add_settings(training_options)
    proxy.add_settings(data_loader_opts)
    proxy.freeze()
    train()


@click.command()
@from_pydantic("generate_opts", GenerateOptions)
@from_pydantic("data_loader_opts", DataloaderOptions)
def generate_cli(
    generate_opts: GenerateOptions,
    data_loader_opts: DataloaderOptions,
):
    proxy: SettingsProxy = SettingsProxy()
    proxy.add_settings(generate_opts)
    proxy.add_settings(data_loader_opts)
    proxy.freeze()
    generate()


test_model_cli.add_command(training_cli)
test_model_cli.add_command(generate_cli)
main_cli.add_command(test_model_cli)


if __name__ == "__main__":
    main_cli()



