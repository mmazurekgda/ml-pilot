from pydantic import DirectoryPath
from pydantic_settings import BaseSettings
from pydanclick import from_pydantic
from typing import Dict, Optional, Any
import click
from typing import Literal
from ml_pilot.logger import activate_logger
import pyfiglet as pf


class MainOptions(BaseSettings):
    verbosity: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    output_area: Optional[DirectoryPath] = None
    project_name: str = "ML-Pilot"


class SettingsProxy:
    _instance: Optional["SettingsProxy"] = None
    _options: Dict[str, BaseSettings] = {}
    _frozen: bool = False
    _model: Optional[Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = None
        return cls._instance

    def __init__(self) -> None:
        if "MainOptions" in self._options and not self.logger:
            main_opts: MainOptions = self._options["MainOptions"]
            self.logger = activate_logger(
                directory=main_opts.output_area,
                logger_level=main_opts.verbosity,
            )

    @staticmethod
    def get_main_cli() -> callable:
        @click.group()
        @from_pydantic("main_opts", MainOptions)
        def main_cli(
            main_opts: MainOptions,
        ):
            proxy: SettingsProxy = SettingsProxy()
            proxy.add_settings(main_opts)
            click.secho(
                pf.figlet_format(main_opts.project_name),
                fg="blue",
                bg=None,
                bold=True,
            )
        return main_cli

    def add_settings(self, settings: BaseSettings) -> None:
        cls_name = settings.__class__.__name__
        if cls_name in self._options:
            raise ValueError(f"Settings model '{cls_name}' already added.")
        new_fields = settings.model_fields
        for model in self._options.values():
            for field in new_fields:
                if field in model.model_fields:
                    raise ValueError(
                        f"Field '{field}' of model '{cls_name}' already exists "
                        f"in model '{model.__class__.__name__}'."
                    )
        self._options[cls_name] = settings

    def __getattr__(self, name: str):
        for model in self._options.values():
            if hasattr(model, name):
                return getattr(model, name)
        return self.__getattribute__(name)

    def __setattr__(self, name: str, value):
        if self._frozen:
            raise ValueError("Settings are frozen.")
        for model in self._options.values():
            if hasattr(model, name):
                setattr(model, name, value)
                return
        super().__setattr__(name, value)

    def get_settings(self, model_name: str) -> BaseSettings:
        return self._options[model_name]

    def freeze(self):
        self._frozen = True
        for model in self._options.values():
            model.model_config["frozen"] = True

    def unfreeze(self):
        self._frozen = False
        for model in self._options.values():
            model.model_config["frozen"] = False

    def set_model(self, model: Any):
        self._model = model

    def get_model(self) -> Any:
        return self._model