import git
import json
import os
import copy
from datetime import datetime
from importlib import import_module


from core.logger import activate_logger
from options import (
    GENERAL_OPTIONS,
    TRAINING_OPTIONS,
)
from core.constants import (
    POSSIBLE_ACTIONS,
    ACTIVE_MODEL_NAMES,
)


class Config:
    general_options = GENERAL_OPTIONS

    training_options = TRAINING_OPTIONS

    model_options = {}

    _model_name = ""

    _action = ""

    @staticmethod
    def default_options():
        return {
            **Config.general_options,
            **Config.training_options,
            **Config.model_options,
        }

    def options(self):
        return {
            **self.general_options,
            **self.training_options,
            **self.model_options,
        }

    _options_with_dirs = []

    _output_area_dirs = []

    def __new__(klass, *_, **__):
        if not hasattr(klass, "instance"):
            klass.instance = super(Config, klass).__new__(klass)
            klass.instance.configured = False
        return klass.instance

    def __init__(self):
        if self.configured:
            return
        self._frozen = False
        self._rigid = False

    def set_action(self, action: str):
        if self._action:
            raise ValueError(
                f"Action already set to '{self._action}'. "
                f"Cannot set it to '{action}'."
            )
        if action not in POSSIBLE_ACTIONS:
            raise ValueError(
                f"Invalid action '{action}'. "
                f"Possible actions are: {POSSIBLE_ACTIONS}"
            )
        self._action = action

    def set_model_name(self, model_name: str):
        if self._model_name:
            raise ValueError(
                f"Model name already set to '{self._model_name}'. "
                f"Cannot set it to '{model_name}'."
            )
        if model_name not in ACTIVE_MODEL_NAMES:
            raise ValueError(
                f"Invalid model name '{model_name}'. "
                f"Possible model names are: {ACTIVE_MODEL_NAMES}"
            )
        if model_name:
            self._model_name = model_name
            if not self.model_options:
                model_options = import_module(
                    f"models.{model_name}.options"
                ).OPTIONS
                self.model_options = model_options
                if self.configured:
                    self._unrigidify()
                    for name, value in model_options.items():
                        setattr(self, name, value["default"])
                    self._rigidify()
        else:
            raise ValueError("No model name specified!")

    def check_readiness(self) -> bool:
        if not self.configured:
            raise ValueError("Configuration not yet initialized!")
        if not self._action:
            raise ValueError("No action specified!")
        if not self._model_name:
            raise ValueError("No model specified!")
        if not self.model_options:
            raise ValueError("No model options specified!")
        return True

    def configure(
        self,
        verbosity: str = "INFO",
        output_area: str = "",
        model_name: str = "",
        action: str = "",
        **kwargs,
    ):
        if self.configured:
            raise ValueError("Already configured!")
        self.configured = True

        if not kwargs.get("run_number") or kwargs.get("add_timestamp"):
            start_time = datetime.now()
            run_number = ""
            if kwargs.get("run_number"):
                run_number = kwargs["run_number"] + "_"
            kwargs[
                "run_number"
            ] = f"{run_number}{start_time.strftime('%Y%m%d_%H%M%S%f')}"
        if not output_area:
            output_area = "./runs/"
            if kwargs.get("experiment_name"):
                parsed_experiment_name = (
                    kwargs["experiment_name"].lower().replace(" ", "_")
                )
                output_area += f"{parsed_experiment_name}/"
            if kwargs.get("model_name"):
                parsed_model_name = (
                    kwargs["model_name"].lower().replace(" ", "_")
                )
                output_area += f"{parsed_model_name}/"
            output_area += f"{kwargs['run_number']}"
        if not os.path.exists(output_area):
            os.makedirs(output_area)
        # activate the logger
        self.log = activate_logger(
            directory=output_area,
            logger_level=verbosity,
        )
        self.log.info("Initialized a new config.")
        self._set_working_area()
        self._set_output_area(output_area)
        self.log.debug("-> Loading all default options with their values.")
        for prop, fields in self.options().items():
            if "default" not in fields:
                msg = f"Option '{prop}' has no default value."
                self.log.error(msg)
                raise ValueError(msg)
            setattr(self, prop, fields["default"])
        self.log.debug("--> Done.")
        # disable adding new members in case the config file is not compatible
        self._rigidify()
        # update the options with the config file if provided
        config_file = kwargs.get("config_file") or self.config_file
        if config_file:
            self.log.debug(f"-> Loading options from file: '{config_file}'.")
            with open(config_file) as json_dump:
                data = json.load(json_dump)
                for prop, value in json.loads(data).items():
                    if prop not in self._output_area_dirs:
                        parsed_value = self._safe_object(prop, value)
                        setattr(self, prop, parsed_value)
            self.log.debug("--> Done.")
        else:
            # self._check_for_non_configurables(all=True)
            self.log.debug(
                "-> No extra config file given. Leaving default values."
            )
        self.log.debug("-> Updating the GENERAL parameters.")
        for prop, value in kwargs.items():
            default = getattr(self, prop)
            if default != value:
                setattr(self, prop, value)
        self.log.debug("--> Done.")
        if model_name:
            self.set_model_name(model_name)
        if action:
            self.set_action(action)

    def __setattr__(self, key, value):
        if key in ["_frozen", "_rigid", "configured"]:
            object.__setattr__(self, key, value)
            return
        if not self.configured:
            raise ValueError("Configuration not yet initialized!")
        new_value = copy.copy(value)  # FIXME: needed? or too paranoid
        if self._frozen and key != "_frozen":
            msg = "Config became a frozen class. "
            "No futher changes possible. "
            f"Tried setting '{key}': '{value}'"
            if hasattr(self, "log"):
                self.log.error(msg)
            raise ValueError(msg)
        if self._rigid and not hasattr(self, key):
            msg = "Config became a rigid class. "
            "No additional options possible. "
            f"Tried setting '{key}': '{value}'"
            if hasattr(self, "log"):
                self.log.error(msg)
            raise ValueError(msg)
        if key in self.options():
            if key in self._output_area_dirs and new_value:
                new_value = "/".join([self.output_area, new_value])
            if key in self._options_with_dirs and new_value:
                new_value = self.change_to_local_paths(key, value)
            override_text = "DEFAULT"
            if self._rigid:
                override_text = "NEW VALUE"
                # if not self._safe_JSON(new_value):
                if not new_value:
                    self.log.warning(
                        f"--> Setting an undefined property for '{key}'"
                    )
            self.log.debug(f"--> ({override_text}) '{key}': {new_value}")
        object.__setattr__(self, key, new_value)
        # if self._rigid:
        #     self._check_for_non_configurables(key=key)

    def _set_working_area(self):
        repo = git.Repo(".", search_parent_directories=True)
        if (
            repo.remotes.origin.url.split(".git")[0].split("/")[-1]
            != "gaussino-metahep"
        ):
            msg = "Invalid working area. "
            "Must be inside 'GaussinoMetaHEP' repository."
            self.log.error(msg)
            raise ValueError(msg)
        self.working_area = repo.working_tree_dir
        self.log.debug(f"-> Working area is: {self.working_area}")

    def _set_output_area(self, directory):
        if not directory:
            self.output_area = "./"
        elif not os.path.exists(directory):
            os.makedirs(directory)
        self.output_area = directory
        self.log.debug(f"-> Output area is: {self.output_area}")

    def _rigidify(self):
        self._rigid = True
        self.log.debug(
            "-> Making the options rigid. No additional members possible."
        )

    def _unrigidify(self):
        self._rigid = False
        self.log.debug(
            "-> Making the options flexible. Additional members possible."
        )

    def _freeze(self):
        self._frozen = True
        self.log.debug("-> Freezing options. No additional changes possible.")

    def _unfreeze(self):
        self._frozen = False
        self.log.debug("-> Unfreezing options. Additional changes possible.")
