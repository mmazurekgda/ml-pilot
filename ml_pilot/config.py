import git
import yaml
import os
import copy
from datetime import datetime
from importlib import import_module


from core.logger import activate_logger
from options import (
    GENERAL_OPTIONS,
    TRAINING_OPTIONS,
    TRAINING_STANDARD_CALLBACKS_OPTIONS,
    TRAINING_TENSORBOARD_OPTIONS,
    DATA_OPTIONS,
    CONVERTER_OPTIONS,
    EVALUATION_OPTIONS,
)
from core.constants import (
    PROJECT_NAME,
    POSSIBLE_ACTIONS,
    ACTIVE_MODEL_NAMES,
)


class Config:
    general_options = GENERAL_OPTIONS

    training_options = {
        **TRAINING_OPTIONS,
        **TRAINING_STANDARD_CALLBACKS_OPTIONS,
        **TRAINING_TENSORBOARD_OPTIONS,
        **DATA_OPTIONS,
    }

    evaluation_options = {
        **DATA_OPTIONS,
        **EVALUATION_OPTIONS,
    }

    data_generator_options = {
        **DATA_OPTIONS,
    }

    converter_options = CONVERTER_OPTIONS

    model_options = {}

    _model_name = ""

    _action = ""

    @staticmethod
    def default_options():
        return {
            **Config.general_options,
            **Config.training_options,
            **Config.data_generator_options,
            **Config.model_options,
            **Config.converter_options,
            **Config.evaluation_options,
        }

    def options(self):
        return {
            **self.general_options,
            **self.training_options,
            **self.data_generator_options,
            **self.model_options,
            **self.converter_options,
            **self.evaluation_options,
        }

    _options_with_dirs = [
        # "tfrecord_training_files",
        # "tfrecord_validation_files",
        # "tfrecord_test_files",
    ]

    _output_area_dirs = [
        "model_checkpoint_out_weight_file",
        "tensorboard_log_dir_name",
    ]

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

    def set_action(self, action: str, ignore_already_set=False):
        if self._action and not ignore_already_set:
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
            model_options = import_module(
                f"models.{model_name}.options"
            ).OPTIONS
            self.model_options.update(model_options)
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
            with open(config_file) as yaml_dump_file:
                data = yaml.load(yaml_dump_file, Loader=yaml.FullLoader)
                if data["_model_name"]:
                    self.set_model_name(data["_model_name"])
                for prop, value in data.items():
                    default = getattr(self, prop)
                    if prop not in self._output_area_dirs and default != value:
                        self.__setattr__(prop, value, from_config_file=True)
            self.log.debug("--> Done.")
        else:
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

    def __setattr__(self, key, value, from_config_file=False):
        if key in ["_frozen", "_rigid", "configured"]:
            object.__setattr__(self, key, value)
            return
        if not self.configured:
            raise ValueError("Configuration not yet initialized!")
        new_value = copy.copy(value)  # FIXME: needed? or too paranoid
        if self._frozen and key != "_frozen":
            msg = (
                "Config became a frozen class. "
                "No futher changes possible. "
                f"Tried setting '{key}': '{value}'"
            )
            if hasattr(self, "log"):
                self.log.error(msg)
            raise ValueError(msg)
        if self._rigid and not hasattr(self, key):
            msg = (
                "Config became a rigid class. "
                "No additional options possible. "
                f"Tried setting '{key}': '{value}'"
            )
            if hasattr(self, "log"):
                self.log.error(msg)
            raise ValueError(msg)
        if key in self.options():
            if key in self._output_area_dirs and new_value:
                new_value = "/".join([self.output_area, new_value])
            if key in self._options_with_dirs and new_value:
                new_value = self.change_to_local_paths(key, value)
            override_text = "DEFAULT"
            if from_config_file:
                override_text = "NEW VALUE FROM CONFIG FILE"
            elif self._rigid:
                override_text = "NEW VALUE FROM CLI"
            self.log.debug(f"--> ({override_text}) '{key}': {new_value}")
        object.__setattr__(self, key, new_value)
        # if self._rigid:
        #     self._check_for_non_configurables(key=key)

    def _set_working_area(self):
        self.working_area = None
        repo = None
        path = None
        try:
            repo = git.Repo(".", search_parent_directories=True)
        except git.exc.InvalidGitRepositoryError:
            pass
        if repo:
            path = repo.working_tree_dir
            if os.path.exists("/".join([path, "core/constants.py"])):
                self.working_area = path
                self.log.debug(f"-> Working area is: {self.working_area}")
        if not self.working_area:
            msg = (
                "Invalid working area. "
                f"Must be inside the repo '{PROJECT_NAME}' repository."
            )
            self.log.error(msg)
            raise ValueError(msg)

    def _set_output_area(self, directory):
        if not directory:
            self.output_area = "./"
        elif not os.path.exists(directory):
            os.makedirs(directory)
        self.output_area = directory
        self.log.debug(f"-> Output area is: {self.output_area}")

    def paths_to_global(self, paths):
        directory = self.working_area
        if type(paths) is str:
            if paths.startswith("/"):
                return paths
            return "/".join([directory, paths])
        if hasattr(paths, "__iter__"):
            new_paths = []
            for path in paths:
                if path.startswith("/"):
                    new_paths.append(path)
                else:
                    new_paths.append("/".join([directory, path]))
            return new_paths

    def change_to_local_paths(self, option, paths):
        if type(paths) is str:
            return self.ensure_local_path(option, paths)
        elif hasattr(paths, "__iter__"):
            new_paths = []
            for path in paths:
                new_paths.append(self.ensure_local_path(option, path))
            return new_paths
        else:
            msg = f"-> Invalid options '{option}': '{paths}'."
            "Must be string or an iterable."
            self.log.error(msg)
            raise TypeError(msg)

    def ensure_local_path(self, option, path):
        if os.path.isabs(path):
            if not os.path.exists(path) and option != "load_weight_path":
                msg = (
                    f"-> The '{option}' has an invalid path. "
                    f"The file '{path}' does not exist."
                )
                self.log.error(msg)
                raise FileNotFoundError(msg)
            if self.working_area not in path:
                msg = (
                    f"-> The '{option}' has an invalid path. "
                    f"Must contain the working area: '{self.working_area}'"
                )
                self.log.error(msg)
                raise FileNotFoundError(msg)
            return path.replace(self.working_area + "/", "")
        else:
            if not os.path.exists("/".join([self.working_area, path])):
                msg = (
                    f"-> The '{option}' has an invalid path. "
                    "Must be with respect to "
                    f"the working area: '{self.working_area}'"
                )
                self.log.error(msg)
                raise FileNotFoundError(msg)
            return path

    def dump_to_file(self, config_file="config.yaml"):
        config_path = "/".join([self.output_area, config_file])
        with open(config_path, "w") as yaml_dump_file:
            options_values = {
                key: (getattr(self, key)) for key in self.options()
            }
            options_values["_model_name"] = self._model_name
            yaml.dump(options_values, yaml_dump_file)
        self.log.debug(f"-> Config dumped to '{config_path}'")

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

    def _freeze(self, dump_to_file=True):
        if self._frozen:
            return
        self._frozen = True
        self.log.debug("-> Freezing options. No additional changes possible.")
        if dump_to_file:
            self.log.debug("--> Dumping options to file.")
            self.dump_to_file()

    def _unfreeze(self):
        self._frozen = False
        self.log.debug("-> Unfreezing options. Additional changes possible.")
