import git
import json
import os
import copy

from core.logger import activate_logger
from options import (
    TRAINING_OPTIONS,
)


class Config:
    training_options = {
        **TRAINING_OPTIONS,
    }

    options = {
        **training_options,
    }

    _options_with_dirs = []

    _output_area_dirs = []

    def __new__(klass, *_, **__):
        if not hasattr(klass, "instance"):
            klass.instance = super(Config, klass).__new__(klass)
            klass.instance.initialized = False
        return klass.instance

    def __init__(
        self,
        output_area: str = "./",
        load_config_file: str = "",
        freeze: bool = False,
    ):
        if self.initialized:
            return
        self.initialized = True
        self._frozen = False
        self._rigid = False
        if not os.path.exists(output_area):
            os.makedirs(output_area)
        # activate the logger
        self.log = activate_logger(
            directory=output_area,
        )
        self.log.info("Initialized a new config.")
        self._set_working_area()
        self._set_output_area(output_area)
        self.log.debug("-> Loading all default options with their values.")
        for prop, fields in self.options.items():
            if not fields.get("default"):
                msg = f"Option '{prop}' has no default value."
                self.log.error(msg)
                raise ValueError(msg)
            setattr(self, prop, fields["default"])
        # disable adding new members in case the config file is not compatible
        self._rigidify()
        if load_config_file:
            self.log.debug(
                f"-> Loading options from file: '{load_config_file}'."
            )
            with open(load_config_file) as json_dump:
                data = json.load(json_dump)
                for prop, value in json.loads(data).items():
                    if prop not in self._output_area_dirs:
                        parsed_value = self._safe_object(prop, value)
                        setattr(self, prop, parsed_value)
        else:
            # self._check_for_non_configurables(all=True)
            self.log.info(
                "-> No config file given. Will use the default values."
            )
        if freeze:
            self._freeze()

    def __setattr__(self, key, value):
        new_value = copy.copy(value)  # FIXME: needed? or too paranoid
        if key in ["_frozen", "_rigid", "initialized"]:
            object.__setattr__(self, key, new_value)
            return
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
        if key in self.options:
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
                        f"--> Setting an undefined property for {key}"
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
