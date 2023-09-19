import click

GEOMETRY_OPTIONS = {
    "cylinder_z_cell_no": {
        "help": "Number of cells in the z-direction of the cylinder",
        "default": 45,
        "type": click.IntRange(min=1),
    },
    "cylinder_rho_cell_no": {
        "help": "Number of cells along rho of the cylinder",
        "default": 18,
        "type": click.IntRange(min=1),
    },
    "cylinder_phi_cell_no": {
        "help": "Number of cells along phi of the cylinder",
        "default": 50,
        "type": click.IntRange(min=1),
    },
    "cylinder_rho_cell_size": {
        "help": "Size of the cells along rho of the cylinder in mm",
        "default": 9.0,
        "type": click.FloatRange(min=0),
    },
    "cylinder_z_cell_size": {
        "help": "Size of the cells along z of the cylinder in mm",
        "default": 13.824,
        "type": click.FloatRange(min=0),
    },
    "max_theta": {
        "help": "Maximum angle of the particles in degrees",
        "default": 90,
        "type": click.FloatRange(min=0, max=360),
    },
    "max_energy": {
        "help": "Maximum energy of the particles in GeV",
        "default": 1024,
        "type": click.FloatRange(min=0),
    },
}

DATALOADER_OPTIONS = {
    "root_files_path": {
        "default": None,
        "help": "Path to the native training files (ROOT). ",
        "type": click.Path(exists=True),
    },
    "hits_key": {
        "default": "Gsino__CaloChallenge__TrainingDataCollector/CaloHits;1",
        "help": "Key of the hits in the ROOT files.",
        "type": str,
    },
    "particles_key": {
        "default": "Gsino__CaloChallenge__TrainingDataCollector/"
        "CollectorHits;1",
        "help": "Key of the particles in the ROOT files.",
        "type": str,
    },
    "validation_split": {
        "default": 0.2,
        "help": "Validation split with respect to number "
        "of events in each  ROOT file.",
        "type": click.FloatRange(min=0, max=1),
    },
    "test_split": {
        "default": 0.1,
        "help": "Test split with respect to number "
        "of events in each ROOT file.",
        "type": click.FloatRange(min=0, max=1),
    },
}

VAE_OPTIONS = {
    "geometry_condition_length": {
        "help": "Length of the geometry condition",
        "default": 3,
        "type": click.IntRange(min=1),
    },
    "geometry_condition_position": {
        "help": "Position in the geometry condition vector.",
        "default": 0,
        "type": click.IntRange(min=0),
    },
    "input_activation": {
        "help": "Activation function for the input layers",
        "default": "leaky_relu",
        "type": click.Choice(["leaky_relu", "relu", "tanh", "sigmoid"]),
    },
    "output_activation": {
        "help": "Activation function for the output layers",
        "default": "sigmoid",
        "type": click.Choice(["leaky_relu", "relu", "tanh", "sigmoid"]),
    },
    "intermediate_dims": {
        "help": "Dimensions of the intermediate layers",
        "default": [100, 50, 20, 15],
        "type": int,
        "multiple": True,
    },
    "latent_dim": {
        "help": "Dimension of the latent space",
        "default": 10,
        "type": int,
    },
    "kernel_initializer": {
        "help": "Initializer for the kernel weights",
        "default": "RandomNormal",
        "type": click.Choice(
            [
                "RandomNormal",
                "RandomUniform",
                "TruncatedNormal",
                "VarianceScaling",
                "Orthogonal",
                "LecunNormal",
                "GlorotNormal",
                "GlorotUniform",
                "HeNormal",
                "HeUniform",
                "Identity",
                "Ones",
                "Zeros",
            ]
        ),
    },
    "bias_initializer": {
        "help": "Initializer for the bias weights",
        "default": "Zeros",
        "type": click.Choice(
            [
                "RandomNormal",
                "RandomUniform",
                "TruncatedNormal",
                "VarianceScaling",
                "Orthogonal",
                "LecunNormal",
                "GlorotNormal",
                "GlorotUniform",
                "HeNormal",
                "HeUniform",
                "Identity",
                "Ones",
                "Zeros",
            ]
        ),
    },
    "reconstruction_loss": {
        "help": "Reconstruction loss",
        "default": "bce",
        "type": click.Choice(["mse", "bce"]),
    },
}

PLOTTING_OPTIONS = {
    "plot_frequency": {
        "help": "Frequency of plotting",
        "default": 10,
        "type": click.IntRange(min=0),
    },
    "plots_full_sim_histogram_color": {
        "help": "Color of the full simulation histogram",
        "default": "blue",
        "type": str,
    },
    "plots_ml_sim_histogram_color": {
        "help": "Color of the ML simulation histogram",
        "default": "red",
        "type": str,
    },
    "plots_full_sim_gaussian_color": {
        "help": "Color of the full simulation gaussian",
        "default": "green",
        "type": str,
    },
    "plots_ml_sim_gaussian_color": {
        "help": "Color of the ML simulation gaussian",
        "default": "orange",
        "type": str,
    },
    "plots_histogram_type": {
        "help": "Type of the histogram",
        "default": "step",
        "type": click.Choice(["step", "bar"]),
    },
    "plots_mplhep_experiment_header_name": {
        "help": "Name of the experiment header",
        "default": "Gauss",
        "type": str,
    },
    "plots_mplhep_llabel_header_name": {
        "help": "Left Label of the experiment header",
        "default": "Private Simulation",
        "type": str,
    },
    "plots_mplhep_rlabel_header_name": {
        "help": "Right Label of the experiment header",
        "default": None,
        "type": str,
    },
    "plots_mplhep_fontsize": {
        "help": "Fontsize of the experiment header",
        "default": 20,
        "type": click.IntRange(min=1),
    },
    "plots_mplhep_legend_loc": {
        "help": "Location of the legend",
        "default": 1,
        "type": click.IntRange(min=0, max=10),
    },
    "plots_figsize_x": {
        "help": "Size of the figure in x",
        "default": 16,
        "type": click.IntRange(min=1),
    },
    "plots_figsize_y": {
        "help": "Size of the figure in y",
        "default": 8,
        "type": click.IntRange(min=1),
    },
}

WANDB_OPTIONS = {
    "wandb_entity": {
        "help": "Wandb entity",
        "default": None,
        "type": str,
    },
    "wandb_tags": {
        "help": "Wandb tags",
        "default": [],
        "type": str,
        "multiple": True,
    },
}

OPTIONS = {
    **GEOMETRY_OPTIONS,
    **DATALOADER_OPTIONS,
    **VAE_OPTIONS,
    **PLOTTING_OPTIONS,
    **WANDB_OPTIONS,
}
