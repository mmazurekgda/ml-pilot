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
}

VAE_OPTIONS = {
    "geometry_condition_length": {
        "help": "Length of the geometry condition",
        "default": 3,
        "type": click.IntRange(min=1),
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
}

OPTIONS = {
    **GEOMETRY_OPTIONS,
    **VAE_OPTIONS,
}
