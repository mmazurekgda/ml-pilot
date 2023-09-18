from core.config import Config
from typing import List


def generate_model():
    import tensorflow as tf

    config = Config()

    class LatentSampling(tf.keras.layers.Layer):
        def __call__(self, inputs, **__):
            z_mean, z_log_var, epsilon = inputs
            z_sigma = tf.keras.backend.exp(0.5 * z_log_var)
            return z_mean + z_sigma * epsilon

    class VAE(tf.keras.models.Model):
        def call(self, inputs, training: bool = True):
            latent_v, particle_v, _ = inputs
            latent_e, z_mean, z_log_var = self.encoder(inputs)
            if training:
                latent_v = latent_e
            kl_loss = -0.5 * (
                1
                + z_log_var
                - tf.keras.backend.square(z_mean)
                - tf.keras.backend.exp(z_log_var)
            )
            return {
                "shower": self.decoder([latent_v, particle_v]),
                "kl_loss": kl_loss,
            }

        def __init__(self):
            super(VAE, self).__init__()
            self._original_dim = (
                config.cylinder_z_cell_no
                * config.cylinder_rho_cell_no
                * config.cylinder_phi_cell_no
            )
            self.sampling_layer = LatentSampling()
            self.encoder = self._build_encoder()
            self.decoder = self._build_decoder()
            self._set_inputs(
                inputs=self.encoder.inputs, outputs=self(self.encoder.inputs)
            )

        def _build_encoder(self) -> tf.keras.models.Model:
            inputs = self._prepare_input_layers()
            x = tf.keras.layers.concatenate(inputs[1:])
            for dim in config.intermediate_dims:
                x = tf.keras.layers.Dense(
                    units=dim,
                    activation=config.input_activation,
                    kernel_initializer=config.kernel_initializer,
                    bias_initializer=config.bias_initializer,
                )(x)
                x = tf.keras.layers.LayerNormalization()(x)
            z_mean = tf.keras.layers.Dense(config.latent_dim, name="z_mean")(x)
            z_log_var = tf.keras.layers.Dense(
                config.latent_dim, name="z_log_var"
            )(x)
            encoder_output = self.sampling_layer(
                [z_mean, z_log_var, inputs[0]]
            )
            return tf.keras.models.Model(
                inputs=inputs,
                outputs=[encoder_output, z_mean, z_log_var],
                name="encoder",
            )

        def _build_decoder(self) -> tf.keras.models.Model:
            inputs = self._prepare_input_layers(training=False)
            x = tf.keras.layers.concatenate(inputs)
            for dim in reversed(config.intermediate_dims):
                x = tf.keras.layers.Dense(
                    units=dim,
                    activation=config.input_activation,
                    kernel_initializer=config.kernel_initializer,
                    bias_initializer=config.bias_initializer,
                )(x)
                x = tf.keras.layers.LayerNormalization()(x)
            decoder_output = tf.keras.layers.Dense(
                units=self._original_dim, activation=config.output_activation
            )(x)
            decoder = tf.keras.models.Model(
                inputs=inputs,
                outputs=decoder_output,
                name="decoder",
            )
            return decoder

        def _prepare_input_layers(
            self, training: bool = True
        ) -> List[tf.keras.Input]:
            inputs = [
                tf.keras.Input((config.latent_dim,)),
                tf.keras.Input(
                    (2 + config.geometry_condition_length,),  # geo + angle
                ),
            ]
            if training:
                inputs.append(tf.keras.Input((self._original_dim,)))
            return inputs

    return VAE()
