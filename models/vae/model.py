from core.config import Config
from typing import List


def generate_model(
    # note: I needed a way to pass the for_training parameter to the model
    #       so I added it here. It is used when serializing the model.
    for_training: bool = True,
):
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
            if for_training:
                (
                    total_hits_no,
                    # profile2d,
                    total_energy,
                    z_profile,
                    rho_profile,
                    phi_profile,
                    e_profile,
                ) = self.decoder([latent_v, particle_v])
                # mask = tf.math.greater(output, 1e-5)
                # output = tf.where(mask, output, 0.0)
                return {
                    "total_hits_no": total_hits_no,
                    # "profile2d": profile2d,
                    "total_energy": total_energy,
                    "z_profile": z_profile,
                    "rho_profile": rho_profile,
                    "phi_profile": phi_profile,
                    "e_profile": e_profile,
                    "kl": kl_loss,
                }
            else:
                output = self.decoder([latent_v, particle_v])
                return output

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
            # profile2d = tf.keras.layers.Dense(
            #     units=config.cyliner_z_cell_no * config.cylinder_rho_cell_no,
            #     activation=config.output_activation
            # )(x)
            z_profile = tf.keras.layers.Dense(
                units=config.cylinder_z_cell_no,
                activation=config.output_activation,
            )(x)
            rho_profile = tf.keras.layers.Dense(
                units=config.cylinder_rho_cell_no,
                activation=config.output_activation,
            )(x)
            phi_profile = tf.keras.layers.Dense(
                units=config.cylinder_phi_cell_no,
                activation=config.output_activation,
            )(x)
            e_profile = tf.keras.layers.Dense(
                units=40, activation=config.output_activation
            )(x)
            # z_profile_sum = tf.keras.layers.Lambda(
            #     lambda x: tf.reduce_sum(x, axis=-1)
            # )(z_profile)
            # rho_profile_sum = tf.keras.layers.Lambda(
            #     lambda x: tf.reduce_sum(x, axis=-1)
            # )(rho_profile)
            # phi_profile_sum = tf.keras.layers.Lambda(
            #     lambda x: tf.reduce_sum(x, axis=-1)
            # )(phi_profile)
            # e_profile_sum = tf.keras.layers.Lambda(
            #     lambda x: tf.reduce_sum(x, axis=-1)
            # )(e_profile)
            total_hits_no = tf.keras.layers.Dense(
                units=1, activation=config.output_activation
            )(x)
            total_hits_no = tf.keras.layers.Lambda(
                lambda x: tf.reduce_sum(x, axis=-1)
            )(total_hits_no)
            # total_hits_no = tf.keras.layers.Add()(
            #     [
            #         z_profile_sum,
            #         rho_profile_sum,
            #         phi_profile_sum,
            #         total_hits_sum,
            #     ]
            # )
            # total_hits_no = tf.keras.layers.Lambda(
            #     lambda x: x / 4.0)(total_hits_no)
            total_energy = tf.keras.layers.Dense(
                units=1, activation=config.output_activation
            )(x)
            total_energy = tf.keras.layers.Lambda(
                lambda x: tf.reduce_sum(x, axis=-1)
            )(total_energy)

            # decoder_output_mask = tf.keras.layers.Dense(
            #     units=self._original_dim, activation=config.output_activation
            # )(x)
            # decoder_output = tf.keras.layers.Multiply()(
            #     [
            #         decoder_output,
            #         decoder_output_mask,
            #     ]
            # )
            # decoder_output = tf.keras.layers.Dense(
            #     units=self._original_dim, activation=config.output_activation
            # )(decoder_output_merged)
            outputs = None
            if for_training:
                outputs = [
                    total_hits_no,
                    # profile2d,
                    total_energy,
                    z_profile,
                    rho_profile,
                    phi_profile,
                    e_profile,
                ]
            else:
                outputs = [
                    total_hits_no,
                    # profile2d
                    total_energy,
                    z_profile,
                    rho_profile,
                    phi_profile,
                    e_profile,
                ]
            decoder = tf.keras.models.Model(
                inputs=inputs,
                outputs=outputs,
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
