from core.config import Config


def generate_model():
    import tensorflow as tf

    config = Config()

    class ReparametrizationTrick(tf.keras.layers.Layer):
        def __call__(self, inputs, **__):
            z_mean, z_log_var, epsilon = inputs
            z_sigma = tf.keras.backend.exp(0.5 * z_log_var)
            return z_mean + z_sigma * epsilon

    class VAE(tf.keras.models.Model):
        def call(self, inputs, **__):
            _, e_input, angle_input, geo_input, _ = inputs
            z, _, _ = self.encoder(inputs)
            return self.decoder([z, e_input, angle_input, geo_input])

        def __init__(self, **kwargs):
            super(VAE, self).__init__(kwargs)
            self._original_dim = (
                config.cylinder_z_cell_no
                * config.cylinder_rho_cell_no
                * config.cylinder_phi_cell_no
            )
            self.encoder = self._build_encoder()
            self.decoder = self._build_decoder()
            self._set_inputs(
                inputs=self.encoder.inputs, outputs=self(self.encoder.inputs)
            )

        def _build_encoder(self) -> tf.keras.models.Model:
            inputs = self._prepare_input_layers(for_encoder=True)
            x = tf.keras.layers.concatenate(inputs[:4])

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
            encoder_output = ReparametrizationTrick()(
                [z_mean, z_log_var, inputs[4]]
            )
            return tf.keras.models.Model(
                inputs=inputs,
                outputs=[encoder_output, z_mean, z_log_var],
                name="encoder",
            )

        def _build_decoder(self) -> tf.keras.models.Model:
            inputs = self._prepare_input_layers(for_encoder=False)
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

        def _prepare_input_layers(self, for_encoder: bool):
            e_input = tf.keras.Input(shape=(1,))
            angle_input = tf.keras.Input(shape=(1,))
            geo_input = tf.keras.Input(
                shape=(config.geometry_condition_length,)
            )
            if for_encoder:
                x_input = tf.keras.Input(shape=self._original_dim)
                eps_input = tf.keras.Input(shape=config.latent_dim)
                return [x_input, e_input, angle_input, geo_input, eps_input]
            else:
                x_input = tf.keras.Input(shape=config.latent_dim)
                return [x_input, e_input, angle_input, geo_input]

    return VAE()
