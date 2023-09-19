def generate_callbacks(test_dataset):
    import tensorflow as tf
    import wandb

    from core.config import Config

    from models.vae.plot import (
        ProfilePlotter,
        EnergyPlotter,
    )

    from models.vae.observables import (
        Energy,
        LongitudinalProfile,
        LateralProfile,
    )

    config = Config()

    def plot(
        e_layer_g4, e_layer_vae, particle_energy, particle_angle, geometry
    ):
        # Reshape the events into 3D
        events_no = e_layer_g4.shape[0]
        rlabel = config.plots_mplhep_rlabel_header_name
        if not rlabel:
            rlabel = (
                f"Experiment: '{config.experiment_name}'\n"
                f"Model: '{config._model_name.upper()}'\n"
                f"Test events no: {events_no}"
            )
        e_layer_vae = e_layer_vae.reshape(
            (
                e_layer_vae.shape[0],
                config.cylinder_rho_cell_no,
                config.cylinder_phi_cell_no,
                config.cylinder_z_cell_no,
            )
        )

        e_layer_g4 = e_layer_g4.reshape(
            (
                e_layer_g4.shape[0],
                config.cylinder_rho_cell_no,
                config.cylinder_phi_cell_no,
                config.cylinder_z_cell_no,
            )
        )

        # Create observables from raw data.
        full_sim_long = LongitudinalProfile(_input=e_layer_g4)
        full_sim_lat = LateralProfile(_input=e_layer_g4)
        full_sim_energy = Energy(_input=e_layer_g4)
        ml_sim_long = LongitudinalProfile(_input=e_layer_vae)
        ml_sim_lat = LateralProfile(_input=e_layer_vae)
        ml_sim_energy = Energy(_input=e_layer_vae)

        # Plot observables
        longitudinal_profile_plotter = ProfilePlotter(
            particle_energy,
            particle_angle,
            geometry,
            rlabel,
            full_sim_long,
            ml_sim_long,
            _plot_gaussian=False,
        )
        lateral_profile_plotter = ProfilePlotter(
            particle_energy,
            particle_angle,
            geometry,
            rlabel,
            full_sim_lat,
            ml_sim_lat,
            _plot_gaussian=False,
        )
        energy_plotter = EnergyPlotter(
            particle_energy,
            particle_angle,
            geometry,
            rlabel,
            full_sim_energy,
            ml_sim_energy,
        )

        longitudinal_profile_plotter.plot_and_save()
        lateral_profile_plotter.plot_and_save()
        energy_plotter.plot_and_save()

    class ValidationPlotCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if not config.plot_frequency:
                return
            if epoch > 0 and epoch % config.plot_frequency == 0:
                config.log.debug(f"Plotting step {epoch}... ")

                showers_pred = []
                showers_true = []

                for x, y in test_dataset:
                    shower_true = y["shower"]
                    xt = []
                    for xx in x:
                        xt.append(tf.expand_dims(xx, axis=0))
                    shower_pred = self.model.decoder(xt)

                    particle_energy = x[1][0] * config.max_energy * 1e3

                    shower_pred *= particle_energy
                    shower_true *= particle_energy

                    showers_pred.append(shower_pred)
                    showers_true.append(tf.expand_dims(shower_true, axis=0))

                showers_pred = tf.concat(showers_pred, axis=0)
                showers_true = tf.concat(showers_true, axis=0)

                plot(
                    showers_true.numpy(),
                    showers_pred.numpy(),
                    None,  # clean this up
                    None,  # clean this up
                    None,  # clean this up
                )

                observable_names = [
                    "LatProf",
                    "LongProf",
                    "E_tot",
                    "E_cell",
                    # "E_layer",
                    "LatFirstMoment",
                    "LatSecondMoment",
                    "LongFirstMoment",
                    "LongSecondMoment",
                ]
                for metric in observable_names:
                    image = f"{config.output_area}/{metric}.png"
                    wandb.log({metric: wandb.Image(image)})

    return [
        ValidationPlotCallback(),
        wandb.keras.WandbCallback(
            monitor="val_loss",
            mode="min",
            save_model=False,
            save_graph=False,
        ),
    ]
