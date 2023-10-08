def generate_callbacks(test_dataset):
    import tensorflow as tf
    import wandb

    from core.config import Config
    from models.vae.evaluate import generate_evaluator

    config = Config()

    class ValidationPlotCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.evaluate = generate_evaluator()

        def on_epoch_end(self, epoch, logs=None):
            if not config.plot_frequency:
                return
            if epoch > 0 and epoch % config.plot_frequency == 0:
                config.log.debug(f"Plotting step {epoch}... ")
                self.evaluate(self.model, test_dataset)
                if config.wandb_entity:
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

    callbacks = [
        ValidationPlotCallback(),
    ]

    if config.wandb_entity:
        callbacks.append(
            wandb.keras.WandbCallback(
                monitor="val_loss",
                mode="min",
                save_model=False,
                save_graph=False,
            )
        )

    return callbacks
