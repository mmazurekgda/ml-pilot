from datetime import datetime
import click

from core.config import Config
from core.utils import add_options

TRAINING_OPTIONS = {
    "learning_rate": {
        "default": 1e-4,
        "help": "Learning rate",
        "type": float,
    },
    "epochs": {
        "default": 1,
        "help": "Number of epochs. If not provided, "
        "the model will be trained until the "
        "validation loss stops decreasing. "
        "This must be set with callbacks.",
        "type": click.IntRange(min=1),
    },
    "batch_size": {
        "default": 1,
        "help": "Batch size",
        "type": click.IntRange(min=1),
    },
    "model_path": {
        "default": None,
        "help": "Path to the model weights.",
        "type": click.Path(),
    },
}

TRAINING_STANDARD_CALLBACKS_OPTIONS = {
    "reduce_lr": {
        "default": False,
        "help": "Reduce learning rate on plateau",
        "type": bool,
    },
    "reduce_lr_verbosity": {
        "default": 1,
        "help": "Reduce learning rate on plateau verbosity",
        "type": int,
    },
    "reduce_lr_patience": {
        "default": 10,
        "help": "Reduce learning rate on plateau patience",
        "type": click.IntRange(min=1),
    },
    "reduce_lr_cooldown": {
        "default": 10,
        "help": "Reduce learning rate on plateau cooldown",
        "type": click.IntRange(min=0),
    },
    "early_stopping": {
        "default": False,
        "help": "Early stopping",
        "type": bool,
    },
    "early_stopping_verbosity": {
        "default": 1,
        "help": "Early stopping verbosity",
        "type": int,
    },
    "early_stopping_patience": {
        "default": 30,
        "help": "Early stopping patience",
        "type": click.IntRange(min=1),
    },
    "early_stopping_min_delta": {
        "default": 1e-6,
        "help": "Early stopping min delta",
        "type": float,
    },
    "early_stopping_restore": {
        "default": True,
        "help": "Early stopping restore best weights",
        "type": bool,
    },
    "model_checkpoint": {
        "default": False,
        "help": "Model checkpoint",
        "type": bool,
    },
    "model_checkpoint_out_weight_file": {
        "default": "model_with_weights.tf",
        "help": "Model checkpoint output weight file",
        "type": str,
    },
    "model_checkpoint_verbosity": {
        "default": 1,
        "help": "Model checkpoint verbosity",
        "type": int,
    },
    "model_checkpoint_save_weights_only": {
        "default": True,
        "help": "Model checkpoint save weights only",
        "type": bool,
    },
    "model_checkpoint_save_best_only": {
        "default": True,
        "help": "Model checkpoint save best only",
        "type": bool,
    },
}

TRAINING_TENSORBOARD_OPTIONS = {
    "tensorboard": {
        "default": False,
        "help": "Tensorboard",
        "type": bool,
    },
    "tensorboard_log_dir_name": {
        "default": "tensorboard",
        "help": "Tensorboard log dir",
        "type": str,
    },
    "tensorboard_histogram_freq": {
        "default": 1,
        "help": "Tensorboard histogram freq",
        "type": click.IntRange(min=1),
    },
    "tensorboard_write_graph": {
        "default": True,
        "help": "Tensorboard write graph",
        "type": bool,
    },
    "tensorboard_write_images": {
        "default": False,
        "help": "Tensorboard write images",
        "type": bool,
    },
    "tensorboard_update_freq": {
        "default": "epoch",
        "help": "Tensorboard update freq",
        "type": click.Choice(
            [
                "batch",
                "epoch",
            ]
        ),
    },
    "tensorboard_profile_batch": {
        "default": 0,
        "help": "Tensorboard profile batch",
        "type": click.IntRange(min=0),
    },
    "tensorboard_embeddings_freq": {
        "default": 1,
        "help": "Tensorboard embeddings freq",
        "type": click.IntRange(min=0),
    },
    "tensorboard_embeddings_metadata": {
        "default": None,
        "help": "Tensorboard embeddings metadata",
        "type": str,
    },
}


def train(
    model,
    loss,
    dataset,
    val_dataset,
    custom_callbacks=[],
):
    import tensorflow as tf

    config = Config()
    config.check_readiness()
    start_time = datetime.now()
    config.log.info(f"-> Started training the '{config._model_name}' model.")
    config.log.info(f"--> Time: {start_time.strftime('%H:%M:%S')}.")
    config.log.debug("-> Model summary: ")
    model.summary(print_fn=lambda x: config.log.debug(x))
    config.log.debug("-> Adding the optimizer.")
    optimizer = tf.keras.optimizers.Adam(lr=config.learning_rate)
    config.log.debug("--> Done.")
    if config.model_path:
        config.log.debug("-> Loading weights: ")
        model.load_weights(config.model_path).expect_partial()
        config.log.debug("--> Done.")
    config.log.debug("-> Compiling model...")
    model.compile(optimizer=optimizer, loss=loss())
    # FIXME: why this needed?
    tf.keras.backend.set_value(model.optimizer.lr, config.learning_rate)
    config.log.debug("--> Done.")
    config.log.debug("-> Dataset preparation...")
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE,
    )
    val_dataset = val_dataset.batch(config.batch_size)
    val_dataset = val_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE,
    )
    config.log.debug("--> Done.")
    config.log.debug("-> Adding callbacks...")
    callbacks = []
    if config.reduce_lr:
        config.log.debug("--> Adding ReduceLROnPlateau callback")
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                verbose=config.reduce_lr_verbosity,
                patience=config.reduce_lr_patience,
                cooldown=config.reduce_lr_cooldown,
            )
        )
    if config.early_stopping:
        config.log.debug("--> Adding EarlyStopping callback")
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                verbose=config.early_stopping_verbosity,
                restore_best_weights=config.early_stopping_restore,
            )
        )
    if config.model_checkpoint:
        config.log.debug("--> Adding ModelCheckpoint callback")
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                config.model_checkpoint_out_weight_file,
                verbose=config.model_checkpoint_verbosity,
                save_weights_only=config.model_checkpoint_save_weights_only,
                save_best_only=config.model_checkpoint_save_best_only,
            ),
        )
    if config.tensorboard:
        config.log.debug("--> Adding TensorBoard callback")
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=config.tensorboard_log_dir_name,
                histogram_freq=config.tensorboard_histogram_freq,
                write_graph=config.tensorboard_write_graph,
                write_images=config.tensorboard_write_images,
                # incompatible with TF 2.0
                # write_steps_per_second=config.tensorboard_write_steps_per_second,
                update_freq=config.tensorboard_update_freq,
                profile_batch=config.tensorboard_profile_batch,
                embeddings_freq=config.tensorboard_embeddings_freq,
                embeddings_metadata=config.tensorboard_embeddings_metadata,
            )
        )
    if custom_callbacks:
        config.log.debug("--> Adding custom callbacks")
        callbacks.extend(custom_callbacks)
    config.log.debug("-> Training...")
    model.fit(
        dataset,
        epochs=config.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
    )
    config.log.info(f"-> Finished training the '{config._model_name}' model.")
    config.log.info(f"--> Time: {start_time.strftime('%H:%M:%S')}.")
    config.log.info(f"--> Took: {datetime.now() - start_time} h.")


def train_cli_generator():
    @click.group(
        name="train",
        context_settings={"show_default": True},
    )
    @add_options(mode="training")
    def train_cli(*_, **kwargs):
        """
        Train the model.
        """
        config = Config()
        config.set_action("train")
        config.log.debug(
            "-> Updating the TRAINING configuration with CLI parameters."
        )
        for prop, value in kwargs.items():
            default = getattr(config, prop)
            if default != value:
                setattr(config, prop, value)
        config.log.debug("--> Done.")

    return train_cli
