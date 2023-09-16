from datetime import datetime
import click

from core.config import Config
from core.utils import add_options


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
    config.log.debug("-> Compiling model...")
    model.compile(optimizer=optimizer, loss=loss())
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
