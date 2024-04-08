from datetime import datetime
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
import logging


class TrainingOptions(BaseSettings):
    learning_rate: float = Field(1e-4, description="Learning rate")
    epochs: int = Field(
        1,
        ge=1,
        description="""
        Number of epochs. If not provided, the model will be trained
        until the validation loss stops decreasing. 
        This must be set with callbacks.
        """,
    )
    batch_size: int = Field(1, ge=1, description="Batch size")
    model_path: str = Field(None, description="Path to the model weights.")
    reduce_lr: bool = Field(
        False,
        description="Reduce learning rate on plateau",
    )
    reduce_lr_verbosity: int = Field(
        1,
        ge=0,
        description="Reduce learning rate on plateau verbosity",
    )
    reduce_lr_patience: int = Field(
        10,
        ge=1,
        description="Reduce learning rate on plateau patience",
    )
    reduce_lr_cooldown: int = Field(
        10,
        ge=0,
        description="Reduce learning rate on plateau cooldown",
    )
    early_stopping: bool = Field(
        False,
        description="Early stopping",
    )
    early_stopping_verbosity: int = Field(
        1,
        ge=0,
        description="Early stopping verbosity",
    )
    early_stopping_patience: int = Field(
        30,
        ge=1,
        description="Early stopping patience",
    )
    early_stopping_min_delta: float = Field(
        1e-6,
        description="Early stopping min delta",
    )
    early_stopping_restore: bool = Field(
        True,
        description="Early stopping restore best weights",
    )
    model_checkpoint: bool = Field(
        False,
        description="Model checkpoint",
    )
    model_checkpoint_out_weight_file: str = Field(
        "model_with_weights.tf",
        description="Model checkpoint output weight file",
    )
    model_checkpoint_verbosity: int = Field(
        1,
        ge=0,
        description="Model checkpoint verbosity",
    )
    model_checkpoint_save_weights_only: bool = Field(
        True,
        description="Model checkpoint save weights only",
    )
    model_checkpoint_save_best_only: bool = Field(
        True,
        description="Model checkpoint save best only",
    )
    # TENSORBOARD
    tensorboard: bool = Field(
        False,
        description="Tensorboard",
    )
    tensorboard_log_dir_name: str = Field(
        "tensorboard",
        description="Tensorboard log dir",
    )
    tensorboard_histogram_freq: int = Field(
        1,
        ge=1,
        description="Tensorboard histogram freq",
    )
    tensorboard_write_graph: bool = Field(
        True,
        description="Tensorboard write graph",
    )
    tensorboard_write_images: bool = Field(
        False,
        description="Tensorboard write images",
    )
    tensorboard_update_freq: Literal["batch", "epoch"] = Field(
        "epoch",
        description="Tensorboard update freq",
    )
    tensorboard_profile_batch: int = Field(
        0,
        ge=0,
        description="Tensorboard profile batch",
    )
    tensorboard_embeddings_freq: int = Field(
        1,
        ge=0,
        description="Tensorboard embeddings freq",
    )
    tensorboard_embeddings_metadata: str = Field(
        None,
        description="Tensorboard embeddings metadata",
    )


def train(
    model,
    loss,
    dataset,
    val_dataset,
    options: TrainingOptions,
    logger: logging.Logger,
    custom_callbacks=[],
):
    import tensorflow as tf
    start_time = datetime.now()
    logger.info(f"-> Started training the '{options._model_name}' model.")
    logger.info(f"--> Time: {start_time.strftime('%H:%M:%S')}.")

    logger.debug("-> Model summary: ")
    model.summary(print_fn=lambda x: logger.debug(x))

    logger.debug("-> Adding the optimizer.")
    optimizer = tf.keras.optimizers.Adam(lr=options.learning_rate)
    logger.debug("--> Done.")

    if options.model_path:
        logger.debug("-> Loading weights: ")
        model.load_weights(options.model_path).expect_partial()
        logger.debug("--> Done.")

    logger.debug("-> Compiling model...")
    model.compile(optimizer=optimizer, loss=loss())
    # FIXME: why this needed?
    tf.keras.backend.set_value(model.optimizer.lr, options.learning_rate)
    logger.debug("--> Done.")
    logger.debug("-> Dataset preparation...")
    dataset = dataset.batch(options.batch_size)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE,
    )
    val_dataset = val_dataset.batch(options.batch_size)
    val_dataset = val_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE,
    )
    logger.debug("--> Done.")
    logger.debug("-> Adding callbacks...")
    callbacks = []
    if options.reduce_lr:
        logger.debug("--> Adding ReduceLROnPlateau callback")
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                verbose=options.reduce_lr_verbosity,
                patience=options.reduce_lr_patience,
                cooldown=options.reduce_lr_cooldown,
            )
        )
    if options.early_stopping:
        logger.debug("--> Adding EarlyStopping callback")
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                patience=options.early_stopping_patience,
                min_delta=options.early_stopping_min_delta,
                verbose=options.early_stopping_verbosity,
                restore_best_weights=options.early_stopping_restore,
            )
        )
    if options.model_checkpoint:
        logger.debug("--> Adding ModelCheckpoint callback")
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                options.model_checkpoint_out_weight_file,
                verbose=options.model_checkpoint_verbosity,
                save_weights_only=options.model_checkpoint_save_weights_only,
                save_best_only=options.model_checkpoint_save_best_only,
            ),
        )
    if options.tensorboard:
        logger.debug("--> Adding TensorBoard callback")
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=options.tensorboard_log_dir_name,
                histogram_freq=options.tensorboard_histogram_freq,
                write_graph=options.tensorboard_write_graph,
                write_images=options.tensorboard_write_images,
                # incompatible with TF 2.0
                # write_steps_per_second=options.tensorboard_write_steps_per_second,
                update_freq=options.tensorboard_update_freq,
                profile_batch=options.tensorboard_profile_batch,
                embeddings_freq=options.tensorboard_embeddings_freq,
                embeddings_metadata=options.tensorboard_embeddings_metadata,
            )
        )
    if custom_callbacks:
        logger.debug("--> Adding custom callbacks")
        callbacks.extend(custom_callbacks)
    logger.debug("-> Training...")
    model.fit(
        dataset,
        epochs=options.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
    )
    logger.info(f"-> Finished training the '{options._model_name}' model.")
    logger.info(f"--> Time: {start_time.strftime('%H:%M:%S')}.")
    logger.info(f"--> Took: {datetime.now() - start_time} h.")
