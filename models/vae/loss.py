from core.config import Config


def generate_loss():
    import tensorflow as tf

    config = Config()

    def bc_loss(y_true, y_pred):
        return tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM
        )(y_true, y_pred)

    def mse_loss(y_true, y_pred):
        return tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM
        )(y_true, y_pred)

    def kl_mean(_, kl_loss):
        return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))

    reco_loss = None
    if config.reconstruction_loss == "bce":
        reco_loss = bc_loss
    elif config.reconstruction_loss == "mse":
        reco_loss = mse_loss
    else:
        raise NotImplementedError(
            f"Reconstruction loss '{config.reconstruction_loss}'"
            "not implemented."
        )

    return {
        "shower": reco_loss,
        "kl": kl_mean,
    }
