def generate_loss():
    import tensorflow as tf

    def bc_loss(y_true, y_pred):
        print(y_true.shape, y_pred.shape)
        print(y_true)
        print(y_pred)
        return tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM
        )(y_true, y_pred)

    def kl_mean(_, kl_loss):
        return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))

    return {
        "shower": bc_loss,
        "kl_loss": kl_mean,
    }
