def generate_loss():
    import tensorflow as tf

    def loss(y_true, y_pred):
        return tf.reduce_sum(tf.square(y_true - y_pred))

    return loss
