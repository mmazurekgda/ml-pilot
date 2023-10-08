# from core.config import Config


def generate_loss():
    import tensorflow as tf

    # config = Config()

    def bc_loss(y_true, y_pred):
        return tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM
        )(y_true, y_pred)

    def mse_loss(y_true, y_pred):
        return tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM
        )(y_true, y_pred)

    def mask_loss(y_true, y_pred):
        return tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM
        )(y_true, y_pred)

    def masked_reconstruction_loss(y_true, y_pred):
        return tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM
        )(y_true, y_pred)

        # # cutoff = 1e-5
        # y_true = tf.expand_dims(y_true, axis=-1)
        # y_pred = tf.expand_dims(y_pred, axis=-1)

        # true_mask = tf.math.greater(y_true, 1e-5)
        # true_obj = tf.ones_like(y_true)
        # true_obj = tf.where(true_mask, true_obj, 1e-5)

        # pred_mask = tf.math.greater(y_pred, 1e-5)
        # pred_obj = tf.ones_like(y_pred)
        # pred_obj = tf.where(pred_mask, pred_obj, 1e-5)

        # # obj_mask = tf.squeeze(true_obj, axis=-1)

        # obj_loss = tf.keras.losses.BinaryCrossentropy(
        #     reduction=tf.keras.losses.Reduction.NONE
        # )(true_obj, pred_obj)

        # reg_loss = tf.keras.losses.BinaryCrossentropy(
        #     reduction=tf.keras.losses.Reduction.NONE
        # )(y_true, y_pred)

        # obj_loss = tf.reduce_mean(obj_loss)
        # reg_loss = tf.reduce_sum(reg_loss)

        # return reg_loss + obj_loss

    def kl_mean(_, kl_loss):
        return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))

    # reco_loss = None
    # if config.reconstruction_loss == "bce":
    #     reco_loss = bc_loss
    # elif config.reconstruction_loss == "mse":
    #     reco_loss = mse_loss
    # elif config.reconstruction_loss == "masked":
    #     reco_loss = masked_reconstruction_loss
    # else:
    #     raise NotImplementedError(
    #         f"Reconstruction loss '{config.reconstruction_loss}'"
    #         "not implemented."
    #     )

    return {
        "total_hits_no": bc_loss,
        # "profile2d": bc_loss,
        "total_energy": bc_loss,
        "z_profile": bc_loss,
        "rho_profile": bc_loss,
        "phi_profile": bc_loss,
        "e_profile": bc_loss,  # bc_loss,
        "kl": kl_mean,
    }
