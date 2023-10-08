from core.config import Config

# careful! order matters!
hits_int_properties = ["RhoID", "PhiID", "ZID"]
hits_float_properties = ["EDep"]
particles_properties = ["KineticEnergy", "Angle"]


def generate_tfrecord_decoder(training: bool = True):
    import tensorflow as tf

    config = Config()

    def decoder(dataset):
        parsed = tf.io.parse_single_example(
            dataset,
            {
                "x": tf.io.FixedLenSequenceFeature(
                    [], dtype=tf.float32, allow_missing=True
                ),
                "y_ids": tf.io.FixedLenSequenceFeature(
                    [], dtype=tf.int64, allow_missing=True
                ),
                "y_edeps": tf.io.FixedLenSequenceFeature(
                    [], dtype=tf.float32, allow_missing=True
                ),
                "y_length": tf.io.FixedLenFeature([], dtype=tf.int64),
            },
        )

        shower = tf.sparse.SparseTensor(
            indices=tf.reshape(
                parsed["y_ids"], (parsed["y_length"], len(hits_int_properties))
            ),
            dense_shape=(
                config.cylinder_rho_cell_no,
                config.cylinder_phi_cell_no,
                config.cylinder_z_cell_no,
            ),
            values=tf.squeeze(parsed["y_edeps"]),
        )

        total_shower_size = (
            config.cylinder_rho_cell_no
            * config.cylinder_phi_cell_no
            * config.cylinder_z_cell_no
        )

        shower = tf.sparse.reorder(shower)
        shower = tf.sparse.to_dense(shower)
        total_shower_energy = tf.reduce_sum(parsed["y_edeps"])
        # if total_shower_energy > 0.0:
        #     shower /= total_shower_energy

        shower_z_profile = tf.reduce_sum(shower, axis=(0, 1))
        shower_rho_profile = tf.reduce_sum(shower, axis=(1, 2))
        shower_phi_profile = tf.reduce_sum(shower, axis=(0, 2))

        # shower_z_profile = tf.math.count_nonzero(shower, axis=(0, 1))
        # shower_z_profile = tf.cast(shower_z_profile, tf.float32)
        # shower_rho_profile = tf.math.count_nonzero(shower, axis=(1, 2))
        # shower_rho_profile = tf.cast(shower_rho_profile, tf.float32)
        # shower_phi_profile = tf.math.count_nonzero(shower, axis=(0, 2))
        # shower_phi_profile = tf.cast(shower_phi_profile, tf.float32)

        shower = tf.reshape(shower, (total_shower_size,))

        latent_v = tf.random.normal(
            mean=0, stddev=1, shape=(config.latent_dim,)
        )
        particle = tf.reshape(parsed["x"], (5,))

        total_number_of_hits = parsed["y_length"] / total_shower_size

        # shower_no_zero = tf.reshape(shower, (total_shower_size,))
        # shower_no_zero = shower_no_zero[shower_no_zero != 0.0]
        # no_zero_count = shower_no_zero.shape[0]
        # print(shower_no_zero.shape)
        # # shower_no_zero = tf.flatten(shower_no_zero)
        # total_number_of_hits = no_zero_count / total_shower_size

        shower_no_zero = shower * particle[0] * config.max_energy * 1e3
        shower_no_zero_mask = shower_no_zero != 0.0
        shower_no_zero = tf.boolean_mask(shower_no_zero, shower_no_zero_mask)
        shower_log = tf.math.log(shower_no_zero) / tf.math.log(10.0)
        e_profile_max = 2
        e_profile_min = -4.0
        e_profile_bins = 40
        e_profile_ind = tf.histogram_fixed_width_bins(
            shower_log, [e_profile_min, e_profile_max], nbins=e_profile_bins
        )
        e_profile = []
        for i in range(e_profile_bins):
            mask = tf.equal(e_profile_ind, i)
            e_profile_bin = tf.boolean_mask(shower_no_zero, mask)
            e_profile_bin_sum = tf.math.count_nonzero(e_profile_bin)
            e_profile_bin_sum = tf.expand_dims(e_profile_bin_sum, axis=0)
            e_profile.append(e_profile_bin_sum)
        e_profile = tf.concat(e_profile, axis=0)
        e_profile = tf.cast(e_profile, tf.float32)

        if tf.cast(parsed["y_length"], tf.float32) > 0:
            e_profile /= tf.cast(parsed["y_length"], tf.float32)
            # shower_z_profile /= tf.cast(parsed["y_length"], tf.float32)
            # shower_rho_profile /= tf.cast(parsed["y_length"], tf.float32)
            # shower_phi_profile /= tf.cast(parsed["y_length"], tf.float32)
        else:
            e_profile = tf.zeros_like(e_profile)

        if tf.cast(total_shower_energy, tf.float32) > 0:
            shower_z_profile /= tf.cast(total_shower_energy, tf.float32)
            shower_rho_profile /= tf.cast(total_shower_energy, tf.float32)
            shower_phi_profile /= tf.cast(total_shower_energy, tf.float32)
        else:
            shower_z_profile = tf.zeros_like(shower_z_profile)
            shower_rho_profile = tf.zeros_like(shower_rho_profile)
            shower_phi_profile = tf.zeros_like(shower_phi_profile)

        # print(e_profile)
        # raise ValueError

        # mask = tf.math.greater(shower, 0.0)
        # masked_shower = tf.ones_like(shower)
        # masked_shower = tf.where(mask, masked_shower, 0.0)

        inputs = [
            latent_v,
            particle,
        ]
        outputs = {
            "total_hits_no": total_number_of_hits,
            "total_energy": total_shower_energy,
            "z_profile": shower_z_profile,
            "rho_profile": shower_rho_profile,
            "phi_profile": shower_phi_profile,
            "e_profile": e_profile,
            "kl": latent_v,  # dummy
        }
        # if training:
        inputs.append(shower)

        return (
            tuple(inputs),
            outputs,
        )

    return decoder


def generate_tfrecord_encoder(datatype: str, files: list, splits: list):
    import numpy as np
    import tensorflow as tf
    import uproot
    import os

    config = Config()

    def float_list(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def int64_list(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def get_initial_id(samples_no: int):
        if datatype == "training":
            return 0
        elif datatype == "validation":
            return int(splits["training"] * samples_no)
        elif datatype == "test":
            return int(
                (splits["training"] + splits["validation"]) * samples_no
            )

    def encoder():
        counter = 0
        while counter < len(files):
            root_file = uproot.open(files[counter])
            particles_tree = root_file[config.particles_key]
            particles_df = particles_tree.arrays(
                particles_tree.keys(), library="pd"
            )
            all_id_tuples = particles_df[["EventID", "TrackID"]]
            all_id_tuples = all_id_tuples.drop_duplicates().to_numpy()
            showers_no = all_id_tuples.shape[0]
            start_id = get_initial_id(showers_no)
            id_tuples = all_id_tuples[
                start_id : start_id  # noqa: E203
                + (int(splits[datatype] * showers_no))
            ]
            hits_tree = root_file[config.hits_key]
            hits_df = hits_tree.arrays(hits_tree.keys(), library="pd")
            examples = []
            for event_id, track_id in id_tuples:
                # fetch particles
                particle = particles_df[
                    (particles_df["EventID"] == event_id)
                    & (particles_df["TrackID"] == track_id)
                ][particles_properties].to_numpy()
                particle = particle.flatten()
                if particle.shape[0] != 2:
                    raise ValueError(
                        f"Particle with event_id={event_id} "
                        f"and track_id={track_id} "
                        f"has {particle.shape[0]} properties instead of 2."
                    )
                particle_energy = particle[0]
                particle[0] /= config.max_energy * 1e3
                particle[1] /= config.max_theta * 2.0 * np.pi / 360.0
                geometry_v = np.zeros(config.geometry_condition_length)
                geometry_v[config.geometry_condition_position] = 1
                x = np.concatenate([particle, geometry_v], axis=0)
                x = x.flatten()
                # fetch showers
                shower_ids = hits_df[
                    (hits_df["EventID"] == event_id)
                    & (hits_df["TrackID"] == track_id)
                ][hits_int_properties].to_numpy()
                y_ids = shower_ids.flatten()
                shower_edeps = hits_df[
                    (hits_df["EventID"] == event_id)
                    & (hits_df["TrackID"] == track_id)
                ][hits_float_properties].to_numpy()
                shower_edeps /= particle_energy
                y_edeps = shower_edeps.flatten()
                y_length = shower_ids.shape[0]
                feature = {
                    "x": float_list(x),
                    "y_ids": int64_list(y_ids),
                    "y_edeps": float_list(y_edeps),
                    "y_length": int64_feature(y_length),
                }
                examples.append(
                    tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    )
                )
            file_name = os.path.basename(files[counter])
            file_name = file_name.replace(".root", ".tf")
            yield file_name, examples
            counter += 1

    return encoder
