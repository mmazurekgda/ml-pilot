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
        shower = tf.reshape(shower, (total_shower_size,))

        latent_v = tf.random.normal(
            mean=0, stddev=1, shape=(config.latent_dim,)
        )
        particle = tf.reshape(parsed["x"], (5,))

        inputs = [
            latent_v,
            particle,
        ]
        outputs = {
            "shower": shower,
            "kl_loss": latent_v,  # dummy
        }
        if training:
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
