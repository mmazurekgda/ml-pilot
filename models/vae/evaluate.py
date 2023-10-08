# flake8: noqa: E501
import numpy as np
from core.config import Config


def generate_evaluator():
    import tensorflow as tf
    from models.vae.plot import plot

    config = Config()

    def evaluate(
        model,
        test_dataset,
    ):
        showers_pred = []
        showers_true = []

        for x, y in test_dataset:
            # profile_2d_true = y["profile2d"]
            total_energy_true = y["total_energy"]
            total_hits_no_true = y["total_hits_no"]
            z_profile_true = y["z_profile"]
            # rho_profile_true = y["rho_profile"]
            # phi_profile_true = y["phi_profile"]
            e_profile_true = y["e_profile"]

            # shower_true = y["shower"]
            xt = []
            for xx in x:
                xt.append(tf.expand_dims(xx, axis=0))
            particle_energy = x[1][0] * config.max_energy * 1e3
            particle_energy = particle_energy.numpy()
            shower_true = x[2]
            (
                total_hits_no,
                total_energy_pred,
                z_profile_pred,
                rho_profile_pred,
                phi_profile_pred,
                e_profile_pred,
            ) = model.decoder(xt[:2])
            total_hits_no = (
                total_hits_no.numpy()[0]
                * config.cylinder_z_cell_no
                * config.cylinder_rho_cell_no
                * config.cylinder_phi_cell_no
            )
            total_hits_no_true = (
                total_hits_no_true.numpy()
                * config.cylinder_z_cell_no
                * config.cylinder_rho_cell_no
                * config.cylinder_phi_cell_no
            )
            total_energy_pred = total_energy_pred.numpy()[0] * particle_energy
            total_energy_true = total_energy_true.numpy() * particle_energy

            # z_profile_pred = z_profile_pred.numpy() * total_hits_no
            z_profile_pred = z_profile_pred.numpy() * total_energy_pred
            z_profile_pred = z_profile_pred.flatten()
            rho_profile_pred = rho_profile_pred.numpy() * total_energy_pred
            rho_profile_pred = rho_profile_pred.flatten()
            phi_profile_pred = phi_profile_pred.numpy() * total_energy_pred
            phi_profile_pred = phi_profile_pred.flatten()

            e_profile_true = (
                e_profile_true.numpy() * total_hits_no_true
            ).astype(np.int32)
            e_profile_pred = e_profile_pred.numpy() * total_hits_no
            e_profile_pred = e_profile_pred.flatten()

            # threshold = 1
            # rho_profile_pred = np.where(rho_profile_pred < threshold, 0.0, rho_profile_pred)
            # phi_profile_pred = np.where(phi_profile_pred < threshold, 0.0, phi_profile_pred)
            # z_profile_pred = np.where(z_profile_pred < threshold, 0.0, z_profile_pred)
            # e_profile_pred = np.where(e_profile_pred < threshold, 0.0, e_profile_pred)

            # min_energy = 10

            # samples_no_pred = int(total_energy_pred / min_energy)
            # samples_no_true = int(total_energy_true / min_energy)
            # min_energy = total_energy_pred / total_hits_no
            # min_energy_true = total_energy_true / total_hits_no_true

            # print(f"Min energy pred: {min_energy}")
            # print(f"Min energy true: {min_energy_true}")
            # print("**********************")
            # print(f"Total energy pred: {total_energy_pred}")
            # print(f"Total energy true: {total_energy_true}")
            # print(f"Total hits no pred: {total_hits_no}")
            # print(f"Total hits no true: {total_hits_no_true}")

            # if min_energy < 1e-4:
            #     print(f"Skipping event with min_energy: {min_energy}")
            #     return

            # samples_no_ratio = samples_no_pred / samples_no_true
            # if samples_no_ratio > 4:
            #     print(f"Skipping event with samples_no_ratio: {samples_no_ratio}")
            #     return

            # print(f"Energy pred: {total_energy_pred}, true: {

            # print(f"Energy pred: {total_energy_pred}, true: {total_energy_true}")
            # # print(f"Samples no pred: {samples_no_pred}, true: {samples_no_true}")
            # #
            # #
            # #
            # print(f"E profile pred: {e_profile_pred}")
            # print(f"Sum E profile pred: {np.sum(e_profile_pred)}")
            # print(f"E profile true: {e_profile_true}")
            # print(f"Sum E profile true: {np.sum(e_profile_true)}")
            # return

            e_profile_pred = e_profile_pred.astype(np.float64)
            e_profile_pred = np.round(e_profile_pred, 6)
            e_profile_pred = np.where(
                e_profile_pred < 1e-2, 0.0, e_profile_pred
            )

            rho_profile_pred = rho_profile_pred.astype(np.float64)
            rho_profile_pred = np.round(rho_profile_pred, 6)
            rho_profile_pred = np.where(
                rho_profile_pred < 1e-2, 0.0, rho_profile_pred
            )

            phi_profile_pred = phi_profile_pred.astype(np.float64)
            phi_profile_pred = np.round(phi_profile_pred, 6)
            phi_profile_pred = np.where(
                phi_profile_pred < 1e-2, 0.0, phi_profile_pred
            )

            z_profile_pred = z_profile_pred.astype(np.float64)
            z_profile_pred = np.round(z_profile_pred, 6)
            z_profile_pred = np.where(
                z_profile_pred < 1e-2, 0.0, z_profile_pred
            )

            e_profile_prob_sum = np.sum(e_profile_pred)
            e_profile_prob = e_profile_pred
            if e_profile_prob_sum <= 1:
                print(
                    f"Skipping event with e_profile_prob_sum: {e_profile_prob_sum}"
                )
                return
            e_profile_prob /= e_profile_prob_sum

            e_profile_ind = np.random.choice(
                np.arange(e_profile_prob.size),
                size=int(total_hits_no * 1.0),
                p=e_profile_prob,
            )
            # e_profile_sampled = e_profile_pred[e_profile_ind]

            if np.sum(rho_profile_pred) <= 1:
                print(
                    f"Skipping event with rho_profile_sum: {np.sum(rho_profile_pred)}"
                )
                return
            rho_prob = rho_profile_pred / np.sum(rho_profile_pred)
            rho_ind = np.random.choice(
                np.arange(rho_prob.size),
                size=int(total_hits_no * 1.0),
                p=rho_prob,
            )
            if np.sum(phi_profile_pred) <= 1:
                print(
                    f"Skipping event with phi_profile_sum: {np.sum(phi_profile_pred)}"
                )
                return
            phi_prob = phi_profile_pred / np.sum(phi_profile_pred)
            phi_ind = np.random.choice(
                np.arange(phi_prob.size),
                size=int(total_hits_no * 1.0),
                p=phi_prob,
            )
            if np.sum(z_profile_pred) <= 1:
                print(
                    f"Skipping event with z_profile_sum: {np.sum(z_profile_pred)}"
                )
                return
            z_prob = z_profile_pred / np.sum(z_profile_pred)
            z_ind = np.random.choice(
                np.arange(z_prob.size),
                size=int(total_hits_no * 1.0),
                p=z_prob,
            )
            energy_max = 2
            energy_min = -4
            bins = 40
            bin_size = abs((energy_max - energy_min) / bins)
            energy_bins = []
            for i in range(bins):
                energy_bins.append(
                    [
                        energy_min + i * bin_size,
                        energy_min + (i + 1) * bin_size,
                        # energy_min + () * bin_size,
                        # energy_min + (i + 2) * bin_size,
                    ]
                )
            energy_bins = np.array(energy_bins)
            # print(f"Energy bins: {energy_bins}")
            shower_pred = np.zeros(
                (
                    config.cylinder_rho_cell_no,
                    config.cylinder_phi_cell_no,
                    config.cylinder_z_cell_no,
                )
            )
            sum_energy = 0.0
            for rho_id, phi_id, z_id, e_ind in zip(
                rho_ind, phi_ind, z_ind, e_profile_ind
            ):
                energy_bin_min, energy_bin_max = energy_bins[e_ind]

                energy = (
                    np.random.uniform() * (energy_bin_max - energy_bin_min)
                    + energy_bin_min
                )
                # if energy > 2.0:
                #     raise ValueError(f"Energy: {energy}")
                energy = 10.0**energy
                sum_energy += energy
                # if sum_energy > total_energy_pred:
                #     break
                shower_pred[rho_id, phi_id, z_id] += energy

            # print(f"Z profile pred: {z_profile_pred.astype(np.int64)}")
            # z_profile_true = z_profile_true.numpy() * total_hits_no_true
            # print(f"Z profile true: {z_profile_true.astype(np.int64) }")
            # print(f"Z profile true - pred: {z_profile_true.astype(np.int64) - z_profile_pred.astype(np.int64)}")

            # print(f"Shower pred: {np.sum(shower_pred)}")
            # return

            # print(f"Rho prob: {rho_prob}")
            # print(f"Rho prob sum: {np.sum(rho_prob)}")
            # print(rho_ind[:100])

            # return

            # shower_pred = self.model.decoder(xt[:2])[0]
            # shower_pred *= particle_energy
            shower_true *= particle_energy

            # print(shower_true.shape)
            # print(shower_pred.shape)
            # return

            showers_pred.append(np.expand_dims(shower_pred, axis=0))
            showers_true.append(tf.expand_dims(shower_true, axis=0).numpy())

        showers_pred = np.concatenate(showers_pred, axis=0)
        showers_true = np.concatenate(showers_true, axis=0)

        plot(
            showers_true,
            showers_pred,
            # None,  # clean this up
            # None,  # clean this up
            # None,  # clean this up
        )

    return evaluate
