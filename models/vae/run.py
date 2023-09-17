# from core.train import train
# from core.generate import generate
# from core.dataset import (
#     generate_tfrecord_datagenerator,
#     generate_tfrecord_dataloader,
# )
from core.config import Config
from models.vae.model import generate_model

# from models.test.loss import generate_loss
# from models.test.dataset import (
#     generate_tfrecord_encoder,
#     generate_tfrecord_decoder,
# )


def run() -> bool:
    config = Config()
    config._freeze()
    config.check_readiness()
    config.log.info(
        f"-> Starting the run for the '{config._model_name}' model."
    )
    if config._action == "train":
        generate_model()
    #     dataset = None
    #     val_dataset = None
    #     if config.dataloader_type == "tfrecord":
    #         dataset = generate_tfrecord_dataloader(
    #             generate_tfrecord_decoder(),
    #             "training",
    #         )()
    #         val_dataset = generate_tfrecord_dataloader(
    #             generate_tfrecord_decoder(),
    #             "validation",
    #         )()
    #     else:
    #         raise NotImplementedError(
    #             f"Dataloader type '{config.dataloader_type}'
    #              not implemented."
    #         )
    #     train(model, generate_loss, dataset, val_dataset)
    # elif config._action == "generate":

    #     def data_generator():
    #         for datatype in ["training", "validation", "test"]:
    #             config.log.info(f"-> Generating '{datatype}' data.")
    #             if getattr(config, f"generator_{datatype}_files_no"):
    #                 if config.dataloader_type == "tfrecord":
    #                     samples_no = getattr(
    #                         config,
    #                         f"generator_{datatype}_samples_no_per_file",
    #                     )
    #                     generate_tfrecord_datagenerator(
    #                         generate_tfrecord_encoder(
    #                             samples_no=samples_no,
    #                         ),
    #                         datatype,
    #                     )()
    #                 else:
    #                     raise NotImplementedError(
    #                         f"Dataloader type '{config.dataloader_type}'"
    #                         " not implemented."
    #                     )
    #             else:
    #                 config.log.warning(
    #                     f"No '{datatype}' data will be generated. "
    #                 )

    #     generate(data_generator)
    # else:
    #     raise NotImplementedError(
    #         f"Action '{config._action}' not implemented."
    #     )
    return True
