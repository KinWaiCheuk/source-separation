import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import models as Model
from AudioLoader.music.mss import Moisesdb23
import hydra
from hydra.utils import to_absolute_path

import os


@hydra.main(config_path="config", config_name="exp_test", version_base=None)
def my_app(cfg):
    cfg.data_root = to_absolute_path(cfg.data_root)
    trainset = Moisesdb23(**cfg.dataset.train)
    valset = Moisesdb23(**cfg.dataset.train)
    trainloader = torch.utils.data.DataLoader(trainset, **cfg.dataloader.train)
    valloader = torch.utils.data.DataLoader(trainset, **cfg.dataloader.train)

    model = getattr(Model, cfg.model_name).load_from_checkpoint(cfg.checkpoint_path, map_location='cpu')

    checkpoint_callback = ModelCheckpoint(monitor="Train/mse_wav",
                                          filename=f"{cfg.model_name}-" + "{epoch:02d}",
                                          save_top_k=2,
                                          mode="min",
                                          auto_insert_metric_name=False,
                                          save_last=True)
    name = f"{cfg.model_name}-{cfg.sr}"
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)      
    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=[checkpoint_callback,],
                         logger=logger)


    trainer.test(model, trainloader)
    # check if bin 0-20 has changed
    
if __name__ == "__main__":
    my_app()