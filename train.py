import torch
import pytorch_lightning as pl
import models as Model
from AudioLoader.music.mss import Moisesdb23
import hydra
from hydra.utils import to_absolute_path

import os


@hydra.main(config_path="config", config_name="exp")
def my_app(cfg):
    cfg.data_root = to_absolute_path(cfg.data_root)
    trainset = Moisesdb23(**cfg.dataset.train)
    trainloader = torch.utils.data.DataLoader(trainset, **cfg.dataloader.train)


    model = getattr(Model, cfg.model_name)()

    trainer = pl.Trainer(**cfg.trainer)


    trainer.fit(model, trainloader)
    # check if bin 0-20 has changed
    
if __name__ == "__main__":
    my_app()