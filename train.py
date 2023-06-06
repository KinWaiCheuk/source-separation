import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import models as Model
import AudioLoader.music.mss as Dataset
import hydra
from hydra.utils import to_absolute_path

import os


@hydra.main(config_path="config", config_name="exp", version_base=None)
def my_app(cfg):
    cfg.dataset.data_root = to_absolute_path(cfg.dataset.data_root)
    # trainset = Moisesdb23(**cfg.dataset.train)
    # valset = Moisesdb23(**cfg.dataset.train)

    trainset = getattr(Dataset, cfg.dataset.name)(**cfg.dataset.train)
    valset = getattr(Dataset, cfg.dataset.name)(**cfg.dataset.val)
    testset = getattr(Dataset, cfg.dataset.name)(**cfg.dataset.test)
    
    trainloader = torch.utils.data.DataLoader(trainset, **cfg.dataloader.train)
    valloader = torch.utils.data.DataLoader(valset, **cfg.dataloader.val)
    testloader = torch.utils.data.DataLoader(testset, **cfg.dataloader.test)

    model = getattr(Model, cfg.model_name)()

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


    trainer.fit(model, trainloader, valloader)
    trainer.test(model, testloader)
    # check if bin 0-20 has changed
    
if __name__ == "__main__":
    my_app()