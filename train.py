import torch
import pytorch_lightning as pl
import models as Model
from AudioLoader.music.mss import Moisesdb23

import os

trainset = Moisesdb23('/root/dataset/moisesdb23_labelnoise_v1.0_16k_stereo/', segment=10)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=8)


model = getattr(Model, 'Conv128')()
print(f"{model=}")

trainer = pl.Trainer(max_epochs=1000, gpus=[1])


trainer.fit(model, trainloader)
# check if bin 0-20 has changed