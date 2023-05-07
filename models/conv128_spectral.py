import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torchaudio

class Conv128Spec(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # spec layer for loss calculation
        self.mel_layer = torchaudio.transforms.MelSpectrogram(n_fft=1024)
        
        self.conv1u = nn.Conv1d(2, 16, 11, padding=5)
        self.conv2u = nn.Conv1d(16, 32, 9, padding=4)
        self.conv3u = nn.Conv1d(32, 64, 7, padding=3)
        self.conv4u = nn.Conv1d(64, 128, 5, padding=2)
        
        self.conv4d = nn.Conv1d(128, 64, 5, padding=2)
        self.conv3d = nn.Conv1d(64, 32, 7, padding=3)
        self.conv2d = nn.Conv1d(32, 16, 9, padding=4)
        self.conv1d = nn.Conv1d(16, 8, 11, padding=5)

    def forward(self, x):
        # x (batch , 1, 2, len)
        x = x.flatten(1,2)
        
        x = self.conv1u(x)
        x = self.conv2u(x)
        x = self.conv3u(x)
        x = self.conv4u(x)
        
        x = self.conv4d(x)
        x = self.conv3d(x)
        x = self.conv2d(x)
        x = self.conv1d(x)
        
        return x # (batch, 8, len)


    def training_step(self, batch, batch_idx):
        pred = self(batch[0]) # (batch, 8, len)
        label = batch[1] # (batch, 4, 2, len)
        
        label_spec = self.mel_layer(batch[1].flatten(0,2))
        pred_spec = self.mel_layer(pred.flatten(0,1))
        
        # loss_wav = torch.nn.functional.mse_loss(pred, label.flatten(1,2))
        
        loss_spec = torch.nn.functional.mse_loss(torch.log(pred_spec+1e-20),
                                                 torch.log(label_spec+1e-20))
        return loss_spec

        

    def configure_optimizers(self):
        r"""Configure optimizer."""
        return optim.Adam(self.parameters())