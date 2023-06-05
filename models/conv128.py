import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

def calculate_sdr(ref, est):
    """
    ref: (B, L)
    est: (B, L)
    """
    assert ref.dim()==est.dim(), f"ref {ref.shape} has a different size than est {est.shape}"
    
    s_true = ref
    s_artif = est - ref

    sdr = 10. * (
        torch.log10(torch.clip(torch.mean(s_true ** 2, -1), 1e-8, torch.inf)) \
        - torch.log10(torch.clip(torch.mean(s_artif ** 2, -1), 1e-8, torch.inf)))
    return sdr

class Conv128(pl.LightningModule):
    def __init__(self):
        super().__init__()
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
        loss = torch.nn.functional.mse_loss(pred, label.flatten(1,2))
        return loss

    def test_step(self, batch, batch_idx):
        pred = self(batch[0]) # (batch, 8, len)
        label = batch[1] # (batch, 4, 2, len)
        loss = torch.nn.functional.mse_loss(pred, label.flatten(1,2))

        self.log('Test/mse_loss', loss)
        
        sdr = calculate_sdr(label.flatten(1,2), pred)
        sdr1, sdr2, sdr3, sdr4 = \
            torch.split(sdr,2, dim=1)
        self.log('Test/sdr', sdr.mean())
        self.log('Test/sdr1', sdr1.mean())
        self.log('Test/sdr2', sdr2.mean())
        self.log('Test/sdr3', sdr3.mean())
        self.log('Test/sdr4', sdr4.mean())
        return loss, sdr, sdr1, sdr2, sdr3, sdr4

        

    def configure_optimizers(self):
        r"""Configure optimizer."""
        return optim.Adam(self.parameters())