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

class Separation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def step(self, batch):
        # batch.shape = (B, 4, 2, L)
        if self.training:
            assert batch.dim() == 4 and batch.shape[1]==4,  \
                f"Batch shape must be (B, 4, L), but got {batch.shape}"

            mix = batch.sum(dim=1)  # (B, 2channel, L)
            sources = batch
        else:
            mix = batch[:,0]
            sources = batch[:,1:]
            
        pred = self(mix) # (batch, 8, len) 
        
        return pred, sources

    def training_step(self, batch, batch_idx):
        # batch.shape = (B, 4, 2, L)
        pred, label = self.step(batch)
        # label.shape = (batch, 4, 2, len)
        loss = torch.nn.functional.mse_loss(pred, label.flatten(1,2))
        sdr = calculate_sdr(label.flatten(1,2), pred)
        sdr1, sdr2, sdr3, sdr4 = \
            torch.split(sdr,2, dim=1)
        
        self.log('Train/mse_wav', loss)
        self.log('Train/sdr', sdr.mean())
        self.log('Train/sdr1', sdr1.mean())
        self.log('Train/sdr2', sdr2.mean())
        self.log('Train/sdr3', sdr3.mean())
        self.log('Train/sdr4', sdr4.mean())
        
        return loss

    def test_step(self, batch, batch_idx):
        # batch.shape = (B, 4, 2, L)
        pred, label = self.step(batch)
        # label.shape = (batch, 4, 2, len)
        loss = torch.nn.functional.mse_loss(pred, label.flatten(1,2))
        sdr = calculate_sdr(label.flatten(1,2), pred)
        sdr1, sdr2, sdr3, sdr4 = \
            torch.split(sdr,2, dim=1)
        
        self.log('Test/mse_wav', loss)
        self.log('Test/sdr', sdr.mean())
        self.log('Test/sdr1', sdr1.mean())
        self.log('Test/sdr2', sdr2.mean())
        self.log('Test/sdr3', sdr3.mean())
        self.log('Test/sdr4', sdr4.mean())
        return loss, sdr, sdr1, sdr2, sdr3, sdr4

        

    def configure_optimizers(self):
        r"""Configure optimizer."""
        return optim.Adam(self.parameters(), lr=1e-6)