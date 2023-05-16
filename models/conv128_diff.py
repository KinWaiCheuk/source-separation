import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torchaudio
import torch.nn.functional as F

def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """
    x_start: x0 (B, T, F) --> (B, 8, L)
    t: timestep information (B,)
    """    
    # sqrt_alphas is mean of the Gaussian N()    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]# extract the value of \bar{\alpha} at time=t
    # sqrt_alphas is variance of the Gaussian N()
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    
    # boardcasting into correct shape
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None].to(x_start.device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None].to(x_start.device)

    # scale down the input, and scale up the noise as time increases?
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def extract_x0(x_t, epsilon, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    x_t: The output from q_sample
    epsilon: The noise predicted from the model
    t: timestep information
    """
    # sqrt_alphas is mean of the Gaussian N()    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t] # extract the value of \bar{\alpha} at time=t
    # sqrt_alphas is variance of the Gaussian N()
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None].to(x_t.device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None].to(x_t.device)    

    # obtaining x0 based on the inverse of eq.4 of DDPM paper
    return (x_t - sqrt_one_minus_alphas_cumprod_t * epsilon) / sqrt_alphas_cumprod_t

class Conv128Diff(pl.LightningModule):
    def __init__(self,
                 beta_start,
                 beta_end, 
                 loss_type,
                 timesteps):
        super().__init__()
        self.save_hyperparameters()
        self.betas = linear_beta_schedule(beta_start,
                                          beta_end,
                                          timesteps=timesteps)
        
        # define alphas
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
        
        self.conv1u = nn.Conv1d(10, 16, 11, padding=5)
        self.conv2u = nn.Conv1d(16, 32, 9, padding=4)
        self.conv3u = nn.Conv1d(32, 64, 7, padding=3)
        self.conv4u = nn.Conv1d(64, 128, 5, padding=2)
        
        self.conv4d = nn.Conv1d(128, 64, 5, padding=2)
        self.conv3d = nn.Conv1d(64, 32, 7, padding=3)
        self.conv2d = nn.Conv1d(32, 16, 9, padding=4)
        self.conv1d = nn.Conv1d(16, 8, 11, padding=5)

    def step(self, batch):
        # batch["frame"] (B, 640, 88)
        # batch["audio"] (B, L)     
        
        batch_size = batch[0].shape[0]
        source_label = batch[1] # target (B, 4, 2, L)
        source_label = source_label.flatten(1,2)
        waveform = batch[0] # mix (B, 1, 2, L)
        waveform = waveform.flatten(1,2)
        device = source_label.device
        
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ## sampling the same t within each batch, might not work well
        # t = torch.randint(0, self.hparams.timesteps, (1,), device=device)[0].long() # [0] to remove dimension
        # t_tensor = t.repeat(batch_size).to(roll.device)
        
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long() # more diverse sampling
        

        noise = torch.randn_like(source_label) # creating label noise
        
        x_t = q_sample( # sampling noise at time t
            x_start=source_label,
            t=t,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            noise=noise)
        
        # feedforward
        source_pred = self(x_t, waveform, t) # predict the noise N(0, 1)
        diffusion_loss = self.p_losses(source_label, source_pred, loss_type=self.hparams.loss_type)

        
        # pred_roll = torch.sigmoid(pred_roll) # to convert logit into probability
        # amt_loss = F.binary_cross_entropy(pred_roll, roll)
        

        
        tensors = {
            "source_pred": source_pred,
            "source_label": source_label,
        }
              

        return diffusion_loss, tensors        
        
    def p_losses(self, label, prediction, loss_type="l1"):

        if loss_type == 'l1':
            loss = F.l1_loss(label, prediction)
        elif loss_type == 'l2':
            loss = F.mse_loss(label, prediction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(label, prediction)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x_t, waveform, t):
        # x (batch , 1, 2, len) --> (batch, 10, len)
        x = torch.concat(
            (waveform, x_t),
            dim=1)
        
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
        diff_loss, pred_dict = self.step(batch) # (batch, 8, len)
        label = batch[1] # (batch, 4, 2, len)
        
        loss_wav = torch.nn.functional.mse_loss(pred_dict['source_pred'],
                                                label.flatten(1,2))
        self.log("Train/mse_wav", loss_wav)
        self.log("Train/mse_diff", diff_loss)
        return loss_wav + diff_loss
    
    def validation_step(self, batch, batch_idx):
        diff_loss, pred_dict = self.step(batch)# (batch, 8, len)
        label = batch[1] # (batch, 4, 2, len)
          
        loss_wav = torch.nn.functional.mse_loss(pred_dict['source_pred'],
                                                label.flatten(1,2))
        self.log("Val/mse_wav", loss_wav)
        self.log("Val/mse_wav", diff_loss)
        return loss_wav + diff_loss

        

    def configure_optimizers(self):
        r"""Configure optimizer."""
        return optim.Adam(self.parameters())