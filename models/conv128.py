import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from task.separation import Separation

class Conv128(Separation):
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
        
        x = self.conv1u(x)
        x = self.conv2u(x)
        x = self.conv3u(x)
        x = self.conv4u(x)
        
        x = self.conv4d(x)
        x = self.conv3d(x)
        x = self.conv2d(x)
        x = self.conv1d(x)
        
        return x # (batch, 8, len)