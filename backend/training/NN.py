import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

class NeuralNetworkTraining:
    def __init__(self, 
                 data_path, 
                 target_horizon=1, 
                 hidden_sizes=[64,128], 
                 dropout=.3,
                 epoch=1000,
                 learning_rate=0.01,
                 weight_decay=1e6,
                 test_size=.2,
                 val_size=.1,
                 device=None):
        self.data_path = data_path
        self.device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
        self.target_horizon = target_horizon
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_size = test_size
        self.val_size = val_size

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_data(self):
        path = Path(self.data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found at: {path.absolute()}")
        data = pd.read_csv(path, index_col=0, parse_dates=True)