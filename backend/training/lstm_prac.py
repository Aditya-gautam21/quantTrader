import pandas as pd
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.attention = nn.attention(

        )


        self.fc = nn.Sequential(
            nn.Linear(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear
        )
        self.sigmoid = nn.Sigmoid()

    def forward():
        lstm_out, (hidden, cell) = self.lstm(x)