import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List

class LSTMTradingModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3, output_dim: int = 3):
        super(LSTMTradingModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers>1 else 0,
            batch_first=True,
            bidirectional=False
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)

        attention_weights  =self.attention(lstm_out)

        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        output = self.fc(context_vector)
        return output
    
class TransformerTradingModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 512, dropout: float = 0.1, output_dim: int = 3):
        super(TransformerTradingModel, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)

        x = self.pos_encoder(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        output = self.fc(x)
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)

        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_length: int = 60):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length
    
    def __getitem__(self, idx):
        x =self.features[idx:idx + self.seq_length]

        y = self.labels[idx + self.seq_length - 1]

        return torch.FloatTensor(x), torch.LongTensor([y])
    
    def train_lstm_model(
            train_features: np.ndarray,
            train_labels: np.ndarray,
            val_features: np.ndarray,
            val_labels: np.ndarray,
            model_type: str = 'lstm',
            seq_length: int = 60,
            batch_size: int = 64,
            epochs: int = 100,
            learning_rate: float = 0.001,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> nn.Module:
        train_dataset = SequenceDataset(train_features, train_labels, seq_length)
        val_dataset = SequenceDataset(val_features, val_labels, seq_length)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        input_dim = train_features.shape[1]
        if model_type == 'lstm':
            model = LSTMTradingModel(input_dim=input_dim).to(device)
        else:
            model = TransformerTradingModel(input_dim=input_dim).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_acc = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_X, batch_y = batch_x.to(device), batch_y.to(device).squeeze()

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)

            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'models/best_lstm_model.pth')

            print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss/len(train_loader):.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss/len(val_loader):.4f} | '
              f'Val Acc: {val_acc:.2f}%')
            
        return model