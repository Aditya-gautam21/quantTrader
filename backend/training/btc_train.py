"""
BTC LSTM & Transformer Training Script
=======================================
Drop-in script: point CSV_PATH to your file and run.
Trains both models sequentially and saves the best of each.
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# CONFIG  — edit these
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent   # training/ → backend/
CSV_PATH = BASE_DIR / 'raw_data' / '2026-01-13' / 'norm_indicators_BTCUSDT.csv'
SEQ_LENGTH    = 60               # lookback window (candles)
BATCH_SIZE    = 64
EPOCHS        = 100
LR            = 0.001
THRESHOLD     = 0.5              # label threshold on RET_1 (tune if classes imbalanced)
TRAIN_SPLIT   = 0.8              # 80% train, 20% val
SAVE_DIR      = "models"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = [
    "RSI_14", "MACD_HIST", "BB_POSITION", "RET_1", "RET_5", "RET_15",
    "PRICE_EMA21_DIST", "ATR", "RET_STD", "VOL", "VWAP_DIST",
    "BODY_ATR", "UPPER_WICK_ATR", "LOWER_WICK_ATR"
]

# ─────────────────────────────────────────────
# 1.  DATA LOADING & LABELLING
# ─────────────────────────────────────────────
def load_data(csv_path: str):
    print(f"\n{'='*55}")
    print("  LOADING DATA")
    print(f"{'='*55}")

    df = pd.read_csv(csv_path, index_col="Timestamp", parse_dates=True)
    df = df.sort_index()                     # ensure chronological order
    df = df[FEATURE_COLS].dropna()

    print(f"  Rows loaded  : {len(df):,}")
    print(f"  Date range   : {df.index[0]}  →  {df.index[-1]}")
    print(f"  Features     : {len(FEATURE_COLS)}")

    features = df[FEATURE_COLS].values.astype(np.float32)

    # Labels from next-candle RET_1 (shift -1 so we predict the FUTURE candle)
    future_ret = df["RET_1"].shift(-1)
    labels = np.where(future_ret > THRESHOLD,  2,
             np.where(future_ret < -THRESHOLD, 0, 1)).astype(np.int64)

    # Drop last row (NaN from shift)
    features = features[:-1]
    labels   = labels[:-1]

    # Class distribution check
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: "DOWN", 1: "SIDEWAYS", 2: "UP"}
    print("\n  Label distribution:")
    for u, c in zip(unique, counts):
        print(f"    {label_names[u]:>8}  ({u})  →  {c:,}  ({100*c/len(labels):.1f}%)")

    # Train / val split (no shuffle — preserve time order)
    split = int(len(features) * TRAIN_SPLIT)
    train_X, val_X = features[:split], features[split:]
    train_y, val_y = labels[:split],   labels[split:]

    print(f"\n  Train samples: {len(train_X):,}  |  Val samples: {len(val_X):,}")
    return train_X, train_y, val_X, val_y


# ─────────────────────────────────────────────
# 2.  DATASET
# ─────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_length: int):
        self.features   = features
        self.labels     = labels
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_length]
        y = self.labels[idx + self.seq_length - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# ─────────────────────────────────────────────
# 3.  MODELS
# ─────────────────────────────────────────────
class LSTMTradingModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3, output_dim: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_w      = self.attention(lstm_out)
        context     = torch.sum(attn_w * lstm_out, dim=1)
        return self.fc(context)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position  = torch.arange(max_len).unsqueeze(1)
        div_term  = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe        = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerTradingModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, dim_feedforward: int = 512,
                 dropout: float = 0.1, output_dim: int = 3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder      = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(self.input_projection(x))
        x = self.transformer(x).mean(dim=1)
        return self.fc(x)


# ─────────────────────────────────────────────
# 4.  TRAINING LOOP
# ─────────────────────────────────────────────
def train_model(model: nn.Module, model_name: str,
                train_loader: DataLoader, val_loader: DataLoader) -> dict:

    print(f"\n{'='*55}")
    print(f"  TRAINING  →  {model_name.upper()}")
    print(f"  Device    :  {DEVICE}")
    print(f"  Params    :  {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*55}")

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path    = os.path.join(SAVE_DIR, f"best_{model_name}.pth")
    best_val_acc = 0.0
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    total_start  = time.time()

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()

        # ── Train ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out  = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss    += loss.item()
            t_correct += (out.argmax(1) == by).sum().item()
            t_total   += by.size(0)

        # ── Validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out  = model(bx)
                loss = criterion(out, by)
                v_loss    += loss.item()
                v_correct += (out.argmax(1) == by).sum().item()
                v_total   += by.size(0)

        t_acc = 100 * t_correct / t_total
        v_acc = 100 * v_correct / v_total
        scheduler.step(v_loss)

        history["train_loss"].append(t_loss / len(train_loader))
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss / len(val_loader))
        history["val_acc"].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), save_path)
            star = " ★ saved"
        else:
            star = ""

        ep_time = time.time() - ep_start
        print(f"  Ep {epoch:>3}/{EPOCHS}  |"
              f"  TLoss {t_loss/len(train_loader):.4f}  TAcc {t_acc:.1f}%  |"
              f"  VLoss {v_loss/len(val_loader):.4f}  VAcc {v_acc:.1f}%  |"
              f"  {ep_time:.1f}s{star}")

    total_time = time.time() - total_start
    print(f"\n  ✓ Done  |  Best val acc: {best_val_acc:.2f}%"
          f"  |  Total time: {total_time/60:.1f} min  |  Saved → {save_path}")
    return history


# ─────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print("  BTC TRADING MODEL  —  LSTM + TRANSFORMER")
    print(f"  Device : {DEVICE}")
    print(f"{'='*55}")

    # ── Load data ──
    train_X, train_y, val_X, val_y = load_data(CSV_PATH)
    input_dim = train_X.shape[1]   # 14

    train_ds = SequenceDataset(train_X, train_y, SEQ_LENGTH)
    val_ds   = SequenceDataset(val_X,   val_y,   SEQ_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n  Batches per epoch  →  train: {len(train_loader)}  |  val: {len(val_loader)}")

    # ── Estimate time ──
    print(f"\n  ⏱  Estimated training time (CPU, both models):")
    print(f"     LSTM        ~  8–15 min   ({EPOCHS} epochs)")
    print(f"     Transformer ~ 20–35 min   ({EPOCHS} epochs)")
    print(f"     Total       ~ 30–50 min")
    print(f"     GPU (CUDA)  ~  3–8  min   (total)")

    # ── Train LSTM ──
    lstm_model = LSTMTradingModel(input_dim=input_dim)
    lstm_hist  = train_model(lstm_model, "lstm", train_loader, val_loader)

    # ── Train Transformer ──
    tf_model  = TransformerTradingModel(input_dim=input_dim)
    tf_hist   = train_model(tf_model, "transformer", train_loader, val_loader)

    # ── Final summary ──
    print(f"\n{'='*55}")
    print("  FINAL RESULTS")
    print(f"{'='*55}")
    print(f"  LSTM        best val acc : {max(lstm_hist['val_acc']):.2f}%")
    print(f"  Transformer best val acc : {max(tf_hist['val_acc']):.2f}%")
    winner = "LSTM" if max(lstm_hist["val_acc"]) >= max(tf_hist["val_acc"]) else "Transformer"
    print(f"\n  🏆  Winner : {winner}")
    print(f"  Models saved in → ./{SAVE_DIR}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()