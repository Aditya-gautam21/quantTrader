import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

class NeuralNetworkTraining:
    def __init__(self):
