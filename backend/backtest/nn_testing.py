import os
import torch
from pathlib import Path
from risk.risk_manager import RiskManager
from dotenv import load_dotenv
from training.NN import NeuralNetworkTraining

load_dotenv()

model = NeuralNetworkTraining()
model_path = os.getenv('NN_MODEL_PATH')

state_dict = torch.load(model_path, map_location=torch.device('gpu'))
model.load_state_dict(state_dict)

model.eval()

with torch.no_grad:
    output = model(input_tensor)