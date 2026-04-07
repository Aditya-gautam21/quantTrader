import os 
import torch
import torch.multiprocessing as mp
from training.LSTM import LSTMTradingModel
from training.NN import NeuralNetworkTraining

def run_lstm(self):
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    lstm = LSTMTradingModel()


def run_nn(self):
    device = torch.device('cuda:0')

    nn = NeuralNetworkTraining()

if __name__ == '__main__':
    nn_process = mp.Process(target=run_lstm)
    lstm_process = mp.Process(target=run_lstm)

    nn_process.start()
    lstm_process.start()

