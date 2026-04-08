import os 
import torch
import torch.multiprocessing as mp
from training.LSTM import LSTMTradingModel
from training.NN import NeuralNetworkTraining

device = 'gpu' if torch.cuda.is_available() else 'cpu'

def run_lstm(self):
    lstm = LSTMTradingModel()
    pass


def run_nn(self):
    nn = NeuralNetworkTraining()
    pass

if __name__ == '__main__':
    nn_process = mp.Process(target=run_lstm)
    lstm_process = mp.Process(target=run_lstm)

    nn_process.start()
    lstm_process.start()

