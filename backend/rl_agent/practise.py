import gymnasium as gym
from gymnasium import spaces 
import pandas as pd
import numpy as np
import torch

class RLTraining:
    metadata = {"render_modes" : ["human"]}
    def __init__(
            df: pd.DataFrame,
            window_size,
            transacton_cost,
            dropdown_penalty
    ):
        super().__init__
lm.