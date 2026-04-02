"""
PPO Agent with a custom LSTM feature extractor.

The policy network processes the (window_size × features) observation through
an LSTM to capture temporal dependencies before the PPO actor-critic heads.
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np
from pathlib import Path


# ------------------------------------------------------------------ #
#  Custom LSTM Feature Extractor                                       #
# ------------------------------------------------------------------ #
class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Processes (batch, window_size, n_features) observations through an LSTM.
    Output: (batch, features_dim)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        n_features = observation_space.shape[1]   # (window, features)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, window, features)
        lstm_out, _ = self.lstm(obs)
        last = lstm_out[:, -1, :]          # take last timestep
        return self.head(last)


# ------------------------------------------------------------------ #
#  Agent builder                                                       #
# ------------------------------------------------------------------ #
def build_agent(
    env: gym.Env,
    log_dir: str = "logs/ppo",
    # PPO hyper-params
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    # LSTM extractor hyper-params
    lstm_hidden: int = 128,
    lstm_layers: int = 2,
    features_dim: int = 128,
    device: str = "cpu",
) -> PPO:
    """Wrap env, build VecNormalize, instantiate PPO with LSTM extractor."""

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            features_dim=features_dim,
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        activation_fn=nn.ReLU,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=1,
        device=device,
    )
    return model


def make_vec_env(env_fn, n_envs: int = 1, normalize: bool = True):
    """Create a (optionally normalised) vectorised environment."""
    vec_env = DummyVecEnv([env_fn] * n_envs)
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def build_callbacks(
    eval_env,
    checkpoint_dir: str = "logs/ppo/checkpoints",
    eval_freq: int = 10_000,
    n_eval_episodes: int = 3,
) -> CallbackList:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=eval_freq,
        save_path=checkpoint_dir,
        name_prefix="ppo_btc",
        verbose=0,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=checkpoint_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=1,
    )
    return CallbackList([checkpoint_cb, eval_cb])
