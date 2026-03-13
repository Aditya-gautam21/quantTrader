"""
Module 4A: Train RL Agent OFFLINE on historical data (Completely FREE)
Using Stable-Baselines3 (free & open-source)
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DDPG, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from pathlib import Path

class OfflineTrainer:
    """Train RL agents on historical data"""
    
    def __init__(self, model_dir="./logs/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def train_agent(self, env_train, env_val, agent_type='PPO', total_timesteps=100000):
        """
        Train agent on historical data
        
        Args:
            env_train: Training environment (historical data)
            env_val: Validation environment
            agent_type: 'PPO', 'A2C', or 'DDPG'
            total_timesteps: Training iterations
        
        Returns:
            Trained model
        """
        print(f"\n[TRAIN] Training {agent_type} agent...")
        print(f"   Total timesteps: {total_timesteps}")
        
        # Create callbacks for checkpointing
        checkpoint_dir = self.model_dir / f"checkpoints_{agent_type.lower()}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1000, total_timesteps // 10),
            save_path=str(checkpoint_dir),
            name_prefix=f"offline_{agent_type.lower()}",
            save_replay_buffer=True,
            verbose=0
        )
        
        eval_callback = EvalCallback(
            env_val,
            best_model_save_path=str(self.model_dir / f"best_{agent_type.lower()}"),
            log_path=str(self.model_dir),
            eval_freq=max(1000, total_timesteps // 20),
            n_eval_episodes=3,
            deterministic=False,
            verbose=0
        )
        
        # Initialize agent
        if agent_type == 'PPO':
            model = PPO(
                'MlpPolicy',
                env_train,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                device='cpu'  # CPU is free, GPU optional
            )
        elif agent_type == 'A2C':
            model = A2C(
                'MlpPolicy',
                env_train,
                learning_rate=7e-4,
                gamma=0.99,
                verbose=1,
                device='cpu'
            )
        elif agent_type == 'DDPG':
            model = DDPG(
                'MlpPolicy',
                env_train,
                learning_rate=1e-3,
                gamma=0.99,
                verbose=1,
                device='cpu'
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            log_interval=10,
            progress_bar=True
        )
        
        # Save final model
        model_path = self.model_dir / f"pretrained_{agent_type.lower()}"
        model.save(str(model_path))
        print(f"[OK] Model saved to {model_path}.zip")
        
        return model
    
    def train_ensemble(self, env_train, env_val, total_timesteps=100000):
        """Train multiple agents for ensemble"""
        models = {}
        
        for agent_type in ['PPO', 'A2C', 'DDPG']:
            print(f"\n{'='*60}")
            print(f"Training {agent_type}...")
            print(f"{'='*60}")
            
            models[agent_type] = self.train_agent(
                env_train, env_val, agent_type, total_timesteps
            )
        
        return models

if __name__ == "__main__":
    print("[LEARN] LEARNING MODULE 4: Train RL Agent on Historical Data\n")
    
    from data_collector.market_collector import MarketDataCollector
    from features.indicators import TechnicalIndicators
    from features.aggregator import FeatureAggregator
    from environment.trading_env import StockTradingEnv
    
    # Step 1: Collect data
    print("Step 1: Collecting market data...")
    collector = MarketDataCollector()
    market_data = collector.download_historical_data(
        tickers=['AAPL'],
        start_date='2022-01-01',
        end_date='2024-10-30'
    )
    
    # Step 2: Create features
    print("\nStep 2: Creating features...")
    aggregator = FeatureAggregator()
    features = aggregator.create_state_features(
        market_data,
        news_data=[],
        ticker='AAPL'
    )
    
    # Step 3: Create environments
    print("\nStep 3: Creating environments...")
    split_idx = int(len(features) * 0.8)
    train_data = features.iloc[:split_idx]
    val_data = features.iloc[split_idx:]
    
    env_train = StockTradingEnv(train_data, initial_balance=100000)
    env_val = StockTradingEnv(val_data, initial_balance=100000)
    
    # Step 4: Train
    print("\nStep 4: Training agent...")
    trainer = OfflineTrainer()
    model = trainer.train_agent(
        env_train, env_val,
        agent_type='PPO',
        total_timesteps=50000  # Reduce for faster testing
    )
    
    print("\n[OK] Training complete!")