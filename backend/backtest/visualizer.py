import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class BacktestVisualizer:
    @staticmethod
    def plot_equity_curve(portfolio_values, title="Portfolio Performance"):
        """Plot equity curve"""
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values)
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_drawdown(portfolio_values):
        """Plot drawdown chart"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100
        
        plt.figure(figsize=(12, 4))
        plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        plt.plot(drawdown, color='red')
        plt.title('Drawdown (%)')
        plt.xlabel('Time Steps')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_comparison(strategy_values, baseline_values):
        """Compare strategy vs baseline"""
        plt.figure(figsize=(12, 6))
        plt.plot(strategy_values, label='Strategy', linewidth=2)
        plt.plot(baseline_values, label='Buy & Hold', linewidth=2)
        plt.title('Strategy vs Buy & Hold')
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_trades(portfolio_values, actions):
        """Plot trades on equity curve"""
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values, label='Portfolio Value')
        
        for action_type, step, price in actions:
            color = 'green' if action_type == 'BUY' else 'red'
            marker = '^' if action_type == 'BUY' else 'v'
            plt.scatter(step, portfolio_values[step], color=color, marker=marker, s=50)
        
        plt.title('Portfolio Performance with Trades')
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.show()