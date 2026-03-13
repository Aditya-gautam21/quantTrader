import numpy as np
import pandas as pd

class PerformanceMetrics:
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
    
    @staticmethod
    def calculate_max_drawdown(portfolio_values):
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return np.min(drawdown)
    
    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.02):
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        return np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
    
    @staticmethod
    def calculate_win_rate(returns):
        """Calculate win rate"""
        return len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
    
    @staticmethod
    def calculate_all_metrics(portfolio_values):
        """Calculate comprehensive performance metrics"""
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        return {
            'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
            'annualized_return': ((portfolio_values[-1] / portfolio_values[0]) ** (252/len(portfolio_values))) - 1,
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(returns),
            'max_drawdown': PerformanceMetrics.calculate_max_drawdown(portfolio_values),
            'win_rate': PerformanceMetrics.calculate_win_rate(returns),
            'volatility': np.std(returns) * np.sqrt(252)
        }