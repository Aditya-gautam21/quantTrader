"""
Backtesting Engine for QuantTrader
Comprehensive backtesting with performance metrics and visualization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt

class BacktestEngine:
    """Comprehensive backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        
        # Results tracking
        self.trades = []
        self.portfolio_values = []
        self.positions = []
        self.cash_history = []
        
    def run_backtest(self, model, env, data: pd.DataFrame, 
                    start_date: str = None, end_date: str = None) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            model: Trained RL model
            env: Trading environment
            data: Historical price data
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with backtest results
        """
        print(f"Running backtest on {len(data)} data points...")
        
        # Filter data by date range if provided
        if start_date or end_date:
            if 'Date' in data.columns:
                data = data.set_index('Date')
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
        
        # Reset environment with backtest data
        obs, _ = env.reset()
        
        # Initialize tracking
        self.trades = []
        self.portfolio_values = [self.initial_capital]
        self.positions = [0]
        self.cash_history = [self.initial_capital]
        
        current_position = 0
        current_cash = self.initial_capital
        
        # Run through historical data
        for step in range(len(data) - 1):
            # Get model prediction
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action in environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Track portfolio state
            current_price = data.iloc[step].get('Close', data.iloc[step].get('PRICE', 100))
            portfolio_value = current_cash + current_position * current_price
            
            self.portfolio_values.append(portfolio_value)
            self.positions.append(current_position)
            self.cash_history.append(current_cash)
            
            # Record trade if position changed
            if len(env.actions_taken) > len(self.trades):
                trade_info = env.actions_taken[-1]
                self.trades.append({
                    'timestamp': data.index[step] if hasattr(data.index, 'to_pydatetime') else step,
                    'action': trade_info[0],
                    'price': trade_info[2],
                    'step': trade_info[1],
                    'portfolio_value': portfolio_value
                })
            
            if done or truncated:
                break
        
        # Calculate performance metrics
        results = self._calculate_metrics(data)
        
        print(f"Backtest completed. Total return: {results['total_return']:.2%}")
        return results
    
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if len(self.portfolio_values) < 2:
            return {"error": "Insufficient data for metrics calculation"}
        
        # Convert to numpy arrays
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Annualized metrics (assuming daily data)
        trading_days = len(returns)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Win rate
        winning_trades = len([t for t in self.trades if 'BUY' in str(t.get('action', ''))])
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Buy and hold comparison
        if 'Close' in data.columns:
            buy_hold_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
        else:
            buy_hold_return = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'final_portfolio_value': portfolio_values[-1],
            'trading_days': trading_days,
            'portfolio_values': portfolio_values.tolist(),
            'trades': self.trades
        }
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot backtest results"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        portfolio_values = results['portfolio_values']
        ax1.plot(portfolio_values, label='Strategy', linewidth=2)
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak * 100
        ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax2.plot(drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown (%)')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Returns distribution
        returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100
        ax3.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.2f}%')
        ax3.set_title('Daily Returns Distribution')
        ax3.set_xlabel('Daily Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance metrics table
        ax4.axis('off')
        metrics_text = f"""
        Performance Metrics:
        
        Total Return: {results['total_return']:.2%}
        Annualized Return: {results['annualized_return']:.2%}
        Volatility: {results['volatility']:.2%}
        Sharpe Ratio: {results['sharpe_ratio']:.2f}
        Sortino Ratio: {results['sortino_ratio']:.2f}
        Max Drawdown: {results['max_drawdown']:.2%}
        Calmar Ratio: {results['calmar_ratio']:.2f}
        Win Rate: {results['win_rate']:.2%}
        Total Trades: {results['total_trades']}
        Buy & Hold Return: {results['buy_hold_return']:.2%}
        Excess Return: {results['excess_return']:.2%}
        """
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, filepath: str):
        """Save backtest results to JSON file"""
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for JSON serialization
        clean_results = {}
        for key, value in results.items():
            clean_results[key] = convert_numpy(value)
        
        # Add metadata
        clean_results['backtest_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'commission': self.commission
        }
        
        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")

if __name__ == "__main__":
    # Example usage
    from stable_baselines3 import PPO
    from environment.trading_env import StockTradingEnv
    
    # Load sample data
    data = pd.DataFrame({
        'Close': np.random.randn(1000).cumsum() + 100,
        'PRICE': np.random.randn(1000).cumsum() + 100,
        'RSI_14': np.random.rand(1000),
        'MACD': np.random.randn(1000) * 0.1,
        'NEWS_SENTIMENT': np.random.rand(1000)
    })
    
    # Create environment and load model
    env = StockTradingEnv(data)
    
    # For demo, create a simple model (in practice, load trained model)
    model = PPO('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=1000)
    
    # Run backtest
    backtester = BacktestEngine(initial_capital=100000)
    results = backtester.run_backtest(model, env, data)
    
    # Plot and save results
    backtester.plot_results(results)
    backtester.save_results(results, 'backtest_results.json')