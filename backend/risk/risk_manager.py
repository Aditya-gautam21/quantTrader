"""
Risk Management Module for QuantTrader
Implements position sizing, stop losses, and risk controls
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

class RiskManager:
    """Comprehensive risk management for trading strategies"""
    
    def __init__(self, 
                 max_position_size: float = 0.1,  # 10% max per position
                 max_portfolio_risk: float = 0.02,  # 2% max portfolio risk
                 stop_loss_pct: float = 0.05,  # 5% stop loss
                 max_drawdown: float = 0.15,  # 15% max drawdown
                 var_confidence: float = 0.05):  # 95% VaR
        
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown = max_drawdown
        self.var_confidence = var_confidence
        
        # Track portfolio state
        self.portfolio_value_history = []
        self.peak_value = 0
        self.current_drawdown = 0
    
    def calculate_position_size(self, 
                              portfolio_value: float,
                              entry_price: float,
                              volatility: float,
                              confidence: float = 0.5) -> float:
        """
        Calculate optimal position size using Kelly Criterion and risk limits
        
        Args:
            portfolio_value: Current portfolio value
            entry_price: Entry price for the position
            volatility: Asset volatility (annualized)
            confidence: Model confidence (0-1)
        
        Returns:
            Position size as fraction of portfolio
        """
        # Kelly Criterion sizing (simplified)
        # f = (bp - q) / b where b=odds, p=win_prob, q=lose_prob
        win_prob = 0.5 + (confidence - 0.5) * 0.3  # Scale confidence to win probability
        lose_prob = 1 - win_prob
        
        # Assume 1:1 risk-reward ratio for simplicity
        kelly_fraction = win_prob - lose_prob
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Volatility-based sizing
        vol_adjusted_size = self.max_portfolio_risk / max(volatility, 0.01)
        
        # Take minimum of Kelly, volatility-based, and max position size
        position_size = min(kelly_fraction, vol_adjusted_size, self.max_position_size)
        
        return max(0, position_size)
    
    def check_risk_limits(self, 
                         current_portfolio_value: float,
                         proposed_position_size: float,
                         asset_price: float) -> Tuple[bool, str]:
        """
        Check if proposed trade violates risk limits
        
        Returns:
            (is_allowed, reason)
        """
        # Update portfolio tracking
        self.portfolio_value_history.append(current_portfolio_value)
        if current_portfolio_value > self.peak_value:
            self.peak_value = current_portfolio_value
        
        self.current_drawdown = (self.peak_value - current_portfolio_value) / self.peak_value
        
        # Check maximum drawdown
        if self.current_drawdown > self.max_drawdown:
            return False, f"Maximum drawdown exceeded: {self.current_drawdown:.2%}"
        
        # Check position size limit
        position_value = proposed_position_size * current_portfolio_value
        if position_value / current_portfolio_value > self.max_position_size:
            return False, f"Position size too large: {position_value/current_portfolio_value:.2%}"
        
        # Check portfolio risk limit
        if len(self.portfolio_value_history) > 30:
            returns = np.diff(self.portfolio_value_history[-30:]) / self.portfolio_value_history[-31:-1]
            var = np.percentile(returns, self.var_confidence * 100)
            if abs(var) > self.max_portfolio_risk:
                return False, f"Portfolio VaR too high: {var:.2%}"
        
        return True, "Risk limits satisfied"
    
    def calculate_stop_loss(self, entry_price: float, position_type: str = 'long') -> float:
        """Calculate stop loss price"""
        if position_type.lower() == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:  # short
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_var(self, returns: np.ndarray, confidence: float = None) -> float:
        """Calculate Value at Risk"""
        if confidence is None:
            confidence = self.var_confidence
        return np.percentile(returns, confidence * 100)
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        if len(self.portfolio_value_history) < 2:
            return {"status": "insufficient_data"}
        
        returns = np.diff(self.portfolio_value_history) / self.portfolio_value_history[:-1]
        
        return {
            "current_drawdown": self.current_drawdown,
            "max_drawdown_limit": self.max_drawdown,
            "var_95": self.calculate_var(returns, 0.05),
            "var_99": self.calculate_var(returns, 0.01),
            "volatility": np.std(returns) * np.sqrt(252),  # Annualized
            "sharpe_ratio": np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            "portfolio_value": self.portfolio_value_history[-1],
            "peak_value": self.peak_value
        }