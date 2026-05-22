import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, initial_capital=10000.0, maker_fee=0.0002, taker_fee=0.0004, slippage=0.0002):
        self.initial_capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage

    def run(self, df, strategy, is_futures=False):
        """
        Runs the backtest on historical OHLCV data.
        
        Args:
            df (pd.DataFrame): OHLCV candle data with indicators and signals.
            strategy (BaseStrategy): An instance of a strategy class.
            is_futures (bool): True for Futures, False for Spot.
            
        Returns:
            metrics (dict): Performance summary.
            df_equity (pd.DataFrame): Time-series of equity, drawdown, etc.
            trades (list): List of completed trades.
        """
        df = df.copy().reset_index(drop=True)
        signals = strategy.generate_signals(df)
        
        # Portfolio states
        cash = self.initial_capital
        position_size = 0.0  # size in asset
        entry_price = 0.0
        position_dir = 0     # 1: Long, -1: Short, 0: Flat
        
        # Risk settings from strategy
        leverage = min(strategy.leverage if is_futures else 1, 3) # Capped at 3x
        stop_loss_pct = strategy.stop_loss_pct
        take_profit_pct = strategy.take_profit_pct
        max_allocation = strategy.max_allocation_pct
        
        # Target SL/TP prices
        sl_price = 0.0
        tp_price = 0.0
        
        equity_history = []
        timestamps = []
        trades = []
        
        current_trade = None
        
        for i in range(len(df)):
            candle = df.iloc[i]
            close_price = candle['close']
            high_price = candle['high']
            low_price = candle['low']
            open_price = candle['open']
            timestamp = candle['timestamp']
            
            # 1. Update Strategy Dynamic Risk if adaptive
            # Some strategies (like AdaptiveRegimeStrategy) can adjust risk dynamically based on regime
            if hasattr(strategy, 'get_dynamic_risk') and 'regime' in candle:
                regime = candle['regime']
                dyn_risk = strategy.get_dynamic_risk(regime)
                # Apply dynamic adjustments
                leverage = min(dyn_risk.get('leverage', leverage), 3)
                stop_loss_pct = dyn_risk.get('stop_loss_pct', stop_loss_pct)
                take_profit_pct = dyn_risk.get('take_profit_pct', take_profit_pct)
                max_allocation = dyn_risk.get('max_allocation_pct', max_allocation)

            # Calculate current portfolio equity before executing candle events
            unrealized_pnl = 0.0
            if position_dir == 1: # Long
                unrealized_pnl = position_size * (close_price - entry_price)
            elif position_dir == -1: # Short (Futures only)
                unrealized_pnl = position_size * (entry_price - close_price)
                
            current_equity = cash + unrealized_pnl
            
            # Check for liquidation in futures
            if is_futures and position_dir != 0:
                # Maintenance margin liquidation approximation (e.g. if equity falls below 10% of position value)
                position_value = abs(position_size) * entry_price
                maintenance_margin = position_value * (0.05 / leverage)  # simple approximation
                margin = current_equity
                if margin <= maintenance_margin:
                    # Liquidation triggered!
                    exit_price = low_price if position_dir == 1 else high_price
                    # Force exit
                    loss = position_size * (entry_price - exit_price) if position_dir == -1 else position_size * (exit_price - entry_price)
                    cash = max(0.0, cash + loss - (position_value * self.taker_fee))
                    position_size = 0.0
                    position_dir = 0
                    current_equity = cash
                    
                    if current_trade:
                        current_trade['exit_price'] = exit_price
                        current_trade['exit_time'] = timestamp
                        current_trade['pnl'] = loss
                        current_trade['exit_reason'] = 'LIQUIDATION'
                        trades.append(current_trade)
                        current_trade = None
            
            # 2. Check Stop Loss / Take Profit triggers during the current candle
            if position_dir != 0:
                triggered_exit = False
                exit_price = 0.0
                exit_reason = ""
                
                if position_dir == 1:  # Long position
                    # Check SL
                    if low_price <= sl_price:
                        exit_price = sl_price
                        triggered_exit = True
                        exit_reason = "STOP_LOSS"
                    # Check TP
                    elif high_price >= tp_price:
                        exit_price = tp_price
                        triggered_exit = True
                        exit_reason = "TAKE_PROFIT"
                elif position_dir == -1: # Short position
                    # Check SL
                    if high_price >= sl_price:
                        exit_price = sl_price
                        triggered_exit = True
                        exit_reason = "STOP_LOSS"
                    # Check TP
                    elif low_price <= tp_price:
                        exit_price = tp_price
                        triggered_exit = True
                        exit_reason = "TAKE_PROFIT"
                
                if triggered_exit:
                    # Apply slippage (slippage goes against us)
                    if position_dir == 1:
                        fill_price = exit_price * (1 - self.slippage)
                        trade_pnl = position_size * (fill_price - entry_price)
                    else:
                        fill_price = exit_price * (1 + self.slippage)
                        trade_pnl = position_size * (entry_price - fill_price)
                    
                    fee = abs(position_size) * fill_price * self.taker_fee
                    cash = cash + trade_pnl - fee
                    current_equity = cash
                    
                    if current_trade:
                        current_trade['exit_price'] = fill_price
                        current_trade['exit_time'] = timestamp
                        current_trade['pnl'] = trade_pnl - fee
                        current_trade['exit_reason'] = exit_reason
                        trades.append(current_trade)
                        current_trade = None
                        
                    position_size = 0.0
                    position_dir = 0
            
            # 3. Check signals for entry or exit
            signal = signals.iloc[i]
            
            if position_dir == 0:  # Flat, look for entry
                if signal == 1:  # Go Long
                    # Sizing: use max_allocation * leverage
                    allocated_cash = current_equity * max_allocation
                    position_value = allocated_cash * leverage
                    
                    # Fill price with slippage (slippage pushes price up for buying)
                    fill_price = open_price * (1 + self.slippage)
                    position_size = position_value / fill_price
                    fee = position_value * self.taker_fee
                    
                    cash = current_equity - fee
                    entry_price = fill_price
                    position_dir = 1
                    
                    # Set SL/TP
                    sl_price = entry_price * (1 - stop_loss_pct)
                    tp_price = entry_price * (1 + take_profit_pct)
                    
                    current_trade = {
                        'entry_time': timestamp,
                        'entry_price': entry_price,
                        'direction': 'LONG',
                        'size': position_size,
                        'leverage': leverage
                    }
                    
                elif signal == -1 and is_futures:  # Go Short (only allowed in futures)
                    allocated_cash = current_equity * max_allocation
                    position_value = allocated_cash * leverage
                    
                    # Fill price with slippage (slippage pushes price down for selling short)
                    fill_price = open_price * (1 - self.slippage)
                    position_size = position_value / fill_price
                    fee = position_value * self.taker_fee
                    
                    cash = current_equity - fee
                    entry_price = fill_price
                    position_dir = -1
                    
                    # Set SL/TP
                    sl_price = entry_price * (1 + stop_loss_pct)
                    tp_price = entry_price * (1 - take_profit_pct)
                    
                    current_trade = {
                        'entry_time': timestamp,
                        'entry_price': entry_price,
                        'direction': 'SHORT',
                        'size': position_size,
                        'leverage': leverage
                    }
            
            else:  # Open position, check if signal opposes or requests exit
                # If signal is 0 (Close signal) or opposite of current direction
                should_close = (signal == 0) or (signal == -1 and position_dir == 1) or (signal == 1 and position_dir == -1)
                
                if should_close:
                    # Close position
                    if position_dir == 1:
                        fill_price = open_price * (1 - self.slippage)
                        trade_pnl = position_size * (fill_price - entry_price)
                    else:
                        fill_price = open_price * (1 + self.slippage)
                        trade_pnl = position_size * (entry_price - fill_price)
                        
                    fee = abs(position_size) * fill_price * self.taker_fee
                    cash = cash + trade_pnl - fee
                    current_equity = cash
                    
                    if current_trade:
                        current_trade['exit_price'] = fill_price
                        current_trade['exit_time'] = timestamp
                        current_trade['pnl'] = trade_pnl - fee
                        current_trade['exit_reason'] = 'SIGNAL_EXIT'
                        trades.append(current_trade)
                        current_trade = None
                        
                    # Re-initialize to flat
                    position_size = 0.0
                    position_dir = 0
                    
                    # If the signal was opposite, we enter immediately in the next step or right now
                    # Let's enter on the next step to keep signals aligned, or we can open immediately.
                    # Opening on next step is standard to avoid double fills in one bar.
            
            # Track equity history
            # Final equity for the current bar is cash + current unrealized pnl
            if position_dir == 1:
                unrealized_pnl = position_size * (close_price - entry_price)
            elif position_dir == -1:
                unrealized_pnl = position_size * (entry_price - close_price)
            else:
                unrealized_pnl = 0.0
                
            equity_history.append(cash + unrealized_pnl)
            timestamps.append(timestamp)
            
        # Create output dataframe
        df_equity = pd.DataFrame({
            'timestamp': timestamps,
            'equity': equity_history
        })
        df_equity['datetime'] = pd.to_datetime(df_equity['timestamp'], unit='ms')
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(df_equity, trades, df)
        
        return metrics, df_equity, trades

    def _calculate_metrics(self, df_equity, trades, df_ohlcv):
        """Calculates performance statistics from the equity curve and trade logs."""
        equity = df_equity['equity']
        
        # Calculate daily or step returns
        df_equity['returns'] = equity.pct_change().fillna(0)
        
        total_return = (equity.iloc[-1] / self.initial_capital) - 1
        
        # Calculate maximum drawdown
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_drawdown = drawdown.min()
        df_equity['drawdown'] = drawdown
        
        # Calculate CAGR (Annualized Return)
        # Determine number of days in the dataset
        if len(df_equity) > 1:
            time_diff = df_equity['datetime'].iloc[-1] - df_equity['datetime'].iloc[0]
            days = max(time_diff.total_seconds() / 86400, 1.0)
            cagr = (equity.iloc[-1] / self.initial_capital) ** (365.25 / days) - 1
        else:
            cagr = 0.0
            
        # Calculate Sharpe & Sortino Ratio
        # Assuming returns are computed per candle. We annualize them.
        # Find time difference in hours between consecutive candles to find frequency
        if len(df_equity) > 1:
            candle_diff = (df_equity['datetime'].iloc[1] - df_equity['datetime'].iloc[0]).total_seconds() / 3600.0
            candles_per_year = (365.25 * 24.0) / max(candle_diff, 0.001)
        else:
            candles_per_year = 365.25
            
        mean_return = df_equity['returns'].mean()
        std_return = df_equity['returns'].std()
        
        if std_return > 0:
            sharpe = (mean_return / std_return) * np.sqrt(candles_per_year)
        else:
            sharpe = 0.0
            
        # Sortino Ratio (only downside deviation)
        downside_returns = df_equity['returns'][df_equity['returns'] < 0]
        downside_std = downside_returns.std()
        if downside_std > 0:
            sortino = (mean_return / downside_std) * np.sqrt(candles_per_year)
        else:
            sortino = 0.0
            
        # Trade statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        gross_profit = sum([t.get('pnl', 0) for t in winning_trades])
        gross_loss = abs(sum([t.get('pnl', 0) for t in losing_trades]))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': equity.iloc[-1],
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
