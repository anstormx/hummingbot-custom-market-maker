import logging
import math
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import pandas_ta as ta

"""PMMV2 for Hummingbot.

This strategy implements an adaptive market making approach that combines:
1. Volatility-based spread adjustment using ATR and Bollinger Bands
2. Trend analysis using EMA crossovers and RSI
3. Dynamic inventory management
4. Sophisticated risk management framework
5. Market regime detection
"""

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent, MarketEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase


class PMMV2(ScriptStrategyBase):
    """
    PMMV2 for Hummingbot.
    
    A sophisticated market making approach that adapts to changing market conditions
    by incorporating volatility metrics, trend indicators, and dynamic risk management.
    """
    # Strategy Parameters
    # Market configuration
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice
    
    # Order parameters
    bid_spread = 0.001
    ask_spread = 0.001
    order_refresh_time = 30  # in seconds
    order_amount = 0.01
    max_order_age = 180  # Cancel orders older than this (seconds)
    
    # Order levels configuration
    order_levels = 3
    level_spread_multiplier = 1.3
    level_amount_multiplier = 1.5
    
    # Candles configuration
    candle_exchange = "binance"
    candles_interval = "5m"
    max_records = 100
    
    # Volatility indicators parameters
    atr_length = 14
    bb_length = 20
    bb_std = 2.0
    
    # Trend indicators parameters
    short_ema = 9
    long_ema = 21
    rsi_length = 14
    rsi_overbought = 70
    rsi_oversold = 30
    
    # Market regime thresholds
    low_volatility_threshold = 0.005  # 0.5%
    high_volatility_threshold = 0.015  # 1.5%
    
    # Spread adjustment factors
    min_spread = 0.0001  # 1 basis point
    max_spread = 0.01    # 1%
    volatility_multiplier = 35
    trend_factor = 0.2   # How much trend affects spreads
    
    # Inventory management
    target_base_pct = 0.5  # Target 50% in base asset
    inventory_range = 0.2  # Acceptable range around target (±20%)
    max_inventory_skew = 0.8  # Maximum allowed skew
    inventory_adjustment_factor = 0.5  # More aggressive adjustment
    
    # Risk management
    max_order_size_pct = 0.05  # Max single order size as % of portfolio
    reduce_orders_in_high_volatility = True
    order_size_volatility_adjustment = 0.5  # Reduce by this factor in high volatility
    
    # Performance tracking
    total_filled_orders = 0
    total_buy_volume = Decimal("0")
    total_sell_volume = Decimal("0")
    
    # Internal state
    create_timestamp = 0
    current_volatility = Decimal("0")
    current_market_regime = "normal"  # "low", "normal", "high"
    current_trend = "neutral"  # "up", "down", "neutral"
    
    # Initialize candles
    candles = CandlesFactory.get_candle(CandlesConfig(
        connector=candle_exchange,
        trading_pair=trading_pair,
        interval=candles_interval,
        max_records=max_records
    ))
    
    # Connect to markets
    markets = {exchange: {trading_pair}}
    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        """Initialize the strategy"""
        super().__init__(connectors)
        self.candles.start()
        self.base_asset, self.quote_asset = self.trading_pair.split("-")
        
        self.log_with_clock(logging.INFO, f"Pure Market Maker initialized for {self.trading_pair}")
        
    def on_stop(self):
        """Clean up when the strategy stops"""
        self.candles.stop()
        self.log_with_clock(logging.INFO, "Strategy stopped")
        
    def on_tick(self):
        """Main execution logic on each tick"""
        # Check if candles data is available
        if self.candles.candles_df.empty:
            return
            
        current_time = self.current_timestamp
        
        # Check if price has moved significantly
        self.cancel_orders_on_price_movement()
        
        # Cancel old orders
        self.cancel_old_orders()
        
        # Check if it's time to refresh orders
        if current_time >= self.create_timestamp:
            self.cancel_all_orders()
            
            # Analyze market conditions
            self.analyze_market_conditions()
            
            # Create and place orders
            proposal = self.create_proposal()
            proposal_adjusted = self.adjust_proposal_to_budget(proposal)
            
            if proposal_adjusted:
                self.place_orders(proposal_adjusted)
                self.create_timestamp = current_time + self.order_refresh_time
                
            # Place market orders during high volatility if trending
            if self.current_market_regime == "high" and self.current_trend != "neutral":
                self.place_market_order_in_trend_direction()
                
    def analyze_market_conditions(self):
        """Analyze market conditions to determine volatility, trend, and pure parameters"""
        candles_df = self.get_candles_with_indicators()
        
        if candles_df.empty:
            return
            
        latest = candles_df.iloc[-1]
        
        # Determine volatility
        atr = latest[f'ATR_{self.atr_length}']
        close_price = latest['close']
        self.current_volatility = Decimal(str(atr / close_price)) # Current volatility based on ATR
        
        # Determine market regime based on volatility
        if self.current_volatility < Decimal(str(self.low_volatility_threshold)):
            self.current_market_regime = "low"
        elif self.current_volatility > Decimal(str(self.high_volatility_threshold)):
            self.current_market_regime = "high"
        else:
            self.current_market_regime = "normal"
            
        # Determine trend
        short_ema = latest[f'EMA_{self.short_ema}']
        long_ema = latest[f'EMA_{self.long_ema}']
        rsi = latest[f'RSI_{self.rsi_length}']
        
        # Trend determination using EMA crossover and RSI
        if short_ema > long_ema and rsi > 60:
            self.current_trend = "up"
        elif short_ema < long_ema and rsi < 40:
            self.current_trend = "down"
        else:
            self.current_trend = "neutral"
            
        # Adjust parameters based on market conditions
        self.adjust_parameters()
        
        self.log_with_clock(
            logging.INFO,
            f"Market analysis - Volatility: {self.current_volatility:.4f} ({self.current_market_regime}), "
            f"Trend: {self.current_trend}, RSI: {rsi:.1f}"
        )
        
    def adjust_parameters(self):
        """Adjust strategy parameters based on market conditions"""
        # Check if we're in active trading hours and adjust accordingly
        if self.is_active_trading_hour():
            # Make spreads tighter during active hours
            self.min_spread = 0.0001  # 1 basis point during active hours
        else:
            # Wider spreads during less active hours
            self.min_spread = 0.0003  # 3 basis points during less active hours
        
        # Base spread on volatility
        current_volatility_float = float(self.current_volatility)
        base_spread = min(self.max_spread, max(self.min_spread, 
                         current_volatility_float * self.volatility_multiplier))
        
        # Adjust spreads based on trend
        trend_adjustment = 0
        if self.current_trend == "up":
            trend_adjustment = self.trend_factor
        elif self.current_trend == "down":
            trend_adjustment = -self.trend_factor
            
        # EMA
        # In uptrend, tighten ask spread (sell higher) and widen bid spread (don't buy as high)
        # In downtrend, tighten bid spread (buy lower) and widen ask spread (don't sell as low)
        self.bid_spread = base_spread * (1 - trend_adjustment)
        self.ask_spread = base_spread * (1 + trend_adjustment)
        
        # Further adjust spreads based on inventory
        inventory_skew = self.calculate_inventory_skew()
        
        # If we have too much base asset, tighten ask spread to sell more
        # If we have too little base asset, tighten bid spread to buy more
        self.bid_spread = self.bid_spread * (1 + inventory_skew * self.inventory_adjustment_factor)
        self.ask_spread = self.ask_spread * (1 - inventory_skew * self.inventory_adjustment_factor)
        
        # More aggressive spread adjustment when inventory is severely imbalanced
        if abs(inventory_skew) > 0.5:
            # Force tighter spreads on the side that would balance inventory
            if inventory_skew > 0:  # Too little base asset, tighten bid more
                self.bid_spread = self.min_spread * 1.2
            else:  # Too much base asset, tighten ask more
                self.ask_spread = self.min_spread * 1.2
        
        # Ensure spreads are within bounds
        self.bid_spread = max(self.min_spread, min(self.max_spread, self.bid_spread))
        self.ask_spread = max(self.min_spread, min(self.max_spread, self.ask_spread))
        
        # Adjust order refresh time based on volatility
        if self.current_market_regime == "high":
            self.order_refresh_time = 15  # More frequent updates in high volatility
        elif self.current_market_regime == "low":
            self.order_refresh_time = 45  # Less frequent updates in low volatility
        else:
            self.order_refresh_time = 30  # Default
            
        # Adjust order size in high volatility if enabled
        if self.reduce_orders_in_high_volatility and self.current_market_regime == "high":
            self.order_amount = self.order_amount * self.order_size_volatility_adjustment
            
    def get_candles_with_indicators(self) -> pd.DataFrame:
        """Add technical indicators to candles dataframe"""
        candles_df = self.candles.candles_df
        
        if candles_df.empty:
            return pd.DataFrame()
        
        # Make a copy of the dataframe to avoid modifying the original
        df = candles_df.copy()
            
        # Add ATR for volatility measurement
        df['ATR_' + str(self.atr_length)] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_length)
        
        # Add Bollinger Bands
        bb = ta.bbands(df['close'], length=self.bb_length, std=self.bb_std)
        df = pd.concat([df, bb], axis=1)
        
        # Add EMAs for trend detection
        df['EMA_' + str(self.short_ema)] = ta.ema(df['close'], length=self.short_ema)
        df['EMA_' + str(self.long_ema)] = ta.ema(df['close'], length=self.long_ema)
        
        # Add RSI for trend confirmation
        df['RSI_' + str(self.rsi_length)] = ta.rsi(df['close'], length=self.rsi_length)
        
        return df
        
    def calculate_inventory_skew(self) -> float:
        """Calculate inventory skew factor (-1 to 1) to adjust spreads"""
        base_balance = self.get_balance(self.exchange, self.base_asset)
        quote_balance = self.get_balance(self.exchange, self.quote_asset)
        
        # If we don't have balances, return 0 (neutral)
        if base_balance == 0 or quote_balance == 0:
            return 0
            
        # Convert quote to base at current price
        price = self.get_mid_price()
        if price == 0:
            return 0
            
        quote_in_base = quote_balance / price
        total_in_base = base_balance + quote_in_base
        
        # Calculate current base percentage
        current_base_pct = base_balance / total_in_base if total_in_base > 0 else 0
        
        # Convert current_base_pct from Decimal to float for the calculation
        current_base_pct_float = float(current_base_pct)
        
        # Calculate skew (-1 to 1)
        inventory_skew = (self.target_base_pct - current_base_pct_float) / self.inventory_range
        
        # Clamp between -1 and 1
        return max(-1.0, min(1.0, inventory_skew))
        
    def get_mid_price(self) -> Decimal:
        """Get the current mid price from the order book"""
        return self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        
    def create_proposal(self) -> List[OrderCandidate]:
        """Create a list of order candidates based on current market conditions"""
        ref_price = self.get_mid_price()
        orders = []
        
        # Create multiple order levels
        for level in range(self.order_levels):
            # Calculate spread for this level
            level_multiplier = 1 + (level * (self.level_spread_multiplier - 1))
            level_bid_spread = self.bid_spread * level_multiplier
            level_ask_spread = self.ask_spread * level_multiplier
            
            # Calculate order amount for this level
            level_amount_multiplier = self.level_amount_multiplier ** level
            level_order_amount = Decimal(str(self.order_amount)) * Decimal(str(level_amount_multiplier))
            
            # Calculate prices
            bid_price = ref_price * (Decimal("1") - Decimal(str(level_bid_spread)))
            ask_price = ref_price * (Decimal("1") + Decimal(str(level_ask_spread)))
            
            # Create order candidates
            buy_order = OrderCandidate(
                trading_pair=self.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=level_order_amount,
                price=bid_price
            )
            
            sell_order = OrderCandidate(
                trading_pair=self.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL,
                amount=level_order_amount,
                price=ask_price
            )
            
            orders.append(buy_order)
            orders.append(sell_order)
            
        return orders
        
    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """Adjust order candidates based on available budget and risk parameters"""
        # First check budget constraints
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=False)
        
        # Apply risk management rules
        risk_managed_proposals = []
        
        # Get total portfolio value in quote asset
        base_balance = self.get_balance(self.exchange, self.base_asset)
        quote_balance = self.get_balance(self.exchange, self.quote_asset)
        mid_price = self.get_mid_price()
        
        base_value_in_quote = base_balance * mid_price
        total_portfolio_value = quote_balance + base_value_in_quote
        
        # Calculate max order size based on portfolio value and risk limit
        max_order_value = total_portfolio_value * Decimal(str(self.max_order_size_pct))
        
        # Current inventory percentages
        base_pct = base_value_in_quote / total_portfolio_value if total_portfolio_value > 0 else 0
        quote_pct = quote_balance / total_portfolio_value if total_portfolio_value > 0 else 0
        
        for order in proposal_adjusted:
            order_value = order.amount * order.price
            
            # Skip orders that are too large
            if order_value > max_order_value:
                reduced_amount = max_order_value / order.price
                order.amount = reduced_amount
                
            # Apply inventory constraints
            if order.order_side == TradeType.BUY:
                # Skip buy orders if we already have too much base asset
                if base_pct > self.max_inventory_skew:
                    continue
            else:  # SELL
                # Skip sell orders if we already have too much quote asset
                if quote_pct > self.max_inventory_skew:
                    continue
                    
            # Skip tiny orders
            if order.amount < Decimal("0.0001"):
                continue
                
            risk_managed_proposals.append(order)
            
        return risk_managed_proposals
        
    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        """Place orders from the proposal list"""
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)
            
    def place_order(self, connector_name: str, order: OrderCandidate):
        """Place a single order"""
        if order.order_side == TradeType.SELL:
            self.sell(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
        elif order.order_side == TradeType.BUY:
            self.buy(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
            
    def cancel_all_orders(self):
        """Cancel all active orders"""
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)
            
    def cancel_old_orders(self):
        """Cancel orders that have been active for too long"""
        current_time = self.current_timestamp
        for order in self.get_active_orders(connector_name=self.exchange):
            order_age = current_time - order.creation_timestamp
            if order_age > self.max_order_age:
                self.log_with_clock(logging.INFO, f"Cancelling old order {order.client_order_id}")
                self.cancel(self.exchange, order.trading_pair, order.client_order_id)
                
    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order filled event"""
        msg = (f"{event.trade_type.name} {round(event.amount, 4)} {event.trading_pair} "
               f"{self.exchange} at {round(event.price, 2)}")
        self.log_with_clock(logging.INFO, msg)
        
        # Update tracking metrics
        self.total_filled_orders += 1
        if event.trade_type == TradeType.BUY:
            self.total_buy_volume += event.amount
        else:
            self.total_sell_volume += event.amount
            
    def get_balance(self, exchange: str, token: str) -> Decimal:
        """Safe way to get balance"""
        try:
            return self.connectors[exchange].get_balance(token)
        except Exception as e:
            self.log_with_clock(logging.ERROR, f"Error getting balance for {token}: {e}")
            return Decimal("0")
            
    def format_status(self) -> str:
        """Format status for display"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
            
        lines = []
        
        # Strategy info
        lines.extend(["", "  Strategy Parameters:"])
        lines.extend([f"    Trading Pair: {self.trading_pair} on {self.exchange}"])
        lines.extend([f"    Bid Spread: {self.bid_spread:.6f}, Ask Spread: {self.ask_spread:.6f}"])
        lines.extend([f"    Order Refresh Time: {self.order_refresh_time} seconds"])
        lines.extend([f"    Order Amount: {self.order_amount} {self.base_asset}"])
        
        # Market conditions
        lines.extend(["", "  Market Conditions:"])
        lines.extend([f"    Volatility: {self.current_volatility:.4f} ({self.current_market_regime})"])
        lines.extend([f"    Trend: {self.current_trend}"])
        
        # Balances
        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + 
                   ["    " + line for line in balance_df.to_string(index=False).split("\n")])
        
        # Inventory status
        base_balance = self.get_balance(self.exchange, self.base_asset)
        quote_balance = self.get_balance(self.exchange, self.quote_asset)
        mid_price = self.get_mid_price()
        
        base_value_in_quote = base_balance * mid_price
        total_value = base_value_in_quote + quote_balance
        
        base_pct = base_value_in_quote / total_value if total_value > 0 else 0
        quote_pct = quote_balance / total_value if total_value > 0 else 0
        
        lines.extend(["", "  Inventory:"])
        lines.extend([f"    Base Asset ({self.base_asset}): {base_balance:.4f} ({base_pct:.2%})"])
        lines.extend([f"    Quote Asset ({self.quote_asset}): {quote_balance:.4f} ({quote_pct:.2%})"])
        lines.extend([f"    Target Base Percentage: {self.target_base_pct:.2%}"])
        lines.extend([f"    Inventory Skew: {self.calculate_inventory_skew():.4f}"])
        
        # Active orders
        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + 
                        ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])
            
        # Performance metrics
        lines.extend(["", "  Performance:"])
        lines.extend([f"    Total Orders Filled: {self.total_filled_orders}"])
        lines.extend([f"    Total Buy Volume: {self.total_buy_volume} {self.base_asset}"])
        lines.extend([f"    Total Sell Volume: {self.total_sell_volume} {self.base_asset}"])
        
        # Add technical indicators section
        try:
            candles_df = self.get_candles_with_indicators()
            if not candles_df.empty:
                latest = candles_df.iloc[-1]
                
                lines.extend(["\n----------------------------------------------------------------------\n"])
                lines.extend(["  Technical Indicators:"])
                lines.extend([f"    ATR: {latest.get(f'ATR_{self.atr_length}', 0):.4f}"])
                
                # Bollinger Bands columns are different with the new method
                bb_upper = latest.get(f'BBU_{self.bb_length}_{self.bb_std}', 0)
                bb_lower = latest.get(f'BBL_{self.bb_length}_{self.bb_std}', 0)
                bb_mid = latest.get(f'BBM_{self.bb_length}_{self.bb_std}', 0)
                
                if bb_mid != 0:
                    bb_width = (bb_upper - bb_lower) / bb_mid
                else:
                    bb_width = 0
                
                lines.extend([f"    BB Width: {bb_width:.4f}"])
                lines.extend([f"    EMA{self.short_ema}: {latest.get(f'EMA_{self.short_ema}', 0):.2f}"])
                lines.extend([f"    EMA{self.long_ema}: {latest.get(f'EMA_{self.long_ema}', 0):.2f}"])
                lines.extend([f"    RSI: {latest.get(f'RSI_{self.rsi_length}', 0):.2f}"])
        except Exception as e:
            lines.extend(["", f"  Failed to get indicators: {e}"])
        
        return "\n".join(lines)

    def cancel_orders_on_price_movement(self):
        """Cancel orders when price moves significantly"""
        candles_df = self.candles.candles_df
        if candles_df.empty or len(candles_df) < 2:
            return
            
        current_price = candles_df.iloc[-1]['close']
        previous_price = candles_df.iloc[-2]['close']
        price_change_pct = abs(current_price - previous_price) / previous_price
        
        # If price moved more than 0.2%, cancel all orders
        if price_change_pct > 0.002:
            self.cancel_all_orders()
            self.log_with_clock(
                logging.INFO,
                f"Canceled all orders due to price movement of {price_change_pct:.2%}"
            )
                
    def place_market_order_in_trend_direction(self):
        """Place a market order in the direction of the trend during high volatility"""
        # Only place a market order if we have a clear trend
        if self.current_trend == "up":
            # In uptrend, buy with market order
            self.log_with_clock(
                logging.INFO,
                f"Placing market BUY order during high volatility uptrend"
            )
            self.buy(
                connector_name=self.exchange,
                trading_pair=self.trading_pair,
                amount=Decimal(str(self.order_amount)),
                order_type=OrderType.MARKET,
                price=None
            )
        elif self.current_trend == "down":
            # In downtrend, sell with market order
            self.log_with_clock(
                logging.INFO,
                f"Placing market SELL order during high volatility downtrend"
            )
            self.sell(
                connector_name=self.exchange,
                trading_pair=self.trading_pair,
                amount=Decimal(str(self.order_amount)),
                order_type=OrderType.MARKET,
                price=None
            )

    def is_active_trading_hour(self):
        """Check if current time is during peak trading hours"""
        current_time = pd.Timestamp.utcnow()
        # Trading is generally more active during these UTC hours
        active_hours = [h for h in range(8, 16)]  # 8 AM to 4 PM UTC
        return current_time.hour in active_hours 