import logging
import math
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

"""Advanced Market Making Strategy for Hummingbot.

This script implements a sophisticated market making strategy with adaptive parameters
based on volatility, trend analysis, and inventory management.
Features include market regime detection, machine learning optimization,
and advanced risk management techniques.
"""

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent, MarketEvent, BuyOrderCompletedEvent, SellOrderCompletedEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase


class AdvancedMarketMaker(ScriptStrategyBase):
    """
    Advanced Market Making Strategy
    
    This strategy combines:
    1. Volatility-based spread adjustment using NATR (Normalized Average True Range)
    2. Trend analysis using EMA and MACD for directional bias
    3. Inventory management to balance holdings
    4. Risk management with position sizing and maximum order limits
    5. Order level configuration for improved liquidity provision
    6. Market regime detection for adaptive parameters
    7. Machine learning optimization for parameter tuning
    8. Advanced execution logic with order type selection
    """
    # Basic strategy parameters
    bid_spread = 0.001  # Default spread, will be adjusted by volatility
    ask_spread = 0.001
    order_refresh_time = 30  # in seconds
    order_amount = 0.001  # Base order size
    create_timestamp = 0
    trading_pair = "BTC-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice

    # Candles parameters for technical analysis
    candle_exchange = "binance"
    candles_interval = "5m"  # 5-minute candles for better signals
    candles_length = 50  # Number of candles to analyze
    max_records = 1000

    # Volatility parameters
    natr_length = 14  # Length for NATR calculation
    bid_spread_scalar = 100  # Multiplier for bid spread based on volatility
    ask_spread_scalar = 100  # Multiplier for ask spread based on volatility
    max_spread = 0.05  # Maximum spread allowed (5%)
    min_spread = 0.0005  # Minimum spread (5 basis points)

    # Bollinger Bands parameters
    bb_length = 20
    bb_std_dev = 2.0
    use_bollinger_bands = True  # Enable Bollinger Bands for volatility measurement

    # Trend analysis parameters
    short_ema_length = 9
    long_ema_length = 21
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    trend_strength_impact = 0.2  # How much trend affects order sizing (0-1)

    # Inventory management parameters
    target_base_pct = 0.5  # Target percentage of portfolio in base asset
    inventory_range = 0.2  # Acceptable range around target (±20%)
    inventory_skew_enabled = True
    dynamic_inventory_skew = True  # Enable dynamic inventory targets based on market conditions
    
    # Risk management parameters
    max_order_size_pct = 0.1  # Maximum single order size as percentage of portfolio
    max_inventory_pct = 0.8  # Maximum inventory percentage in either asset
    reduce_risk_in_high_volatility = True  # Reduce position size in high volatility
    dynamic_risk_adjustment = True  # Enable dynamic risk adjustment based on market conditions
    max_drawdown_pct = 0.05  # Maximum drawdown percentage before reducing risk
    recovery_factor = 0.5  # How quickly to recover from drawdown (0-1)
    max_daily_volume_pct = 0.01  # Maximum trading volume as percentage of 24h market volume
    
    # Order levels configuration
    order_levels = 3  # Number of orders on each side
    order_level_spread_multiplier = 1.5  # Multiplier for spread between order levels
    order_level_amount_multiplier = 0.8  # Multiplier for amount in subsequent levels
    
    # Filled order handling
    filled_order_delay = 60  # Delay in seconds after an order is filled
    last_fill_timestamp = 0

    # Market regime detection parameters
    enable_regime_detection = True  # Enable market regime detection
    regime_lookback_periods = 50  # Number of periods to look back for regime detection
    n_regimes = 3  # Number of regimes to identify (low/medium/high volatility)
    regime_detection_interval = 60  # How often to update regime detection (in seconds)
    last_regime_update = 0
    current_regime = 1  # Default to medium volatility regime (0=low, 1=medium, 2=high)
    
    # ML optimization parameters
    enable_ml_optimization = True  # Enable machine learning parameter optimization
    ml_optimization_interval = 3600  # How often to run ML optimization (in seconds)
    last_ml_optimization = 0
    
    # Advanced execution parameters
    use_limit_orders_only = False  # When False, may use market orders for quick inventory rebalancing
    enable_dynamic_order_refresh = True  # Dynamically adjust order refresh time based on volatility
    minimum_spread_for_orders = 0.0001  # Minimum spread to place orders (otherwise will wait)

    # Initialize candles
    candles = CandlesFactory.get_candle(CandlesConfig(
        connector=candle_exchange,
        trading_pair=trading_pair,
        interval=candles_interval,
        max_records=max_records
    ))

    # Connect to markets
    markets = {exchange: {trading_pair}}
    
    # Performance tracking
    total_filled_orders = 0
    total_buy_volume = Decimal("0")
    total_sell_volume = Decimal("0")
    total_profit_loss = Decimal("0")
    
    # Historical data for optimization
    historical_spreads = []
    historical_performance = []
    historical_regimes = []
    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.candles.start()
        self.base_asset, self.quote_asset = self.trading_pair.split("-")
        self.log_with_clock(logging.INFO, "Advanced Market Maker initialized!")
        self.log_with_clock(logging.INFO, f"Trading {self.trading_pair} on {self.exchange}")
        
        # Subscribe to order completed events
        self.add_listener(MarketEvent.BuyOrderCompleted, self.did_complete_buy_order)
        self.add_listener(MarketEvent.SellOrderCompleted, self.did_complete_sell_order)

    def on_stop(self):
        self.candles.stop()
        self.log_with_clock(logging.INFO, "Strategy stopped. Summary:")
        self.log_with_clock(logging.INFO, f"Total filled orders: {self.total_filled_orders}")
        self.log_with_clock(logging.INFO, f"Total buy volume: {self.total_buy_volume} {self.base_asset}")
        self.log_with_clock(logging.INFO, f"Total sell volume: {self.total_sell_volume} {self.base_asset}")
        self.log_with_clock(logging.INFO, f"Estimated P&L: {self.total_profit_loss} {self.quote_asset}")

    def on_tick(self):
        """Main execution logic on each tick"""
        if not self.candles.is_ready:
            self.log_with_clock(logging.INFO, "Candles not ready yet. Waiting...")
            return
            
        current_time = self.current_timestamp
        
        # Update market regime detection periodically
        if self.enable_regime_detection and current_time >= self.last_regime_update + self.regime_detection_interval:
            self.detect_market_regime()
            self.last_regime_update = current_time
            
        # Run ML optimization periodically
        if self.enable_ml_optimization and current_time >= self.last_ml_optimization + self.ml_optimization_interval:
            self.run_ml_optimization()
            self.last_ml_optimization = current_time
            
        # Check if it's time to refresh orders
        if (current_time >= self.create_timestamp and 
                current_time >= self.last_fill_timestamp + self.filled_order_delay):
            self.cancel_all_orders()
            
            # Calculate indicators and update strategy parameters
            self.update_parameters_from_indicators()
            
            # Adjust order refresh time based on volatility if enabled
            if self.enable_dynamic_order_refresh:
                self.adjust_order_refresh_time()
            
            # Create order proposals
            proposal: List[OrderCandidate] = self.create_proposal()
            
            # Adjust orders based on budget and risk management
            proposal_adjusted: List[OrderCandidate] = self.apply_risk_management(proposal)
            
            # Place the orders
            if proposal_adjusted:
                # Check if the current market conditions indicate spread is too low
                candles_df = self.get_candles_with_features()
                if not candles_df.empty:
                    current_spread = (self.bid_spread + self.ask_spread) / 2
                    if current_spread < self.minimum_spread_for_orders:
                        self.log_with_clock(logging.INFO, 
                            f"Current spread ({current_spread:.6f}) too low, waiting for better conditions")
                        self.create_timestamp = 5 + current_time  # Try again in 5 seconds
                        return
                        
                self.place_orders(proposal_adjusted)
                self.create_timestamp = self.order_refresh_time + current_time
            else:
                self.log_with_clock(logging.WARNING, "No valid orders to place after risk adjustment")
                self.create_timestamp = 5 + current_time  # Try again in 5 seconds

    def detect_market_regime(self):
        """
        Detect current market regime using K-means clustering on price action features
        Regimes typically represent different volatility states: low, medium, high
        """
        candles_df = self.get_candles_with_features()
        
        if candles_df.empty or len(candles_df) < self.regime_lookback_periods:
            self.log_with_clock(logging.WARNING, "Not enough data for market regime detection")
            return
            
        # Extract relevant features for regime detection
        features = candles_df.iloc[-self.regime_lookback_periods:].copy()
        
        # Create feature set for clustering: volatility, trend strength, trading range
        feature_cols = [
            f"NATR_{self.natr_length}",  # Volatility
            'trend_strength_normalized',  # Trend strength
            'RSI_14'                      # Momentum
        ]
        
        # Add Bollinger Band width if available
        if 'bb_width_normalized' in features.columns:
            feature_cols.append('bb_width_normalized')
            
        # Check if all required columns exist
        missing_cols = [col for col in feature_cols if col not in features.columns]
        if missing_cols:
            self.log_with_clock(logging.WARNING, f"Missing columns for regime detection: {missing_cols}")
            return
            
        # Extract and scale features
        regime_features = features[feature_cols].copy()
        
        # Handle missing values
        regime_features = regime_features.fillna(method='ffill').fillna(method='bfill')
        
        if regime_features.isnull().any().any():
            self.log_with_clock(logging.WARNING, "Still have NaN values after filling")
            return
            
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(regime_features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        regimes = kmeans.fit_predict(scaled_features)
        
        # Determine current regime (use last value)
        self.current_regime = regimes[-1]
        
        # Identify regime characteristics
        regime_volatility = []
        for i in range(self.n_regimes):
            mask = regimes == i
            if np.any(mask):
                mean_volatility = np.mean(regime_features.iloc[mask][f"NATR_{self.natr_length}"])
                regime_volatility.append((i, mean_volatility))
                
        # Sort regimes by volatility low to high
        regime_volatility.sort(key=lambda x: x[1])
        
        # Map original regime labels to low/medium/high volatility
        regime_mapping = {regime_volatility[i][0]: i for i in range(len(regime_volatility))}
        self.current_regime = regime_mapping[self.current_regime]
        
        # Store for historical analysis
        self.historical_regimes.append(self.current_regime)
        if len(self.historical_regimes) > 100:
            self.historical_regimes.pop(0)
            
        # Log current regime
        regime_names = ["Low Volatility", "Medium Volatility", "High Volatility"]
        self.log_with_clock(logging.INFO, 
            f"Current market regime: {regime_names[self.current_regime]} (Regime {self.current_regime})")
        
        # Adjust strategy parameters based on regime
        self.adjust_parameters_for_regime()

    def adjust_parameters_for_regime(self):
        """Adjust strategy parameters based on the current market regime"""
        if self.current_regime == 0:  # Low volatility
            # In low volatility, we can use tighter spreads and more aggressive order sizing
            self.bid_spread_scalar = 80
            self.ask_spread_scalar = 80
            self.order_levels = 4  # More levels for granularity
            self.order_level_spread_multiplier = 1.3  # Tighter spread between levels
            self.max_order_size_pct = 0.12  # Slightly larger orders
            self.order_refresh_time = 45  # Less frequent updates
            
        elif self.current_regime == 1:  # Medium volatility
            # In medium volatility, use balanced parameters
            self.bid_spread_scalar = 100
            self.ask_spread_scalar = 100
            self.order_levels = 3
            self.order_level_spread_multiplier = 1.5
            self.max_order_size_pct = 0.1
            self.order_refresh_time = 30
            
        elif self.current_regime == 2:  # High volatility
            # In high volatility, use wider spreads and more conservative sizing
            self.bid_spread_scalar = 150
            self.ask_spread_scalar = 150
            self.order_levels = 2  # Fewer levels
            self.order_level_spread_multiplier = 2.0  # Wider spread between levels
            self.max_order_size_pct = 0.05  # Smaller orders
            self.order_refresh_time = 15  # More frequent updates
            
        # Adjust target inventory based on trend within regime
        if self.dynamic_inventory_skew:
            candles_df = self.get_candles_with_features()
            if not candles_df.empty:
                latest = candles_df.iloc[-1]
                trend_direction = latest['trend_direction']
                trend_strength = latest['trend_strength_normalized']
                
                # Adjust target base percentage based on trend
                # In strong uptrend, hold more base asset; in strong downtrend, hold more quote asset
                base_adjustment = trend_direction * trend_strength * 0.2  # Max 20% adjustment
                self.target_base_pct = 0.5 + base_adjustment
                self.target_base_pct = max(0.3, min(0.7, self.target_base_pct))  # Cap between 30-70%
                
                self.log_with_clock(logging.INFO, 
                    f"Adjusted target base percentage to {self.target_base_pct:.2%} based on trend")

    def run_ml_optimization(self):
        """
        Use machine learning to optimize strategy parameters based on historical performance
        """
        # Ensure we have enough historical data
        if len(self.historical_spreads) < 20 or len(self.historical_performance) < 20:
            self.log_with_clock(logging.INFO, "Not enough historical data for ML optimization yet")
            return
            
        self.log_with_clock(logging.INFO, "Running ML optimization...")
        
        try:
            # Prepare features and target variable for regression
            features = pd.DataFrame(self.historical_spreads)
            performance = pd.DataFrame({'profit': self.historical_performance})
            
            # If we have regimes data, add it as a feature
            if len(self.historical_regimes) == len(features):
                features['regime'] = self.historical_regimes
                
            # Simple linear regression for spread optimization
            model = LinearRegression()
            model.fit(features, performance)
            
            # Extract coefficients for insights
            coefficients = model.coef_[0]
            
            # Predict optimal parameters based on current market conditions
            candles_df = self.get_candles_with_features()
            if not candles_df.empty:
                latest = candles_df.iloc[-1]
                
                # Create feature vector for current conditions
                current_features = [latest[f"NATR_{self.natr_length}"], 
                                   latest['trend_strength_normalized']]
                
                # Add regime if available
                if 'regime' in features.columns:
                    current_features.append(self.current_regime)
                    
                # Predict optimal spread for current conditions
                optimal_spread = model.predict([current_features])[0][0]
                
                # Apply constraints and update parameters
                optimal_spread = max(self.min_spread, min(self.max_spread, optimal_spread))
                
                # Update spread parameters
                base_spread = optimal_spread / 2  # Split between bid and ask
                self.bid_spread = base_spread
                self.ask_spread = base_spread
                
                self.log_with_clock(logging.INFO, 
                    f"ML optimization updated spread to {optimal_spread:.6f} " +
                    f"(bid: {self.bid_spread:.6f}, ask: {self.ask_spread:.6f})")
                
        except Exception as e:
            self.log_with_clock(logging.ERROR, f"Error in ML optimization: {e}")

    def adjust_order_refresh_time(self):
        """Dynamically adjust order refresh time based on volatility"""
        candles_df = self.get_candles_with_features()
        if candles_df.empty:
            return
            
        latest = candles_df.iloc[-1]
        natr = latest[f"NATR_{self.natr_length}"]
        
        # Calculate volatility percentile
        natr_series = candles_df[f"NATR_{self.natr_length}"].dropna()
        if len(natr_series) > 0:
            natr_90th_percentile = natr_series.quantile(0.9)
            volatility_ratio = natr / natr_90th_percentile if natr_90th_percentile > 0 else 1
            
            # Adjust refresh time based on volatility
            # More volatile = more frequent refresh
            base_refresh_time = 30  # Base refresh time (seconds)
            
            if volatility_ratio > 1.5:  # Very high volatility
                self.order_refresh_time = max(10, base_refresh_time // 3)
            elif volatility_ratio > 1.0:  # High volatility
                self.order_refresh_time = max(15, base_refresh_time // 2)
            elif volatility_ratio < 0.5:  # Low volatility
                self.order_refresh_time = min(60, base_refresh_time * 2)
            else:  # Normal volatility
                self.order_refresh_time = base_refresh_time
                
            self.log_with_clock(logging.INFO, 
                f"Adjusted order refresh time to {self.order_refresh_time}s based on volatility ratio {volatility_ratio:.2f}")

    def get_candles_with_features(self) -> pd.DataFrame:
        """Add technical indicators to candles dataframe"""
        candles_df = self.candles.candles_df
        
        if candles_df.empty or len(candles_df) < self.candles_length:
            return pd.DataFrame()
            
        # Volatility indicator - NATR
        candles_df.ta.natr(length=self.natr_length, scalar=1, append=True)
        
        # Bollinger Bands for additional volatility measurement
        if self.use_bollinger_bands:
            candles_df.ta.bbands(length=self.bb_length, std=self.bb_std_dev, append=True)
            # Calculate BB width as a percentage of price
            bb_col_prefix = f"BBB_{self.bb_length}_{self.bb_std_dev}"
            candles_df['bb_width'] = (
                (candles_df[f"{bb_col_prefix}_upper"] - candles_df[f"{bb_col_prefix}_lower"]) / 
                candles_df[f"{bb_col_prefix}_mid"]
            )
            # Normalize BB width for easier usage
            candles_df['bb_width_normalized'] = (
                candles_df['bb_width'] / 
                candles_df['bb_width'].rolling(window=self.candles_length).mean()
            )
        
        # Trend indicators
        # EMA
        candles_df.ta.ema(length=self.short_ema_length, append=True)
        candles_df.ta.ema(length=self.long_ema_length, append=True)
        
        # MACD
        candles_df.ta.macd(
            fast=self.macd_fast, 
            slow=self.macd_slow, 
            signal=self.macd_signal, 
            append=True
        )
        
        # RSI for additional trend confirmation
        candles_df.ta.rsi(length=14, append=True)
        
        # Add derived features
        candles_df['trend_direction'] = np.where(
            candles_df[f'EMA_{self.short_ema_length}'] > candles_df[f'EMA_{self.long_ema_length}'], 
            1,  # Uptrend
            -1  # Downtrend
        )
        
        # Trend strength based on MACD histogram (signal - MACD)
        macd_hist_col = f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
        candles_df['trend_strength'] = candles_df[macd_hist_col].abs()
        
        # Scale trend strength to 0-1 range for easier usage
        max_strength = candles_df['trend_strength'].rolling(window=self.candles_length).max()
        candles_df['trend_strength_normalized'] = candles_df['trend_strength'] / max_strength.replace(0, 1)
        
        # Calculate dynamic bid/ask spreads based on volatility
        natr = candles_df[f"NATR_{self.natr_length}"].iloc[-1]
        candles_df['bid_spread'] = candles_df[f"NATR_{self.natr_length}"] * self.bid_spread_scalar
        candles_df['ask_spread'] = candles_df[f"NATR_{self.natr_length}"] * self.ask_spread_scalar
        
        # Apply min/max spread constraints
        candles_df['bid_spread'] = candles_df['bid_spread'].clip(self.min_spread, self.max_spread)
        candles_df['ask_spread'] = candles_df['ask_spread'].clip(self.min_spread, self.max_spread)
        
        # Add a volume-based indicator - On-Balance Volume (OBV)
        if 'volume' in candles_df.columns:
            candles_df.ta.obv(append=True)
            
        # Add ATR bands (like Keltner Channels)
        candles_df.ta.atr(length=14, append=True)
        candles_df['atr_upper'] = candles_df['close'] + 2 * candles_df[f'ATR_14']
        candles_df['atr_lower'] = candles_df['close'] - 2 * candles_df[f'ATR_14']
        
        # Calculate price velocity (rate of price change)
        candles_df['price_velocity'] = candles_df['close'].pct_change(5)  # 5-period rate of change
        
        return candles_df

    def update_parameters_from_indicators(self):
        """Update strategy parameters based on technical indicators"""
        candles_df = self.get_candles_with_features()
        
        if candles_df.empty:
            self.log_with_clock(logging.WARNING, "Not enough candle data to calculate indicators")
            return
            
        # Get latest values from indicators
        latest = candles_df.iloc[-1]
        
        # Update spreads based on volatility (NATR)
        natr_col = f"NATR_{self.natr_length}"
        base_bid_spread = max(
            self.min_spread, 
            min(self.max_spread, latest[natr_col] * self.bid_spread_scalar)
        )
        base_ask_spread = max(
            self.min_spread, 
            min(self.max_spread, latest[natr_col] * self.ask_spread_scalar)
        )
        
        # Adjust spreads using Bollinger Bands if enabled
        if self.use_bollinger_bands and 'bb_width_normalized' in latest:
            # If BB width is wider than average, increase spreads proportionally
            bb_width_factor = latest['bb_width_normalized']
            # Cap the factor to prevent extreme values
            bb_width_factor = min(3.0, max(0.5, bb_width_factor))
            
            # Apply Bollinger Band width to spreads
            base_bid_spread = base_bid_spread * bb_width_factor
            base_ask_spread = base_ask_spread * bb_width_factor
            
            # Check if price is near Bollinger Bands edges
            bb_col_prefix = f"BBB_{self.bb_length}_{self.bb_std_dev}"
            current_price = self.get_mid_price()
            bb_upper = latest[f"{bb_col_prefix}_upper"]
            bb_lower = latest[f"{bb_col_prefix}_lower"]
            bb_mid = latest[f"{bb_col_prefix}_mid"]
            
            # Calculate how close price is to BB edges (0 = at middle, 1 = at edge)
            price_relative_to_bb = abs((current_price - bb_mid) / (bb_upper - bb_lower) * 2)
            
            # If price is near BB edge, adjust spreads to prepare for potential reversal
            if price_relative_to_bb > 0.8:
                # If near upper band, tighten ask spread and widen bid spread
                if current_price > bb_mid:
                    base_ask_spread = base_ask_spread * 0.8  # Tighten ask spread
                    base_bid_spread = base_bid_spread * 1.2  # Widen bid spread
                # If near lower band, tighten bid spread and widen ask spread
                else:
                    base_bid_spread = base_bid_spread * 0.8  # Tighten bid spread
                    base_ask_spread = base_ask_spread * 1.2  # Widen ask spread
        
        # Adjust spreads based on trend direction and strength
        trend_direction = latest['trend_direction']
        trend_strength = latest['trend_strength_normalized']
        trend_factor = trend_direction * trend_strength * self.trend_strength_impact
        
        # In uptrend, tighten ask spread and widen bid spread
        # In downtrend, tighten bid spread and widen ask spread
        self.bid_spread = base_bid_spread * (1 - trend_factor)
        self.ask_spread = base_ask_spread * (1 + trend_factor)
        
        # Ensure spreads are within limits
        self.bid_spread = max(self.min_spread, min(self.max_spread, self.bid_spread))
        self.ask_spread = max(self.min_spread, min(self.max_spread, self.ask_spread))
        
        # Logging the updated parameters for debugging
        self.log_with_clock(
            logging.INFO, 
            f"Updated parameters - NATR: {latest[natr_col]:.6f}, "
            f"Trend: {trend_direction} (Strength: {trend_strength:.2f}), "
            f"Bid spread: {self.bid_spread:.6f}, Ask spread: {self.ask_spread:.6f}"
        )

    def calculate_inventory_skew(self) -> float:
        """Calculate the inventory skew factor (-1 to 1) based on current inventory"""
        if not self.inventory_skew_enabled:
            return 0.0
            
        base_balance = self.get_balance(self.exchange, self.base_asset)
        quote_balance = self.get_balance(self.exchange, self.quote_asset)
        
        if base_balance == 0 or quote_balance == 0:
            return 0.0
            
        # Get the current market price to convert quote to base
        price = self.get_mid_price()
        if price == 0:
            return 0.0
            
        base_value = base_balance
        quote_value_in_base = quote_balance / price
        total_value_in_base = base_value + quote_value_in_base
        
        if total_value_in_base == 0:
            return 0.0
            
        current_base_pct = base_value / total_value_in_base
        
        # Calculate skew (-1 to 1)
        # -1 means we have too much base asset, +1 means too much quote asset
        inventory_skew = (self.target_base_pct - current_base_pct) / self.inventory_range
        
        # Clamp between -1 and 1
        return max(-1.0, min(1.0, inventory_skew))

    def get_mid_price(self) -> Decimal:
        """Get the current mid price"""
        return self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)

    def create_proposal(self) -> List[OrderCandidate]:
        """Create a list of order candidates based on current market conditions"""
        ref_price = self.get_mid_price()
        inventory_skew = self.calculate_inventory_skew()
        
        # Adjust spreads based on inventory skew
        adjusted_bid_spread = self.bid_spread * (1 - inventory_skew * 0.5)
        adjusted_ask_spread = self.ask_spread * (1 + inventory_skew * 0.5)
        
        # Make sure spreads are positive
        adjusted_bid_spread = max(0.00001, adjusted_bid_spread)
        adjusted_ask_spread = max(0.00001, adjusted_ask_spread)
        
        # Get reference prices
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)
        
        orders = []
        
        # Create multiple order levels
        for level in range(self.order_levels):
            # Calculate spread for this level
            level_multiplier = 1 + (level * (self.order_level_spread_multiplier - 1))
            level_bid_spread = adjusted_bid_spread * level_multiplier
            level_ask_spread = adjusted_ask_spread * level_multiplier
            
            # Calculate order sizes for this level
            size_multiplier = self.order_level_amount_multiplier ** level
            level_order_amount = Decimal(str(self.order_amount * size_multiplier))
            
            # Skip tiny orders
            if level_order_amount < Decimal("0.0001"):
                continue
                
            # Calculate prices
            bid_price = ref_price * Decimal(1 - level_bid_spread)
            ask_price = ref_price * Decimal(1 + level_ask_spread)
            
            # Make sure our orders don't cross the orderbook
            bid_price = min(bid_price, best_bid)
            ask_price = max(ask_price, best_ask)
            
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
            
            # Add to our proposal
            orders.append(buy_order)
            orders.append(sell_order)
        
        return orders

    def apply_risk_management(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """Apply risk management rules to order proposals"""
        if not proposal:
            return []
            
        # First check budget constraints
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=False)
        
        # Apply risk management rules
        risk_managed_proposals = []
        
        # Get total portfolio value in quote asset
        base_balance = self.get_balance(self.exchange, self.base_asset)
        quote_balance = self.get_balance(self.exchange, self.quote_asset)
        mid_price = self.get_mid_price()
        
        base_value_in_quote = base_balance * mid_price
        total_portfolio_value_in_quote = quote_balance + base_value_in_quote
        
        # Calculate current exposure percentages
        base_exposure_pct = base_value_in_quote / total_portfolio_value_in_quote if total_portfolio_value_in_quote > 0 else Decimal("0")
        quote_exposure_pct = quote_balance / total_portfolio_value_in_quote if total_portfolio_value_in_quote > 0 else Decimal("0")
        
        # Maximum size for a single order based on portfolio value
        max_order_value = total_portfolio_value_in_quote * Decimal(str(self.max_order_size_pct))
        
        # Check volatility for additional risk reduction
        candles_df = self.get_candles_with_features()
        current_natr = Decimal(str(candles_df[f"NATR_{self.natr_length}"].iloc[-1]))
        
        # Calculate historical volatility percentiles
        natr_series = candles_df[f"NATR_{self.natr_length}"].dropna()
        if len(natr_series) > 0:
            natr_90th_percentile = Decimal(str(natr_series.quantile(0.9)))
            volatility_ratio = current_natr / natr_90th_percentile if natr_90th_percentile > 0 else Decimal("1")
        else:
            volatility_ratio = Decimal("1")
        
        # Apply volatility-based risk reduction
        risk_scalar = Decimal("1")
        if self.reduce_risk_in_high_volatility and volatility_ratio > Decimal("1"):
            # Reduce position size when volatility is above 90th percentile
            risk_scalar = Decimal("1") / volatility_ratio
            risk_scalar = max(Decimal("0.25"), risk_scalar)  # Don't reduce by more than 75%
        
        # Apply dynamic risk adjustment based on recent performance if enabled
        if self.dynamic_risk_adjustment:
            # Check for drawdown by analyzing recent price movement
            if len(candles_df) >= 24:  # At least 24 candles (2 hours with 5-min candles)
                # Calculate highest price in last 24 candles
                highest_price = candles_df['close'].iloc[-24:].max()
                current_price = candles_df['close'].iloc[-1]
                
                # Calculate drawdown
                drawdown = (highest_price - current_price) / highest_price if highest_price > 0 else 0
                
                # If drawdown exceeds threshold, reduce risk further
                if drawdown > self.max_drawdown_pct:
                    drawdown_factor = 1 - ((drawdown - self.max_drawdown_pct) / self.max_drawdown_pct) * self.recovery_factor
                    drawdown_factor = max(Decimal("0.25"), min(Decimal("1"), Decimal(str(drawdown_factor))))
                    risk_scalar = risk_scalar * drawdown_factor
                    
                    self.log_with_clock(
                        logging.INFO,
                        f"Reducing risk due to drawdown of {drawdown:.2%}, factor: {drawdown_factor}"
                    )
        
        # Get 24h trading volume for limiting our orders (to avoid market impact)
        try:
            ticker = self.connectors[self.exchange].get_ticker(self.trading_pair)
            market_volume_24h = ticker.volume
            max_order_size_by_volume = Decimal(str(market_volume_24h * self.max_daily_volume_pct))
        except Exception as e:
            self.log_with_clock(logging.WARNING, f"Could not get 24h volume, using default risk parameters: {e}")
            max_order_size_by_volume = Decimal("999999")  # Large default value
        
        for order in proposal_adjusted:
            order_amount = order.amount
            order_value = order_amount * order.price if order.order_side == TradeType.BUY else order_amount * mid_price
            
            # Skip orders that are too large based on portfolio
            if order_value > max_order_value * risk_scalar:
                reduced_amount = (max_order_value * risk_scalar) / order.price if order.order_side == TradeType.BUY else (max_order_value * risk_scalar) / mid_price
                order.amount = reduced_amount
                self.log_with_clock(logging.INFO, f"Reduced order size to {reduced_amount} due to portfolio risk limit")
            
            # Limit order size based on market volume to avoid market impact
            if order.amount > max_order_size_by_volume:
                order.amount = max_order_size_by_volume
                self.log_with_clock(logging.INFO, f"Reduced order size to {max_order_size_by_volume} due to volume limit")
            
            # Check inventory limits
            if order.order_side == TradeType.BUY:
                # Skip if buying would exceed maximum base asset allocation
                if base_exposure_pct >= Decimal(str(self.max_inventory_pct)):
                    self.log_with_clock(logging.INFO, f"Skipping buy order due to max base asset exposure ({base_exposure_pct:.2%})")
                    continue
                
                # Adjust amount if buying would exceed maximum base asset allocation
                new_base_value = base_value_in_quote + order_value
                new_base_pct = new_base_value / (total_portfolio_value_in_quote + order_value - order_value)
                
                if new_base_pct > Decimal(str(self.max_inventory_pct)):
                    # Calculate the maximum amount that would keep us within inventory limits
                    max_additional_base_value = (Decimal(str(self.max_inventory_pct)) * total_portfolio_value_in_quote) - base_value_in_quote
                    max_amount = max_additional_base_value / order.price
                    
                    if max_amount > Decimal("0"):
                        order.amount = min(order.amount, max_amount)
                        self.log_with_clock(logging.INFO, f"Adjusted buy amount to {order.amount} to stay within inventory limits")
                    else:
                        continue  # Skip this order
                
            elif order.order_side == TradeType.SELL:
                # Skip if selling would exceed maximum quote asset allocation
                if quote_exposure_pct >= Decimal(str(self.max_inventory_pct)):
                    self.log_with_clock(logging.INFO, f"Skipping sell order due to max quote asset exposure ({quote_exposure_pct:.2%})")
                    continue
                
                # Adjust amount if selling would exceed maximum quote asset allocation
                new_quote_balance = quote_balance + (order_amount * order.price)
                new_base_balance = base_balance - order_amount
                new_total_value = new_quote_balance + (new_base_balance * mid_price)
                new_quote_pct = new_quote_balance / new_total_value if new_total_value > 0 else Decimal("0")
                
                if new_quote_pct > Decimal(str(self.max_inventory_pct)):
                    # Calculate the maximum amount that would keep us within inventory limits
                    current_quote_asset_value = quote_balance
                    max_quote_asset_value = Decimal(str(self.max_inventory_pct)) * total_portfolio_value_in_quote
                    max_additional_quote = max_quote_asset_value - current_quote_asset_value
                    max_amount = max_additional_quote / order.price
                    
                    if max_amount > Decimal("0"):
                        order.amount = min(order.amount, max_amount)
                        self.log_with_clock(logging.INFO, f"Adjusted sell amount to {order.amount} to stay within inventory limits")
                    else:
                        continue  # Skip this order
            
            # Final check for minimum order size
            if order.amount < Decimal("0.0001"):  # Minimum amount for BTC
                self.log_with_clock(logging.INFO, f"Skipping order with too small amount: {order.amount}")
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

    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order filled event"""
        msg = (f"{event.trade_type.name} {round(event.amount, 4)} {event.trading_pair} {self.exchange} at {round(event.price, 2)}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)
        
        # Track order statistics
        self.total_filled_orders += 1
        if event.trade_type == TradeType.BUY:
            self.total_buy_volume += event.amount
        else:
            self.total_sell_volume += event.amount
            
        # Calculate and track profit/loss (simplistic P&L for tracking)
        mid_price = self.get_mid_price()
        
        if event.trade_type == TradeType.BUY:
            # For buys, unrealized P&L is current mid price vs. purchase price
            trade_value = event.amount * event.price
            current_value = event.amount * mid_price
            unrealized_pl = current_value - trade_value
        else:  # SELL
            # For sells, realized P&L is selling price vs. mid price
            trade_value = event.amount * event.price
            mid_value = event.amount * mid_price
            unrealized_pl = trade_value - mid_value
            
        self.total_profit_loss += Decimal(str(unrealized_pl))
        
        # Track spread and performance for ML optimization
        spread_used = self.bid_spread if event.trade_type == TradeType.BUY else self.ask_spread
        natr = Decimal("0")
        trend_strength = Decimal("0")
        
        candles_df = self.get_candles_with_features()
        if not candles_df.empty:
            latest = candles_df.iloc[-1]
            natr = Decimal(str(latest[f"NATR_{self.natr_length}"]))
            trend_strength = Decimal(str(latest['trend_strength_normalized']))
            
        self.historical_spreads.append([float(natr), float(trend_strength)])
        self.historical_performance.append(float(unrealized_pl))
        
        # Trim historical data if too long
        max_history = 100
        if len(self.historical_spreads) > max_history:
            self.historical_spreads = self.historical_spreads[-max_history:]
            self.historical_performance = self.historical_performance[-max_history:]
            
        # Update last fill timestamp to delay new orders
        self.last_fill_timestamp = self.current_timestamp
        
        # After a fill, consider executing a rebalancing trade if inventory is skewed
        self.consider_rebalancing_after_fill(event)

    def consider_rebalancing_after_fill(self, event: OrderFilledEvent):
        """Consider executing a rebalancing trade after a fill if inventory is significantly skewed"""
        # Skip if not using dynamic inventory management
        if not self.dynamic_inventory_skew:
            return
            
        # Calculate current inventory skew
        inventory_skew = self.calculate_inventory_skew()
        
        # If inventory is significantly skewed, consider a rebalancing trade
        if abs(inventory_skew) > 0.7:  # 70% skew threshold
            self.log_with_clock(logging.INFO, 
                f"Considering rebalancing trade due to high inventory skew: {inventory_skew:.2f}")
            
            # Calculate the direction and size of rebalancing trade
            base_balance = self.get_balance(self.exchange, self.base_asset)
            quote_balance = self.get_balance(self.exchange, self.quote_asset)
            mid_price = self.get_mid_price()
            
            base_value_in_quote = base_balance * mid_price
            total_value = base_value_in_quote + quote_balance
            
            # Calculate current and target base value
            current_base_value = base_value_in_quote
            target_base_value = total_value * Decimal(str(self.target_base_pct))
            
            # Calculate rebalancing amount
            rebalance_value = target_base_value - current_base_value
            
            # Only rebalance if the amount is significant
            min_rebalance_threshold = total_value * Decimal("0.05")  # 5% of portfolio
            
            if abs(rebalance_value) > min_rebalance_threshold:
                # Determine trade direction and size
                if rebalance_value > 0:  # Need to buy base asset
                    amount = rebalance_value / mid_price
                    # Limit to 50% of the calculated rebalance to avoid over-correction
                    amount = amount * Decimal("0.5")
                    
                    # Create market buy order if not using limit orders only
                    if not self.use_limit_orders_only:
                        self.log_with_clock(logging.INFO, 
                            f"Executing rebalancing market buy of {amount} {self.base_asset}")
                        self.buy(
                            connector_name=self.exchange,
                            trading_pair=self.trading_pair,
                            amount=amount,
                            order_type=OrderType.MARKET,
                            price=mid_price * Decimal("1.005")  # Slight buffer for market orders
                        )
                    else:
                        # Create aggressive limit buy
                        buy_price = mid_price * Decimal("1.001")  # Very tight spread
                        self.log_with_clock(logging.INFO, 
                            f"Executing rebalancing limit buy of {amount} {self.base_asset} at {buy_price}")
                        self.buy(
                            connector_name=self.exchange,
                            trading_pair=self.trading_pair,
                            amount=amount,
                            order_type=OrderType.LIMIT,
                            price=buy_price
                        )
                        
                else:  # Need to sell base asset
                    amount = abs(rebalance_value) / mid_price
                    # Limit to 50% of the calculated rebalance to avoid over-correction
                    amount = amount * Decimal("0.5")
                    
                    # Create market sell order if not using limit orders only
                    if not self.use_limit_orders_only:
                        self.log_with_clock(logging.INFO, 
                            f"Executing rebalancing market sell of {amount} {self.base_asset}")
                        self.sell(
                            connector_name=self.exchange,
                            trading_pair=self.trading_pair,
                            amount=amount,
                            order_type=OrderType.MARKET,
                            price=mid_price * Decimal("0.995")  # Slight buffer for market orders
                        )
                    else:
                        # Create aggressive limit sell
                        sell_price = mid_price * Decimal("0.999")  # Very tight spread
                        self.log_with_clock(logging.INFO, 
                            f"Executing rebalancing limit sell of {amount} {self.base_asset} at {sell_price}")
                        self.sell(
                            connector_name=self.exchange,
                            trading_pair=self.trading_pair,
                            amount=amount,
                            order_type=OrderType.LIMIT,
                            price=sell_price
                        )
    
    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        """Handle completed buy order"""
        self.log_with_clock(logging.INFO, f"Buy order completed - traded {event.base_asset_amount} {self.base_asset}")
    
    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        """Handle completed sell order"""
        self.log_with_clock(logging.INFO, f"Sell order completed - traded {event.base_asset_amount} {self.base_asset}")
        
    def get_balance(self, exchange: str, token: str) -> Decimal:
        """Safe way to get balance"""
        try:
            return self.connectors[exchange].get_balance(token)
        except Exception as e:
            self.log_with_clock(logging.ERROR, f"Error getting balance for {token}: {e}")
            return Decimal("0")

    def format_status(self) -> str:
        """Return status of the strategy for display"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
            
        lines = []
        
        # Show strategy parameters section
        lines.extend(["", "  Strategy Parameters:"])
        lines.extend([f"    Trading Pair: {self.trading_pair} on {self.exchange}"])
        lines.extend([f"    Bid Spread: {self.bid_spread:.6f}, Ask Spread: {self.ask_spread:.6f}"])
        lines.extend([f"    Order Refresh Time: {self.order_refresh_time} seconds"])
        lines.extend([f"    Base Order Amount: {self.order_amount} {self.base_asset}"])
        lines.extend([f"    Order Levels: {self.order_levels}"])
        
        # Market regime information
        regime_names = ["Low Volatility", "Medium Volatility", "High Volatility"]
        if self.enable_regime_detection:
            lines.extend([f"    Current Market Regime: {regime_names[self.current_regime]}"])
        
        # Show balances
        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + 
                    ["    " + line for line in balance_df.to_string(index=False).split("\n")])
        
        # Show inventory metrics
        base_balance = self.get_balance(self.exchange, self.base_asset)
        quote_balance = self.get_balance(self.exchange, self.quote_asset)
        mid_price = self.get_mid_price()
        
        total_value_in_quote = quote_balance + (base_balance * mid_price)
        base_pct = ((base_balance * mid_price) / total_value_in_quote 
                    if total_value_in_quote > 0 else 0)
        quote_pct = quote_balance / total_value_in_quote if total_value_in_quote > 0 else 0
        
        lines.extend(["", "  Inventory:"])
        lines.extend([f"    Base Asset ({self.base_asset}): {base_balance:.4f} ({base_pct:.2%})"])
        lines.extend([f"    Quote Asset ({self.quote_asset}): {quote_balance:.4f} ({quote_pct:.2%})"])
        lines.extend([f"    Target Base Percentage: {self.target_base_pct:.2%}"])
        lines.extend([f"    Inventory Skew: {self.calculate_inventory_skew():.4f}"])
        
        # Show performance metrics
        lines.extend(["", "  Performance:"])
        lines.extend([f"    Total Orders Filled: {self.total_filled_orders}"])
        lines.extend([f"    Total Buy Volume: {self.total_buy_volume} {self.base_asset}"])
        lines.extend([f"    Total Sell Volume: {self.total_sell_volume} {self.base_asset}"])
        lines.extend([f"    Estimated P&L: {self.total_profit_loss} {self.quote_asset}"])
        
        # Add active orders
        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + 
                        ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])
        
        # Show technical indicators section (abbreviated version)
        try:
            candles_df = self.get_candles_with_features()
            if not candles_df.empty:
                latest = candles_df.iloc[-1]
                
                lines.extend(["\n----------------------------------------------------------------------\n"])
                lines.extend(["  Technical Indicators:"])
                lines.extend([f"    NATR: {latest[f'NATR_{self.natr_length}']:.6f}"])
                
                if self.use_bollinger_bands:
                    bb_col_prefix = f"BBB_{self.bb_length}_{self.bb_std_dev}"
                    lines.extend([f"    BB Width: {latest['bb_width_normalized']:.2f}"])
                
                lines.extend([f"    Trend: {'Up' if latest['trend_direction'] > 0 else 'Down'} (Strength: {latest['trend_strength_normalized']:.2f})"])
                lines.extend([f"    RSI: {latest['RSI_14']:.2f}"])
        except Exception as e:
            lines.extend(["", f"  Failed to get indicators: {e}"])
        
        return "\n".join(lines) 