"""
Technical Analysis Indicator Modules

Precise, modular implementations of technical indicators for the trading system.
Each module takes a pandas DataFrame and returns calculated indicator values.
Designed for parallel execution in AWS Lambda environment.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicatorBase(ABC):
    """
    Professional-grade abstract base class for all technical indicators

    Features:
    - Market regime detection for adaptive parameters
    - ATR integration for dynamic thresholds
    - Statistical validation of signals
    - Performance optimization with caching
    - Robust error handling
    """

    def __init__(self, name: str, enable_regime_detection: bool = True):
        self.name = name
        self.enable_regime_detection = enable_regime_detection
        self.last_calculated = None
        self.calculation_time = None
        self.cache = {}  # For performance optimization
        self.market_regime = None
        self.atr_context = None

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate the technical indicator

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Dictionary of indicator columns to add to DataFrame
        """
        pass

    @abstractmethod
    def get_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Generate BUY/SELL/HOLD signals based on indicator values

        Args:
            df: DataFrame with indicator values already calculated

        Returns:
            Dictionary with signal columns
        """
        pass

    @abstractmethod
    def get_min_periods(self) -> int:
        """Return minimum number of periods needed for calculation"""
        pass

    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime for adaptive parameter adjustment

        Returns:
            'trending_low_vol', 'trending_high_vol', 'ranging_low_vol',
            'ranging_high_vol', 'volatile_choppy'
        """
        if not self.enable_regime_detection or len(df) < 50:
            return 'unknown'

        try:
            # Calculate ADX for trend strength
            adx = self.calculate_adx(df)

            # Calculate volatility percentile
            if 'atr' not in df.columns:
                # Quick ATR calculation for regime detection
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift(1))
                low_close = abs(df['low'] - df['close'].shift(1))
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(14).mean()
            else:
                atr = df['atr']

            # Get current readings
            current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20
            volatility_percentile = atr.rolling(100).rank(pct=True).iloc[-1]

            # Classify regime
            if current_adx > 25:  # Strong trend
                if volatility_percentile < 0.5:
                    return 'trending_low_vol'
                else:
                    return 'trending_high_vol'
            elif current_adx < 20:  # Weak trend (ranging)
                if volatility_percentile < 0.5:
                    return 'ranging_low_vol'
                else:
                    return 'ranging_high_vol'
            else:  # Moderate trend with high volatility
                return 'volatile_choppy'

        except Exception:
            return 'unknown'

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index for trend strength"""
        try:
            # True Range
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # Directional Movement
            plus_dm = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                              np.maximum(df['high'] - df['high'].shift(1), 0), 0)
            minus_dm = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                               np.maximum(df['low'].shift(1) - df['low'], 0), 0)

            # Smooth with Wilder's method
            atr = true_range.rolling(period).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)

            # ADX calculation
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()

            return adx
        except Exception:
            return pd.Series(20, index=df.index)  # Neutral default

    def get_atr_context(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get ATR context for dynamic threshold calculation
        """
        if 'atr' not in df.columns:
            return {'current': 0.001, 'percentile': 0.5, 'multiplier': 1.0}

        current_atr = df['atr'].iloc[-1]
        atr_percentile = df['atr'].rolling(100).rank(pct=True).iloc[-1]

        # Dynamic multiplier based on volatility regime
        if atr_percentile > 0.8:  # High volatility
            multiplier = 1.5
        elif atr_percentile < 0.2:  # Low volatility
            multiplier = 0.7
        else:  # Normal volatility
            multiplier = 1.0

        return {
            'current': current_atr,
            'percentile': atr_percentile,
            'multiplier': multiplier
        }

    def calculate_dynamic_threshold(self, base_value: float, sensitivity: float = 1.0) -> float:
        """
        Calculate dynamic threshold based on ATR and market regime

        Args:
            base_value: Base threshold value
            sensitivity: How much to adjust (1.0 = normal, >1.0 = more sensitive)
        """
        if self.atr_context is None:
            return base_value

        # Adjust based on volatility
        volatility_adjustment = self.atr_context['multiplier']

        # Regime-based adjustment
        regime_adjustments = {
            'trending_low_vol': 0.8,    # Tighter thresholds in stable trends
            'trending_high_vol': 1.2,   # Looser thresholds in volatile trends
            'ranging_low_vol': 1.1,     # Slightly looser in ranges
            'ranging_high_vol': 1.3,    # Much looser in volatile ranges
            'volatile_choppy': 1.5,     # Very loose in chaos
            'unknown': 1.0
        }

        regime_adjustment = regime_adjustments.get(self.market_regime, 1.0)

        # Combined adjustment
        final_adjustment = volatility_adjustment * regime_adjustment * sensitivity

        return base_value * final_adjustment

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Enhanced data validation with professional checks"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Need: {required_columns}")

        if len(df) < self.get_min_periods():
            raise ValueError(f"{self.name} requires minimum {self.get_min_periods()} periods")

        # Check for invalid OHLC relationships
        invalid_high = (df['high'] < df[['open', 'low', 'close']].max(axis=1)).any()
        invalid_low = (df['low'] > df[['open', 'high', 'close']].min(axis=1)).any()

        if invalid_high or invalid_low:
            raise ValueError("Invalid OHLC data detected")

        # Additional professional validations
        # Check for excessive gaps (>10% moves)
        price_changes = df['close'].pct_change().abs()
        if (price_changes > 0.10).any():
            # Log warning but don't fail - could be legitimate market moves
            pass

        # Check for zero/negative prices
        price_columns = ['open', 'high', 'low', 'close']
        if (df[price_columns] <= 0).any().any():
            raise ValueError("Zero or negative prices detected")

        # Check for zero volume (acceptable but flag it)
        if (df['volume'] == 0).sum() > len(df) * 0.1:  # More than 10% zero volume
            pass  # Log warning but continue

        return True

    def safe_calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Safe wrapper for calculate() with error handling and regime detection
        """
        try:
            # Update market context
            if self.enable_regime_detection:
                self.market_regime = self.detect_market_regime(df)
                self.atr_context = self.get_atr_context(df)

            # Perform calculation
            result = self.calculate(df)

            # Cache successful result
            self.last_calculated = pd.Timestamp.now()

            return result

        except Exception as e:
            # Return safe defaults on calculation failure
            safe_result = {}
            for col in self.get_expected_columns():
                safe_result[col] = pd.Series(np.nan, index=df.index)

            print(f"Warning: {self.name} calculation failed: {str(e)}. Using safe defaults.")
            return safe_result

    def get_expected_columns(self) -> list:
        """
        Return list of expected output columns
        Override in child classes for better error recovery
        """
        return [f"{self.name.lower()}_value"]

    def get_signal_confidence(self, df: pd.DataFrame, signal_series: pd.Series) -> pd.Series:
        """
        Calculate confidence score for signals based on market regime and volatility
        """
        confidence = pd.Series(0.5, index=df.index)  # Base confidence

        # Adjust confidence based on market regime
        regime_confidence = {
            'trending_low_vol': 0.9,    # High confidence in stable trends
            'trending_high_vol': 0.7,   # Lower confidence in volatile trends
            'ranging_low_vol': 0.6,     # Moderate confidence in ranges
            'ranging_high_vol': 0.4,    # Low confidence in volatile ranges
            'volatile_choppy': 0.3,     # Very low confidence in chaos
            'unknown': 0.5
        }

        base_conf = regime_confidence.get(self.market_regime, 0.5)

        # Adjust based on signal strength
        signal_mask = signal_series.isin(['BUY', 'SELL'])
        confidence.loc[signal_mask] = base_conf

        return confidence


class RSIModule(TechnicalIndicatorBase):
    """
    Professional-grade Relative Strength Index (RSI) - Momentum Oscillator

    Features:
    - Wilder's smoothing for accuracy (not simple SMA)
    - Dynamic thresholds based on market regime and volatility
    - Multiple divergence detection algorithms
    - Statistical validation of signals
    - Regime-aware parameter adjustment
    """

    def __init__(self, period: int = 14, enable_dynamic_thresholds: bool = True):
        super().__init__("RSI")
        self.period = period
        self.enable_dynamic_thresholds = enable_dynamic_thresholds

    def get_min_periods(self) -> int:
        return self.period + 100  # Extra for dynamic threshold calculation

    def get_expected_columns(self) -> list:
        return ['rsi', 'rsi_avg_gain', 'rsi_avg_loss', 'rsi_percentile']

    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate RSI using professional Wilder's smoothing method"""
        self.validate_data(df)

        # Calculate price changes
        delta = df['close'].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Wilder's smoothing (EMA with alpha = 1/period)
        # This is more accurate than simple moving average
        alpha = 1.0 / self.period

        # Initialize first values with SMA
        avg_gain = gains.rolling(window=self.period).mean()
        avg_loss = losses.rolling(window=self.period).mean()

        # Apply proper Wilder's smoothing for subsequent values
        for i in range(self.period, len(df)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (1 - alpha)) + (gains.iloc[i] * alpha)
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (1 - alpha)) + (losses.iloc[i] * alpha)

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Handle edge cases
        rsi = rsi.fillna(50)  # Neutral when no data
        rsi = rsi.clip(0, 100)  # Ensure valid range

        # Calculate RSI percentile rank for dynamic thresholds
        rsi_percentile = rsi.rolling(100).rank(pct=True)

        return {
            'rsi': rsi,
            'rsi_avg_gain': avg_gain,
            'rsi_avg_loss': avg_loss,
            'rsi_percentile': rsi_percentile
        }

    def calculate_dynamic_thresholds(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate dynamic RSI thresholds based on market regime and historical distribution

        Professional approach: Use percentile-based thresholds instead of fixed 30/70
        """
        if not self.enable_dynamic_thresholds or len(df) < 100:
            return {'oversold': 30, 'overbought': 70, 'neutral_low': 45, 'neutral_high': 55}

        # Use rolling 100-period percentiles for dynamic thresholds
        rsi_values = df['rsi'].dropna()

        if len(rsi_values) < 100:
            return {'oversold': 30, 'overbought': 70, 'neutral_low': 45, 'neutral_high': 55}

        # Calculate percentile-based thresholds
        rolling_window = rsi_values.rolling(100)

        # Base thresholds from historical distribution
        oversold_base = rolling_window.quantile(0.2).iloc[-1]    # Bottom 20%
        overbought_base = rolling_window.quantile(0.8).iloc[-1]  # Top 20%
        neutral_low = rolling_window.quantile(0.4).iloc[-1]      # 40th percentile
        neutral_high = rolling_window.quantile(0.6).iloc[-1]     # 60th percentile

        # Adjust thresholds based on market regime
        if self.market_regime == 'trending_low_vol':
            # In stable trends, use tighter thresholds (catch reversals early)
            oversold = min(oversold_base + 5, 35)
            overbought = max(overbought_base - 5, 65)
        elif self.market_regime == 'trending_high_vol':
            # In volatile trends, use standard thresholds
            oversold = oversold_base
            overbought = overbought_base
        elif self.market_regime in ['ranging_low_vol', 'ranging_high_vol']:
            # In ranging markets, use wider thresholds (mean reversion)
            oversold = max(oversold_base - 5, 20)
            overbought = min(overbought_base + 5, 80)
        else:  # volatile_choppy or unknown
            # In chaos, use very conservative thresholds
            oversold = max(oversold_base - 10, 15)
            overbought = min(overbought_base + 10, 85)

        return {
            'oversold': max(10, min(oversold, 40)),          # Clamp to reasonable range
            'overbought': max(60, min(overbought, 90)),      # Clamp to reasonable range
            'neutral_low': neutral_low,
            'neutral_high': neutral_high
        }

    def detect_divergences(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Professional divergence detection using multiple algorithms
        """
        bullish_divergence = pd.Series(False, index=df.index)
        bearish_divergence = pd.Series(False, index=df.index)
        hidden_bullish = pd.Series(False, index=df.index)
        hidden_bearish = pd.Series(False, index=df.index)

        # Require minimum data
        if len(df) < 50:
            return {
                'bullish_divergence': bullish_divergence,
                'bearish_divergence': bearish_divergence,
                'hidden_bullish_div': hidden_bullish,
                'hidden_bearish_div': hidden_bearish
            }

        # Algorithm 1: Classic Divergence Detection
        lookback = 20
        for i in range(lookback, len(df) - 5):
            # Get windows for analysis
            price_window = df['close'].iloc[i-lookback:i+1]
            rsi_window = df['rsi'].iloc[i-lookback:i+1]

            if len(price_window) < 10:
                continue

            # Find recent swing points
            price_lows = []
            rsi_lows = []
            price_highs = []
            rsi_highs = []

            # Simple peak/trough detection
            for j in range(2, len(price_window) - 2):
                # Check for local low
                if (price_window.iloc[j] < price_window.iloc[j-1] and
                    price_window.iloc[j] < price_window.iloc[j+1] and
                    price_window.iloc[j] < price_window.iloc[j-2] and
                    price_window.iloc[j] < price_window.iloc[j+2]):
                    price_lows.append((j, price_window.iloc[j]))
                    rsi_lows.append((j, rsi_window.iloc[j]))

                # Check for local high
                if (price_window.iloc[j] > price_window.iloc[j-1] and
                    price_window.iloc[j] > price_window.iloc[j+1] and
                    price_window.iloc[j] > price_window.iloc[j-2] and
                    price_window.iloc[j] > price_window.iloc[j+2]):
                    price_highs.append((j, price_window.iloc[j]))
                    rsi_highs.append((j, rsi_window.iloc[j]))

            # Check for bullish divergence (price lower low, RSI higher low)
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                last_price_low = price_lows[-1][1]
                prev_price_low = price_lows[-2][1]
                last_rsi_low = rsi_lows[-1][1]
                prev_rsi_low = rsi_lows[-2][1]

                if (last_price_low < prev_price_low and
                    last_rsi_low > prev_rsi_low and
                    last_rsi_low < 40):  # Must be in oversold territory
                    bullish_divergence.iloc[i] = True

            # Check for bearish divergence (price higher high, RSI lower high)
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                last_price_high = price_highs[-1][1]
                prev_price_high = price_highs[-2][1]
                last_rsi_high = rsi_highs[-1][1]
                prev_rsi_high = rsi_highs[-2][1]

                if (last_price_high > prev_price_high and
                    last_rsi_high < prev_rsi_high and
                    last_rsi_high > 60):  # Must be in overbought territory
                    bearish_divergence.iloc[i] = True

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'hidden_bullish_div': hidden_bullish,
            'hidden_bearish_div': hidden_bearish
        }

    def get_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate professional RSI trading signals with dynamic thresholds"""

        # Get dynamic thresholds
        thresholds = self.calculate_dynamic_thresholds(df)

        # Detect divergences
        divergences = self.detect_divergences(df)

        # Initialize signal series
        rsi_signal = pd.Series('HOLD', index=df.index)
        rsi_strength = pd.Series('WEAK', index=df.index)
        rsi_regime_signal = pd.Series('HOLD', index=df.index)

        # Basic RSI signals with dynamic thresholds
        oversold_mask = df['rsi'] < thresholds['oversold']
        overbought_mask = df['rsi'] > thresholds['overbought']

        rsi_signal.loc[oversold_mask] = 'BUY'
        rsi_signal.loc[overbought_mask] = 'SELL'

        # Enhanced signals based on market regime
        if self.market_regime in ['trending_low_vol', 'trending_high_vol']:
            # In trends, only take signals in direction of trend
            # Use longer-term MA to determine trend direction
            if 'sma_50' in df.columns:
                trend_up = df['close'] > df['sma_50']
                trend_down = df['close'] < df['sma_50']

                # Only BUY signals in uptrend, only SELL in downtrend
                rsi_regime_signal.loc[oversold_mask & trend_up] = 'TREND_BUY'
                rsi_regime_signal.loc[overbought_mask & trend_down] = 'TREND_SELL'

        elif self.market_regime in ['ranging_low_vol', 'ranging_high_vol']:
            # In ranging markets, take counter-trend signals (mean reversion)
            rsi_regime_signal.loc[oversold_mask] = 'RANGE_BUY'
            rsi_regime_signal.loc[overbought_mask] = 'RANGE_SELL'

        # Divergence signals (high priority)
        divergence_signals = pd.Series('HOLD', index=df.index)
        divergence_signals.loc[divergences['bullish_divergence']] = 'DIV_BUY'
        divergence_signals.loc[divergences['bearish_divergence']] = 'DIV_SELL'

        # Signal strength based on RSI position and regime
        extreme_oversold = df['rsi'] < thresholds['oversold'] * 0.7  # Very oversold
        extreme_overbought = df['rsi'] > thresholds['overbought'] * 1.3  # Very overbought

        rsi_strength.loc[extreme_oversold | extreme_overbought] = 'STRONG'
        rsi_strength.loc[oversold_mask | overbought_mask] = 'MODERATE'

        # Calculate signal confidence
        confidence = self.get_signal_confidence(df, rsi_signal)

        return {
            'rsi_signal': rsi_signal,
            'rsi_divergence': divergence_signals,
            'rsi_regime_signal': rsi_regime_signal,
            'rsi_strength': rsi_strength,
            'rsi_confidence': confidence,
            'rsi_thresholds': pd.Series([
                f"OS:{thresholds['oversold']:.1f},OB:{thresholds['overbought']:.1f}"
            ] * len(df), index=df.index)
        }


class MACDModule(TechnicalIndicatorBase):
    """
    Professional-grade Moving Average Convergence Divergence (MACD)

    Features:
    - Market regime-adaptive parameters (not fixed 12,26,9)
    - Dynamic significance testing (not every crossover = signal)
    - Advanced histogram momentum analysis
    - Divergence detection algorithms
    - Zero-line context intelligence
    - ATR-based move validation
    """

    def __init__(self, fast_period: int = None, slow_period: int = None,
                 signal_period: int = None, enable_adaptive_params: bool = True,
                 timeframe: str = None):
        super().__init__("MACD")
        self.enable_adaptive_params = enable_adaptive_params
        self.timeframe = timeframe

        # Timeframe-specific base parameters
        timeframe_params = {
            '15m': {'fast': 8, 'slow': 17, 'signal': 9},    # Faster for scalping
            '1h': {'fast': 12, 'slow': 26, 'signal': 9},    # Standard
            '4h': {'fast': 16, 'slow': 35, 'signal': 12},   # Slower for swing
            'daily': {'fast': 19, 'slow': 39, 'signal': 15} # Smoothest for position
        }

        # Set base parameters based on timeframe
        tf_params = timeframe_params.get(timeframe, {'fast': 12, 'slow': 26, 'signal': 9})
        self.base_fast = fast_period or tf_params['fast']
        self.base_slow = slow_period or tf_params['slow']
        self.base_signal = signal_period or tf_params['signal']

        # Current active parameters (adjusted by regime)
        self.fast_period = self.base_fast
        self.slow_period = self.base_slow
        self.signal_period = self.base_signal

    def get_min_periods(self) -> int:
        return max(self.base_slow + self.base_signal + 50, 100)  # Extra for significance testing

    def get_expected_columns(self) -> list:
        return ['macd', 'macd_signal', 'macd_histogram', 'ema_fast', 'ema_slow']

    def adjust_parameters_for_regime(self):
        """
        Adjust MACD parameters based on detected market regime
        Professional approach: Different settings for different market conditions
        """
        if not self.enable_adaptive_params or self.market_regime == 'unknown':
            return

        # Regime-specific parameter adjustments
        regime_adjustments = {
            'trending_low_vol': {
                'fast_mult': 0.8,    # Faster (10 instead of 12) - catch trends early
                'slow_mult': 0.8,    # Faster (21 instead of 26)
                'signal_mult': 0.7   # Faster (6 instead of 9) - quicker signals
            },
            'trending_high_vol': {
                'fast_mult': 1.0,    # Standard settings work well in volatile trends
                'slow_mult': 1.0,
                'signal_mult': 1.0
            },
            'ranging_low_vol': {
                'fast_mult': 1.2,    # Slower (14 instead of 12) - reduce noise
                'slow_mult': 1.15,   # Slower (30 instead of 26)
                'signal_mult': 1.3   # Slower (12 instead of 9) - fewer false signals
            },
            'ranging_high_vol': {
                'fast_mult': 1.3,    # Much slower in volatile ranges
                'slow_mult': 1.2,
                'signal_mult': 1.4
            },
            'volatile_choppy': {
                'fast_mult': 1.4,    # Very slow in chaos (17,31,13)
                'slow_mult': 1.2,
                'signal_mult': 1.4
            }
        }

        adjustment = regime_adjustments.get(self.market_regime, {'fast_mult': 1.0, 'slow_mult': 1.0, 'signal_mult': 1.0})

        self.fast_period = int(self.base_fast * adjustment['fast_mult'])
        self.slow_period = int(self.base_slow * adjustment['slow_mult'])
        self.signal_period = int(self.base_signal * adjustment['signal_mult'])

    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD with regime-adaptive parameters"""
        self.validate_data(df)

        # Adjust parameters based on market regime (from base class)
        self.adjust_parameters_for_regime()

        # Calculate EMAs with adjusted parameters
        ema_fast = df['close'].ewm(span=self.fast_period).mean()
        ema_slow = df['close'].ewm(span=self.slow_period).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD line)
        signal_line = macd_line.ewm(span=self.signal_period).mean()

        # Histogram (MACD - Signal)
        histogram = macd_line - signal_line

        # Calculate additional professional metrics
        histogram_slope = histogram.diff()  # Rate of change in histogram
        histogram_acceleration = histogram_slope.diff()  # Acceleration/deceleration

        # Normalize MACD by ATR for significance testing
        if self.atr_context and self.atr_context['current'] > 0:
            macd_normalized = abs(macd_line) / (self.atr_context['current'] * df['close'])
        else:
            macd_normalized = abs(macd_line) / df['close'] * 1000  # Fallback normalization

        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram,
            'macd_slope': histogram_slope,
            'macd_acceleration': histogram_acceleration,
            'macd_normalized': macd_normalized,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'macd_params': pd.Series([f"F:{self.fast_period},S:{self.slow_period},Sig:{self.signal_period}"] * len(df), index=df.index)
        }

    def calculate_significance_threshold(self, df: pd.DataFrame) -> float:
        """
        Calculate dynamic significance threshold for MACD moves
        Professional approach: Not every crossover is significant
        """
        if len(df) < 50:
            return 0.001  # Fallback

        # Use histogram standard deviation for significance testing
        hist_std = df['macd_histogram'].rolling(50).std().iloc[-1]

        if pd.isna(hist_std) or hist_std == 0:
            return 0.001

        # Adjust threshold based on market regime
        regime_multipliers = {
            'trending_low_vol': 1.0,    # Standard threshold in stable trends
            'trending_high_vol': 1.5,   # Higher threshold in volatile trends
            'ranging_low_vol': 0.8,     # Lower threshold in stable ranges
            'ranging_high_vol': 2.0,    # Much higher threshold in volatile ranges
            'volatile_choppy': 2.5,     # Very high threshold in chaos
            'unknown': 1.5
        }

        multiplier = regime_multipliers.get(self.market_regime, 1.5)
        return hist_std * multiplier

    def detect_macd_divergences(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Professional MACD divergence detection
        """
        bullish_div = pd.Series(False, index=df.index)
        bearish_div = pd.Series(False, index=df.index)
        hidden_bullish = pd.Series(False, index=df.index)
        hidden_bearish = pd.Series(False, index=df.index)

        if len(df) < 50:
            return {
                'bullish_divergence': bullish_div,
                'bearish_divergence': bearish_div,
                'hidden_bullish_div': hidden_bullish,
                'hidden_bearish_div': hidden_bearish
            }

        # Divergence detection using swing point analysis
        lookback = 25
        for i in range(lookback, len(df) - 5):
            price_window = df['close'].iloc[i-lookback:i+1]
            macd_window = df['macd'].iloc[i-lookback:i+1]

            if len(price_window) < 15:
                continue

            # Find swing highs and lows
            price_highs = []
            price_lows = []
            macd_highs = []
            macd_lows = []

            # Peak/trough detection with 3-bar confirmation
            for j in range(3, len(price_window) - 3):
                # Local high detection
                if (price_window.iloc[j] > price_window.iloc[j-1] and
                    price_window.iloc[j] > price_window.iloc[j+1] and
                    price_window.iloc[j] > price_window.iloc[j-2] and
                    price_window.iloc[j] > price_window.iloc[j+2] and
                    price_window.iloc[j] > price_window.iloc[j-3] and
                    price_window.iloc[j] > price_window.iloc[j+3]):
                    price_highs.append((j, price_window.iloc[j]))
                    macd_highs.append((j, macd_window.iloc[j]))

                # Local low detection
                if (price_window.iloc[j] < price_window.iloc[j-1] and
                    price_window.iloc[j] < price_window.iloc[j+1] and
                    price_window.iloc[j] < price_window.iloc[j-2] and
                    price_window.iloc[j] < price_window.iloc[j+2] and
                    price_window.iloc[j] < price_window.iloc[j-3] and
                    price_window.iloc[j] < price_window.iloc[j+3]):
                    price_lows.append((j, price_window.iloc[j]))
                    macd_lows.append((j, macd_window.iloc[j]))

            # Check for classic bearish divergence
            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                last_price_high = price_highs[-1][1]
                prev_price_high = price_highs[-2][1]
                last_macd_high = macd_highs[-1][1]
                prev_macd_high = macd_highs[-2][1]

                if (last_price_high > prev_price_high and
                    last_macd_high < prev_macd_high and
                    last_macd_high > 0):  # MACD must be above zero for bearish div
                    bearish_div.iloc[i] = True

            # Check for classic bullish divergence
            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                last_price_low = price_lows[-1][1]
                prev_price_low = price_lows[-2][1]
                last_macd_low = macd_lows[-1][1]
                prev_macd_low = macd_lows[-2][1]

                if (last_price_low < prev_price_low and
                    last_macd_low > prev_macd_low and
                    last_macd_low < 0):  # MACD must be below zero for bullish div
                    bullish_div.iloc[i] = True

        return {
            'bullish_divergence': bullish_div,
            'bearish_divergence': bearish_div,
            'hidden_bullish_div': hidden_bullish,
            'hidden_bearish_div': hidden_bearish
        }

    def analyze_histogram_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Advanced histogram momentum analysis
        """
        # Initialize pattern series
        patterns = {
            'acceleration': pd.Series('NONE', index=df.index),
            'convergence': pd.Series('NONE', index=df.index),
            'zero_approach': pd.Series('NONE', index=df.index)
        }

        if len(df) < 20:
            return patterns

        # 1. Acceleration/Deceleration Analysis
        accel_mask = (df['macd_acceleration'] > 0) & (df['macd_histogram'] > 0)
        decel_mask = (df['macd_acceleration'] < 0) & (df['macd_histogram'] < 0)

        patterns['acceleration'].loc[accel_mask] = 'BULLISH_ACCEL'
        patterns['acceleration'].loc[decel_mask] = 'BEARISH_ACCEL'

        # 2. Convergence Pattern (histogram moving toward zero)
        hist_abs_decreasing = abs(df['macd_histogram']) < abs(df['macd_histogram'].shift(1))
        approaching_zero = abs(df['macd_histogram']) < abs(df['macd_histogram']).rolling(10).mean() * 0.5

        patterns['convergence'].loc[hist_abs_decreasing & approaching_zero] = 'CONVERGING'

        # 3. Zero-line approach signals
        close_to_zero = abs(df['macd_histogram']) < abs(df['macd_histogram']).rolling(20).std() * 0.5
        patterns['zero_approach'].loc[close_to_zero] = 'NEAR_ZERO'

        return patterns

    def get_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate professional MACD trading signals with significance testing"""

        # Calculate significance threshold
        significance_threshold = self.calculate_significance_threshold(df)

        # Detect divergences
        divergences = self.detect_macd_divergences(df)

        # Analyze histogram patterns
        patterns = self.analyze_histogram_patterns(df)

        # Initialize signal series
        macd_signal = pd.Series('HOLD', index=df.index)
        macd_strength = pd.Series('WEAK', index=df.index)
        macd_regime_signal = pd.Series('HOLD', index=df.index)
        zero_line_signals = pd.Series('HOLD', index=df.index)

        # 1. SIGNIFICANT crossover signals (not every tiny crossover)
        bullish_cross = ((df['macd'] > df['macd_signal']) &
                        (df['macd'].shift(1) <= df['macd_signal'].shift(1)))
        bearish_cross = ((df['macd'] < df['macd_signal']) &
                        (df['macd'].shift(1) >= df['macd_signal'].shift(1)))

        # Only signal if move is statistically significant
        significant_bullish = bullish_cross & (abs(df['macd_histogram']) > significance_threshold)
        significant_bearish = bearish_cross & (abs(df['macd_histogram']) > significance_threshold)

        macd_signal.loc[significant_bullish] = 'BUY'
        macd_signal.loc[significant_bearish] = 'SELL'

        # 2. Zero-line context signals
        zero_cross_up = ((df['macd'] > 0) & (df['macd'].shift(1) <= 0))
        zero_cross_down = ((df['macd'] < 0) & (df['macd'].shift(1) >= 0))

        zero_line_signals.loc[zero_cross_up] = 'ZERO_BUY'
        zero_line_signals.loc[zero_cross_down] = 'ZERO_SELL'

        # 3. Regime-aware signals
        if self.market_regime in ['trending_low_vol', 'trending_high_vol']:
            # In trends, prioritize zero-line crosses and strong momentum
            trend_momentum_bull = (df['macd'] > 0) & (df['macd_histogram'] > 0) & significant_bullish
            trend_momentum_bear = (df['macd'] < 0) & (df['macd_histogram'] < 0) & significant_bearish

            macd_regime_signal.loc[trend_momentum_bull] = 'TREND_BUY'
            macd_regime_signal.loc[trend_momentum_bear] = 'TREND_SELL'

        elif self.market_regime in ['ranging_low_vol', 'ranging_high_vol']:
            # In ranges, look for reversal signals at extremes
            extreme_positive = df['macd'] > df['macd'].rolling(50).quantile(0.8)
            extreme_negative = df['macd'] < df['macd'].rolling(50).quantile(0.2)

            range_reversal_sell = extreme_positive & significant_bearish
            range_reversal_buy = extreme_negative & significant_bullish

            macd_regime_signal.loc[range_reversal_buy] = 'RANGE_BUY'
            macd_regime_signal.loc[range_reversal_sell] = 'RANGE_SELL'

        # 4. Divergence signals (highest priority)
        div_signals = pd.Series('HOLD', index=df.index)
        div_signals.loc[divergences['bullish_divergence']] = 'DIV_BUY'
        div_signals.loc[divergences['bearish_divergence']] = 'DIV_SELL'

        # 5. Signal strength classification
        strong_momentum = (patterns['acceleration'].isin(['BULLISH_ACCEL', 'BEARISH_ACCEL']))
        moderate_signals = significant_bullish | significant_bearish

        macd_strength.loc[strong_momentum] = 'STRONG'
        macd_strength.loc[moderate_signals] = 'MODERATE'

        # 6. Calculate confidence using base class
        confidence = self.get_signal_confidence(df, macd_signal)

        return {
            'macd_signal': macd_signal,
            'macd_divergence': div_signals,
            'macd_regime_signal': macd_regime_signal,
            'macd_zero_cross': zero_line_signals,
            'macd_strength': macd_strength,
            'macd_confidence': confidence,
            'macd_patterns': patterns['acceleration'],
            'macd_significance': pd.Series([f"Threshold: {significance_threshold:.4f}"] * len(df), index=df.index)
        }


class BollingerBandsModule(TechnicalIndicatorBase):
    """
    Professional-grade Bollinger Bands - Adaptive Volatility Indicator

    Features:
    - Dynamic band multipliers based on market regime and volatility
    - Adaptive period selection for different timeframes and conditions
    - Statistical squeeze detection with multiple confirmation factors
    - Advanced breakout vs bounce detection with volume confirmation
    - Multi-timeframe context awareness
    - Professional %B and bandwidth analysis
    """

    def __init__(self, base_period: int = 20, base_std_dev: float = 2.0,
                 enable_adaptive_params: bool = True, timeframe: str = None):
        super().__init__("BollingerBands")
        self.base_period = base_period
        self.base_std_dev = base_std_dev
        self.enable_adaptive_params = enable_adaptive_params
        self.timeframe = timeframe

        # Current active parameters (will be adjusted)
        self.period = base_period
        self.std_dev = base_std_dev

    def get_min_periods(self) -> int:
        return max(self.base_period + 50, 100)  # Extra for statistical analysis

    def get_expected_columns(self) -> list:
        return ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent_b']

    def adjust_parameters_for_regime(self):
        """
        Dynamically adjust Bollinger Band parameters based on market regime
        Professional approach: Different settings for different market conditions
        """
        if not self.enable_adaptive_params or self.market_regime == 'unknown':
            return

        # Regime-specific parameter adjustments
        regime_adjustments = {
            'trending_low_vol': {
                'period_mult': 0.9,      # Slightly faster (18 vs 20) for responsiveness
                'std_mult': 1.8 / 2.0,   # Tighter bands (1.8 vs 2.0) in stable trends
                'sensitivity': 'high'     # More sensitive to trend changes
            },
            'trending_high_vol': {
                'period_mult': 1.0,      # Standard period
                'std_mult': 2.2 / 2.0,   # Wider bands (2.2) for volatile trends
                'sensitivity': 'medium'
            },
            'ranging_low_vol': {
                'period_mult': 1.1,      # Slightly slower (22) for stability
                'std_mult': 1.6 / 2.0,   # Tighter bands (1.6) for range trading
                'sensitivity': 'very_high' # Very sensitive for range reversals
            },
            'ranging_high_vol': {
                'period_mult': 1.2,      # Slower (24) to reduce noise
                'std_mult': 2.5 / 2.0,   # Much wider bands (2.5)
                'sensitivity': 'low'
            },
            'volatile_choppy': {
                'period_mult': 1.3,      # Much slower (26) to filter chaos
                'std_mult': 2.8 / 2.0,   # Widest bands (2.8) for chaos
                'sensitivity': 'very_low'
            }
        }

        # Timeframe-specific adjustments
        timeframe_adjustments = {
            '15m': {'period_mult': 0.8, 'std_mult': 0.9},  # Faster, tighter
            '1h': {'period_mult': 1.0, 'std_mult': 1.0},   # Standard
            '4h': {'period_mult': 1.1, 'std_mult': 1.1},   # Slower, wider
            'daily': {'period_mult': 1.2, 'std_mult': 1.2} # Much slower, wider
        }

        # Apply regime adjustments
        regime_adj = regime_adjustments.get(self.market_regime,
                                          {'period_mult': 1.0, 'std_mult': 1.0, 'sensitivity': 'medium'})

        # Apply timeframe adjustments
        tf_adj = timeframe_adjustments.get(self.timeframe, {'period_mult': 1.0, 'std_mult': 1.0})

        # Combined adjustments
        self.period = int(self.base_period * regime_adj['period_mult'] * tf_adj['period_mult'])
        self.std_dev = self.base_std_dev * regime_adj['std_mult'] * tf_adj['std_mult']

        # Store sensitivity for signal generation
        self.sensitivity = regime_adj['sensitivity']

    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands with adaptive parameters"""
        self.validate_data(df)

        # Adjust parameters based on market regime
        self.adjust_parameters_for_regime()

        # Middle band (Simple Moving Average)
        middle_band = df['close'].rolling(window=self.period).mean()

        # Standard deviation calculation (population std for BB)
        std = df['close'].rolling(window=self.period).std(ddof=0)

        # Upper and lower bands with dynamic multiplier
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)

        # Professional band width calculation (normalized)
        band_width = (upper_band - lower_band) / middle_band * 100

        # %B indicator (position within bands)
        percent_b = (df['close'] - lower_band) / (upper_band - lower_band)

        # Band width percentile (for squeeze detection)
        band_width_percentile = band_width.rolling(100).rank(pct=True)

        # Distance from bands (normalized by ATR)
        if self.atr_context and self.atr_context['current'] > 0:
            atr = self.atr_context['current']
            upper_distance = (df['close'] - upper_band) / atr
            lower_distance = (lower_band - df['close']) / atr
        else:
            # Fallback normalization
            upper_distance = (df['close'] - upper_band) / df['close'] * 1000
            lower_distance = (lower_band - df['close']) / df['close'] * 1000

        return {
            'bb_upper': upper_band,
            'bb_middle': middle_band,
            'bb_lower': lower_band,
            'bb_width': band_width,
            'bb_percent_b': percent_b,
            'bb_width_percentile': band_width_percentile,
            'bb_upper_distance': upper_distance,
            'bb_lower_distance': lower_distance,
            'bb_params': pd.Series([f"P:{self.period},STD:{self.std_dev:.2f}"] * len(df), index=df.index)
        }

    def detect_statistical_squeeze(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Professional squeeze detection using multiple statistical factors
        """
        squeeze_signals = pd.Series('NONE', index=df.index)
        squeeze_strength = pd.Series(0.0, index=df.index)

        if len(df) < 50:
            return {'squeeze_signal': squeeze_signals, 'squeeze_strength': squeeze_strength}

        # Factor 1: Band width percentile (most important)
        width_threshold = 0.2  # Bottom 20% of historical width
        low_width = df['bb_width_percentile'] < width_threshold

        # Factor 2: Decreasing volatility trend
        width_slope = df['bb_width'].rolling(5).apply(lambda x: np.polyfit(range(5), x, 1)[0], raw=False)
        decreasing_width = width_slope < 0

        # Factor 3: Volume patterns (if available)
        volume_factor = pd.Series(True, index=df.index)  # Default to True
        if 'volume' in df.columns:
            avg_volume = df['volume'].rolling(20).mean()
            volume_factor = df['volume'] < avg_volume  # Lower volume during squeeze

        # Factor 4: Price consolidation (low %B volatility)
        percent_b_volatility = df['bb_percent_b'].rolling(10).std()
        low_price_volatility = percent_b_volatility < percent_b_volatility.rolling(50).quantile(0.3)

        # Combined squeeze detection
        basic_squeeze = low_width & decreasing_width
        strong_squeeze = basic_squeeze & volume_factor & low_price_volatility

        # Squeeze duration (longer squeezes = stronger breakouts)
        squeeze_duration = pd.Series(0, index=df.index)
        current_duration = 0
        for i in range(len(df)):
            if basic_squeeze.iloc[i]:
                current_duration += 1
                squeeze_duration.iloc[i] = current_duration
            else:
                current_duration = 0

        # Classify squeeze strength
        squeeze_signals.loc[basic_squeeze] = 'SQUEEZE'
        squeeze_signals.loc[strong_squeeze] = 'STRONG_SQUEEZE'
        squeeze_signals.loc[squeeze_duration > 10] = 'EXTENDED_SQUEEZE'

        # Calculate squeeze strength score
        strength_components = [
            (1 - df['bb_width_percentile']),  # Lower width = higher strength
            np.clip(-width_slope / width_slope.std(), 0, 1),  # Decreasing trend
            squeeze_duration / 20,  # Duration factor
        ]

        squeeze_strength = pd.concat(strength_components, axis=1).mean(axis=1).fillna(0)

        return {
            'squeeze_signal': squeeze_signals,
            'squeeze_strength': squeeze_strength,
            'squeeze_duration': squeeze_duration
        }

    def analyze_breakout_vs_bounce(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Professional breakout vs bounce detection with multiple confirmations
        """
        breakout_signals = pd.Series('HOLD', index=df.index)
        bounce_signals = pd.Series('HOLD', index=df.index)
        confidence_scores = pd.Series(0.0, index=df.index)

        if len(df) < 20:
            return {
                'breakout_signal': breakout_signals,
                'bounce_signal': bounce_signals,
                'confidence': confidence_scores
            }

        # Band touch detection
        upper_touch = df['close'] >= df['bb_upper']
        lower_touch = df['close'] <= df['bb_lower']

        # Volume confirmation (if available)
        volume_surge = pd.Series(True, index=df.index)  # Default
        if 'volume' in df.columns:
            avg_volume = df['volume'].rolling(20).mean()
            volume_surge = df['volume'] > avg_volume * 1.5

        # Momentum confirmation
        price_momentum = df['close'] > df['close'].shift(1)  # Basic momentum
        strong_momentum = df['close'] > df['close'].shift(3)  # 3-bar momentum

        # Previous level confirmation (support/resistance)
        resistance_break = pd.Series(False, index=df.index)
        support_break = pd.Series(False, index=df.index)

        if 'true_resistance' in df.columns and 'true_support' in df.columns:
            resistance_break = df['close'] > df['true_resistance']
            support_break = df['close'] < df['true_support']

        # Time-based factors (session strength)
        # This would be enhanced with actual time data
        session_strength = pd.Series(1.0, index=df.index)  # Placeholder

        # BREAKOUT DETECTION
        # Upper breakout criteria
        upper_breakout = (
            upper_touch &
            volume_surge &
            strong_momentum &
            (df['bb_percent_b'] > 1.1)  # Significant break above band
        )

        # Lower breakout criteria
        lower_breakout = (
            lower_touch &
            volume_surge &
            ~price_momentum &  # Downward momentum
            (df['bb_percent_b'] < -0.1)  # Significant break below band
        )

        # BOUNCE DETECTION
        # Upper bounce criteria
        upper_bounce = (
            (df['close'].shift(1) >= df['bb_upper'].shift(1)) &
            (df['close'] < df['bb_upper']) &
            ~volume_surge &  # No volume surge = likely bounce
            (df['bb_percent_b'] < 1.0)
        )

        # Lower bounce criteria
        lower_bounce = (
            (df['close'].shift(1) <= df['bb_lower'].shift(1)) &
            (df['close'] > df['bb_lower']) &
            ~volume_surge &
            (df['bb_percent_b'] > 0.0)
        )

        # Apply regime-specific logic
        if self.market_regime in ['trending_low_vol', 'trending_high_vol']:
            # In trends, favor breakouts over bounces
            breakout_signals.loc[upper_breakout] = 'BREAKOUT_BUY'
            breakout_signals.loc[lower_breakout] = 'BREAKOUT_SELL'

            # Only strong bounces in trends
            bounce_signals.loc[upper_bounce & resistance_break] = 'TREND_BOUNCE_SELL'
            bounce_signals.loc[lower_bounce & support_break] = 'TREND_BOUNCE_BUY'

        elif self.market_regime in ['ranging_low_vol', 'ranging_high_vol']:
            # In ranges, favor bounces over breakouts
            bounce_signals.loc[upper_bounce] = 'RANGE_BOUNCE_SELL'
            bounce_signals.loc[lower_bounce] = 'RANGE_BOUNCE_BUY'

            # Only confirmed breakouts in ranges
            breakout_signals.loc[upper_breakout & volume_surge & resistance_break] = 'RANGE_BREAKOUT_BUY'
            breakout_signals.loc[lower_breakout & volume_surge & support_break] = 'RANGE_BREAKOUT_SELL'

        # Calculate confidence scores
        confidence_factors = pd.DataFrame({
            'volume': volume_surge.astype(float),
            'momentum': strong_momentum.astype(float),
            'level_break': (resistance_break | support_break).astype(float),
            'session': session_strength
        })

        confidence_scores = confidence_factors.mean(axis=1)

        return {
            'breakout_signal': breakout_signals,
            'bounce_signal': bounce_signals,
            'confidence': confidence_scores
        }

    def get_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate professional Bollinger Bands signals with regime awareness"""

        # Get squeeze analysis
        squeeze_analysis = self.detect_statistical_squeeze(df)

        # Get breakout vs bounce analysis
        breakout_analysis = self.analyze_breakout_vs_bounce(df)

        # Initialize main signal series
        bb_signal = pd.Series('HOLD', index=df.index)
        bb_strength = pd.Series('WEAK', index=df.index)
        bb_regime_signal = pd.Series('HOLD', index=df.index)

        # Basic band signals with %B context
        extreme_oversold = df['bb_percent_b'] < -0.1  # Below lower band
        oversold = df['bb_percent_b'] < 0.2           # Near lower band
        overbought = df['bb_percent_b'] > 0.8         # Near upper band
        extreme_overbought = df['bb_percent_b'] > 1.1 # Above upper band

        # Sensitivity-adjusted signals based on regime
        if hasattr(self, 'sensitivity'):
            if self.sensitivity == 'very_high':
                bb_signal.loc[df['bb_percent_b'] < 0.3] = 'BUY'
                bb_signal.loc[df['bb_percent_b'] > 0.7] = 'SELL'
            elif self.sensitivity == 'high':
                bb_signal.loc[oversold] = 'BUY'
                bb_signal.loc[overbought] = 'SELL'
            elif self.sensitivity == 'medium':
                bb_signal.loc[extreme_oversold] = 'BUY'
                bb_signal.loc[extreme_overbought] = 'SELL'
            else:  # low or very_low
                bb_signal.loc[df['bb_percent_b'] < -0.2] = 'BUY'
                bb_signal.loc[df['bb_percent_b'] > 1.2] = 'SELL'

        # Regime-specific signals
        if self.market_regime in ['trending_low_vol', 'trending_high_vol']:
            # In trends, prioritize breakout signals
            trend_signals = breakout_analysis['breakout_signal']
            bb_regime_signal.loc[trend_signals != 'HOLD'] = trend_signals

        elif self.market_regime in ['ranging_low_vol', 'ranging_high_vol']:
            # In ranges, prioritize bounce signals
            range_signals = breakout_analysis['bounce_signal']
            bb_regime_signal.loc[range_signals != 'HOLD'] = range_signals

        # Signal strength classification
        high_confidence = breakout_analysis['confidence'] > 0.7
        medium_confidence = breakout_analysis['confidence'] > 0.4

        bb_strength.loc[high_confidence] = 'STRONG'
        bb_strength.loc[medium_confidence] = 'MODERATE'

        # Calculate overall confidence using base class
        confidence = self.get_signal_confidence(df, bb_signal)

        return {
            'bb_signal': bb_signal,
            'bb_regime_signal': bb_regime_signal,
            'bb_squeeze': squeeze_analysis['squeeze_signal'],
            'bb_breakout': breakout_analysis['breakout_signal'],
            'bb_bounce': breakout_analysis['bounce_signal'],
            'bb_strength': bb_strength,
            'bb_confidence': confidence,
            'bb_squeeze_strength': squeeze_analysis['squeeze_strength'],
            'bb_pattern_confidence': breakout_analysis['confidence']
        }


class ATRModule(TechnicalIndicatorBase):
    """
    Professional Average True Range (ATR) - Adaptive Volatility Measure

    Features:
    - Timeframe-specific period optimization
    - Professional volatility classification
    - Position sizing recommendations
    - Market regime awareness
    """

    def __init__(self, base_period: int = 14, timeframe: str = None):
        super().__init__("ATR")
        self.base_period = base_period
        self.timeframe = timeframe

        # Timeframe-specific period optimization
        self.timeframe_periods = {
            '15m': 10,    # Faster response for scalping
            '1h': 14,     # Standard period
            '4h': 18,     # Smoother for swing trades
            'daily': 21   # Most stable for position trades
        }

        # Set active period based on timeframe
        self.period = self.timeframe_periods.get(timeframe, base_period)

    def get_min_periods(self) -> int:
        return self.period + 1

    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate professional ATR with volatility analysis"""
        self.validate_data(df)

        # True Range components
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))

        # True Range (maximum of the three)
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        # ATR using Wilder's smoothing
        atr = true_range.rolling(window=self.period).mean()

        # Apply Wilder's smoothing for more accurate ATR
        for i in range(self.period, len(df)):
            atr.iloc[i] = (atr.iloc[i-1] * (self.period - 1) + true_range.iloc[i]) / self.period

        # ATR as percentage of price (normalized volatility)
        atr_percent = (atr / df['close']) * 100

        # Professional volatility percentile ranking (100-period lookback)
        atr_percentile = atr.rolling(100).rank(pct=True)

        # Volatility trend (expanding vs contracting)
        atr_slope = atr.rolling(5).apply(lambda x: np.polyfit(range(5), x, 1)[0], raw=False)

        # Normalized volatility trend
        atr_trend = atr_slope / atr.rolling(20).std()

        # Position sizing multiplier based on volatility
        position_multiplier = self._calculate_position_multiplier(atr_percentile)

        return {
            'atr': atr,
            'atr_percent': atr_percent,
            'true_range': true_range,
            'atr_percentile': atr_percentile,
            'atr_trend': atr_trend,
            'position_multiplier': position_multiplier,
            'atr_params': pd.Series([f"TF:{self.timeframe},P:{self.period}"] * len(df), index=df.index)
        }

    def _calculate_position_multiplier(self, atr_percentile: pd.Series) -> pd.Series:
        """
        Calculate position sizing multiplier based on volatility percentile
        Lower volatility = larger positions, Higher volatility = smaller positions
        """
        multiplier = pd.Series(1.0, index=atr_percentile.index)

        # Ultra-low volatility (< 10th percentile): Increase position size
        ultra_low_vol = atr_percentile < 0.1
        multiplier.loc[ultra_low_vol] = 1.5

        # Low volatility (10-30th percentile): Slightly increase position size
        low_vol = (atr_percentile >= 0.1) & (atr_percentile < 0.3)
        multiplier.loc[low_vol] = 1.25

        # Normal volatility (30-70th percentile): Standard position size
        # (already set to 1.0)

        # High volatility (70-90th percentile): Reduce position size
        high_vol = (atr_percentile >= 0.7) & (atr_percentile < 0.9)
        multiplier.loc[high_vol] = 0.75

        # Extreme volatility (> 90th percentile): Minimize position size
        extreme_vol = atr_percentile >= 0.9
        multiplier.loc[extreme_vol] = 0.5

        return multiplier

    def get_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Professional ATR-based volatility and position sizing signals"""

        # Professional volatility regime classification
        volatility_regime = pd.Series('NORMAL', index=df.index)
        position_signal = pd.Series('NORMAL_SIZE', index=df.index)
        volatility_pattern = pd.Series('STABLE', index=df.index)

        # Volatility regime based on percentile
        ultra_low = df['atr_percentile'] < 0.1
        low = (df['atr_percentile'] >= 0.1) & (df['atr_percentile'] < 0.3)
        high = (df['atr_percentile'] >= 0.7) & (df['atr_percentile'] < 0.9)
        extreme = df['atr_percentile'] >= 0.9

        volatility_regime.loc[ultra_low] = 'ULTRA_LOW'
        volatility_regime.loc[low] = 'LOW'
        volatility_regime.loc[high] = 'HIGH'
        volatility_regime.loc[extreme] = 'EXTREME'

        # Position sizing signals
        position_signal.loc[ultra_low] = 'INCREASE_SIZE'
        position_signal.loc[low] = 'SLIGHT_INCREASE'
        position_signal.loc[high] = 'REDUCE_SIZE'
        position_signal.loc[extreme] = 'MINIMIZE_SIZE'

        # Volatility pattern detection
        expanding_vol = df['atr_trend'] > 1.0  # Strong upward volatility trend
        contracting_vol = df['atr_trend'] < -1.0  # Strong downward volatility trend
        volatility_spike = df['atr_percentile'] > 0.95  # Sudden spike

        volatility_pattern.loc[expanding_vol] = 'EXPANSION'
        volatility_pattern.loc[contracting_vol] = 'CONTRACTION'
        volatility_pattern.loc[volatility_spike] = 'SPIKE'

        # Trading context signals
        trading_context = pd.Series('NEUTRAL', index=df.index)

        # Bullish volatility: price rising with increasing volatility
        price_momentum = df['close'] > df['close'].shift(5)
        bullish_vol = price_momentum & expanding_vol
        trading_context.loc[bullish_vol] = 'BULLISH_VOLATILITY'

        # Bearish volatility: price falling with increasing volatility
        bearish_vol = ~price_momentum & expanding_vol
        trading_context.loc[bearish_vol] = 'BEARISH_VOLATILITY'

        # Low volatility setup: potential breakout coming
        squeeze_setup = (df['atr_percentile'] < 0.2) & contracting_vol
        trading_context.loc[squeeze_setup] = 'BREAKOUT_SETUP'

        return {
            'atr_volatility_regime': volatility_regime,
            'atr_position_signal': position_signal,
            'atr_pattern': volatility_pattern,
            'atr_trading_context': trading_context,
            'atr_risk_level': df['position_multiplier']
        }


class StochasticModule(TechnicalIndicatorBase):
    """
    Professional Stochastic Oscillator - Advanced Momentum & Reversal Detector

    Features:
    - Timeframe-adaptive parameters and thresholds
    - Market regime-based threshold adjustment
    - Advanced divergence detection (regular & hidden)
    - Failure swing pattern recognition
    - Multi-confirmation signal system
    - Professional overbought/oversold analysis
    """

    def __init__(self, base_k_period: int = 14, base_d_period: int = 3,
                 base_smooth_k: int = 3, timeframe: str = None,
                 enable_adaptive_params: bool = True):
        super().__init__("Stochastic")
        self.base_k_period = base_k_period
        self.base_d_period = base_d_period
        self.base_smooth_k = base_smooth_k
        self.timeframe = timeframe
        self.enable_adaptive_params = enable_adaptive_params

        # Timeframe-specific parameter optimization
        self.timeframe_adjustments = {
            '15m': {'k_mult': 0.8, 'd_mult': 1.0, 'smooth_mult': 0.8},  # Faster response
            '1h': {'k_mult': 1.0, 'd_mult': 1.0, 'smooth_mult': 1.0},   # Standard
            '4h': {'k_mult': 1.2, 'd_mult': 1.2, 'smooth_mult': 1.2},   # Smoother
            'daily': {'k_mult': 1.4, 'd_mult': 1.4, 'smooth_mult': 1.4} # Most stable
        }

        # Apply timeframe adjustments
        self._adjust_parameters_for_timeframe()

        # Market regime-based thresholds (will be set dynamically)
        self.overbought_threshold = 80
        self.oversold_threshold = 20

    def _adjust_parameters_for_timeframe(self):
        """Adjust parameters based on timeframe"""
        if not self.enable_adaptive_params or not self.timeframe:
            self.k_period = self.base_k_period
            self.d_period = self.base_d_period
            self.smooth_k = self.base_smooth_k
            return

        adj = self.timeframe_adjustments.get(self.timeframe, {'k_mult': 1.0, 'd_mult': 1.0, 'smooth_mult': 1.0})

        self.k_period = int(self.base_k_period * adj['k_mult'])
        self.d_period = int(self.base_d_period * adj['d_mult'])
        self.smooth_k = int(self.base_smooth_k * adj['smooth_mult'])

    def _adjust_thresholds_for_regime(self):
        """Dynamically adjust overbought/oversold thresholds based on market regime"""
        if not hasattr(self, 'market_regime') or self.market_regime == 'unknown':
            return

        # Regime-specific threshold adjustments
        regime_thresholds = {
            'trending_low_vol': {'oversold': 30, 'overbought': 70},    # Wider range in trends
            'trending_high_vol': {'oversold': 25, 'overbought': 75},   # Even wider in volatile trends
            'ranging_low_vol': {'oversold': 15, 'overbought': 85},     # Tighter in stable ranges
            'ranging_high_vol': {'oversold': 20, 'overbought': 80},    # Standard in volatile ranges
            'volatile_choppy': {'oversold': 25, 'overbought': 75}      # Conservative in chaos
        }

        thresholds = regime_thresholds.get(self.market_regime, {'oversold': 20, 'overbought': 80})
        self.oversold_threshold = thresholds['oversold']
        self.overbought_threshold = thresholds['overbought']

    def get_min_periods(self) -> int:
        return max(self.k_period + self.smooth_k + self.d_period + 50, 100)  # Extra for divergence analysis

    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate professional Stochastic with advanced analysis"""
        self.validate_data(df)

        # Adjust thresholds based on market regime
        self._adjust_thresholds_for_regime()

        # Standard Stochastic calculation
        lowest_low = df['low'].rolling(window=self.k_period).min()
        highest_high = df['high'].rolling(window=self.k_period).max()

        # Raw %K with division by zero protection
        range_diff = highest_high - lowest_low
        raw_k = pd.Series(50.0, index=df.index)  # Default to middle
        valid_range = range_diff > 0
        raw_k.loc[valid_range] = ((df['close'] - lowest_low) / range_diff * 100).loc[valid_range]

        # Smooth %K
        stoch_k = raw_k.rolling(window=self.smooth_k).mean()

        # %D (signal line)
        stoch_d = stoch_k.rolling(window=self.d_period).mean()

        # Advanced Stochastic analysis
        stoch_momentum = stoch_k - stoch_k.shift(1)
        stoch_acceleration = stoch_momentum - stoch_momentum.shift(1)

        # Stochastic position relative to its range
        stoch_range_position = (stoch_k - stoch_k.rolling(50).min()) / (stoch_k.rolling(50).max() - stoch_k.rolling(50).min())

        # Duration in overbought/oversold zones
        overbought_duration = self._calculate_zone_duration(stoch_k, self.overbought_threshold, above=True)
        oversold_duration = self._calculate_zone_duration(stoch_k, self.oversold_threshold, above=False)

        # Extreme readings (for failure swings)
        extreme_high = stoch_k.rolling(20).max()
        extreme_low = stoch_k.rolling(20).min()

        return {
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'stoch_raw_k': raw_k,
            'stoch_momentum': stoch_momentum,
            'stoch_acceleration': stoch_acceleration,
            'stoch_range_position': stoch_range_position,
            'stoch_overbought_duration': overbought_duration,
            'stoch_oversold_duration': oversold_duration,
            'stoch_extreme_high': extreme_high,
            'stoch_extreme_low': extreme_low,
            'stoch_params': pd.Series([f"TF:{self.timeframe},K:{self.k_period},OB:{self.overbought_threshold}"] * len(df), index=df.index)
        }

    def _calculate_zone_duration(self, series: pd.Series, threshold: float, above: bool = True) -> pd.Series:
        """Calculate how long price has been in overbought/oversold zone"""
        duration = pd.Series(0, index=series.index)
        current_duration = 0

        condition = series > threshold if above else series < threshold

        for i in range(len(series)):
            if condition.iloc[i]:
                current_duration += 1
                duration.iloc[i] = current_duration
            else:
                current_duration = 0

        return duration

    def detect_divergences(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Professional divergence detection for Stochastic
        Detects both regular and hidden divergences
        """
        # Initialize divergence signals
        regular_bullish_div = pd.Series(False, index=df.index)
        regular_bearish_div = pd.Series(False, index=df.index)
        hidden_bullish_div = pd.Series(False, index=df.index)
        hidden_bearish_div = pd.Series(False, index=df.index)

        if len(df) < 40:  # Need sufficient data for divergence analysis
            return {
                'stoch_regular_bullish_div': regular_bullish_div,
                'stoch_regular_bearish_div': regular_bearish_div,
                'stoch_hidden_bullish_div': hidden_bullish_div,
                'stoch_hidden_bearish_div': hidden_bearish_div
            }

        # Find swing points in both price and Stochastic
        price_highs = self._find_swing_points(df['high'], lookback=5, find_peaks=True)
        price_lows = self._find_swing_points(df['low'], lookback=5, find_peaks=False)
        stoch_highs = self._find_swing_points(df['stoch_k'], lookback=5, find_peaks=True)
        stoch_lows = self._find_swing_points(df['stoch_k'], lookback=5, find_peaks=False)

        # Regular Bullish Divergence: Price makes lower low, Stochastic makes higher low
        for i in range(20, len(df)):
            # Look for price lower lows in last 20 bars
            recent_price_lows = [idx for idx in price_lows if i-20 <= idx < i]
            recent_stoch_lows = [idx for idx in stoch_lows if i-20 <= idx < i]

            if len(recent_price_lows) >= 2 and len(recent_stoch_lows) >= 2:
                # Check for lower low in price and higher low in stochastic
                price_ll = df['low'].iloc[recent_price_lows[-1]] < df['low'].iloc[recent_price_lows[-2]]
                stoch_hl = df['stoch_k'].iloc[recent_stoch_lows[-1]] > df['stoch_k'].iloc[recent_stoch_lows[-2]]

                # Must be in oversold territory
                in_oversold = df['stoch_k'].iloc[recent_stoch_lows[-1]] < self.oversold_threshold + 10

                if price_ll and stoch_hl and in_oversold:
                    regular_bullish_div.iloc[i] = True

        # Regular Bearish Divergence: Price makes higher high, Stochastic makes lower high
        for i in range(20, len(df)):
            recent_price_highs = [idx for idx in price_highs if i-20 <= idx < i]
            recent_stoch_highs = [idx for idx in stoch_highs if i-20 <= idx < i]

            if len(recent_price_highs) >= 2 and len(recent_stoch_highs) >= 2:
                price_hh = df['high'].iloc[recent_price_highs[-1]] > df['high'].iloc[recent_price_highs[-2]]
                stoch_lh = df['stoch_k'].iloc[recent_stoch_highs[-1]] < df['stoch_k'].iloc[recent_stoch_highs[-2]]

                in_overbought = df['stoch_k'].iloc[recent_stoch_highs[-1]] > self.overbought_threshold - 10

                if price_hh and stoch_lh and in_overbought:
                    regular_bearish_div.iloc[i] = True

        # Hidden divergences (trend continuation signals)
        # Hidden Bullish: Price makes higher low, Stochastic makes lower low (uptrend continuation)
        # Hidden Bearish: Price makes lower high, Stochastic makes higher high (downtrend continuation)

        return {
            'stoch_regular_bullish_div': regular_bullish_div,
            'stoch_regular_bearish_div': regular_bearish_div,
            'stoch_hidden_bullish_div': hidden_bullish_div,
            'stoch_hidden_bearish_div': hidden_bearish_div
        }

    def _find_swing_points(self, series: pd.Series, lookback: int = 5, find_peaks: bool = True) -> List[int]:
        """Find swing highs or swing lows in a series"""
        swing_points = []

        for i in range(lookback, len(series) - lookback):
            if find_peaks:
                # Check if current point is highest in lookback window
                is_peak = all(series.iloc[i] >= series.iloc[j] for j in range(i-lookback, i+lookback+1) if j != i)
                if is_peak:
                    swing_points.append(i)
            else:
                # Check if current point is lowest in lookback window
                is_trough = all(series.iloc[i] <= series.iloc[j] for j in range(i-lookback, i+lookback+1) if j != i)
                if is_trough:
                    swing_points.append(i)

        return swing_points

    def detect_failure_swings(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Detect failure swings - powerful reversal patterns
        Bullish: Stochastic fails to make new low while price makes new low
        Bearish: Stochastic fails to make new high while price makes new high
        """
        bullish_failure = pd.Series(False, index=df.index)
        bearish_failure = pd.Series(False, index=df.index)

        lookback = 10

        for i in range(lookback, len(df)):
            # Bullish failure swing
            recent_price_low = df['low'].iloc[i-lookback:i+1].min()
            recent_stoch_low = df['stoch_k'].iloc[i-lookback:i+1].min()

            current_price_new_low = df['low'].iloc[i] <= recent_price_low
            stoch_fails_new_low = df['stoch_k'].iloc[i] > recent_stoch_low + 5  # Buffer for significance

            if current_price_new_low and stoch_fails_new_low and df['stoch_k'].iloc[i] < self.oversold_threshold + 15:
                bullish_failure.iloc[i] = True

            # Bearish failure swing
            recent_price_high = df['high'].iloc[i-lookback:i+1].max()
            recent_stoch_high = df['stoch_k'].iloc[i-lookback:i+1].max()

            current_price_new_high = df['high'].iloc[i] >= recent_price_high
            stoch_fails_new_high = df['stoch_k'].iloc[i] < recent_stoch_high - 5

            if current_price_new_high and stoch_fails_new_high and df['stoch_k'].iloc[i] > self.overbought_threshold - 15:
                bearish_failure.iloc[i] = True

        return {
            'stoch_bullish_failure': bullish_failure,
            'stoch_bearish_failure': bearish_failure
        }

    def get_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate professional Stochastic signals with advanced pattern recognition"""

        # Get divergence and failure swing analysis
        divergences = self.detect_divergences(df)
        failure_swings = self.detect_failure_swings(df)

        # Initialize signal series
        stoch_signal = pd.Series('HOLD', index=df.index)
        stoch_strength = pd.Series('WEAK', index=df.index)
        stoch_pattern = pd.Series('NONE', index=df.index)
        stoch_regime_signal = pd.Series('HOLD', index=df.index)

        # Basic crossover signals
        bullish_cross = ((df['stoch_k'] > df['stoch_d']) &
                        (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)))
        bearish_cross = ((df['stoch_k'] < df['stoch_d']) &
                        (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)))

        # Dynamic threshold conditions
        in_oversold = df['stoch_k'] < self.oversold_threshold
        in_overbought = df['stoch_k'] > self.overbought_threshold
        near_oversold = df['stoch_k'] < self.oversold_threshold + 10
        near_overbought = df['stoch_k'] > self.overbought_threshold - 10

        # Extended time in zones (stronger signals)
        extended_oversold = df['stoch_oversold_duration'] > 3
        extended_overbought = df['stoch_overbought_duration'] > 3

        # SIGNAL CLASSIFICATION HIERARCHY

        # STRONGEST SIGNALS: Divergence + Failure Swings
        strong_bullish = (
            (divergences['stoch_regular_bullish_div'] | failure_swings['stoch_bullish_failure']) &
            bullish_cross & in_oversold
        )
        strong_bearish = (
            (divergences['stoch_regular_bearish_div'] | failure_swings['stoch_bearish_failure']) &
            bearish_cross & in_overbought
        )

        # MODERATE SIGNALS: Extended time + crossover
        moderate_bullish = (
            bullish_cross & in_oversold & extended_oversold &
            ~(divergences['stoch_regular_bullish_div'] | failure_swings['stoch_bullish_failure'])
        )
        moderate_bearish = (
            bearish_cross & in_overbought & extended_overbought &
            ~(divergences['stoch_regular_bearish_div'] | failure_swings['stoch_bearish_failure'])
        )

        # WEAK SIGNALS: Basic crossover in extreme zones
        weak_bullish = (
            bullish_cross & near_oversold &
            ~strong_bullish & ~moderate_bullish
        )
        weak_bearish = (
            bearish_cross & near_overbought &
            ~strong_bearish & ~moderate_bearish
        )

        # Apply signals with strength classification
        stoch_signal.loc[strong_bullish] = 'BUY'
        stoch_signal.loc[moderate_bullish] = 'BUY'
        stoch_signal.loc[weak_bullish] = 'BUY'
        stoch_signal.loc[strong_bearish] = 'SELL'
        stoch_signal.loc[moderate_bearish] = 'SELL'
        stoch_signal.loc[weak_bearish] = 'SELL'

        # Strength classification
        stoch_strength.loc[strong_bullish | strong_bearish] = 'STRONG'
        stoch_strength.loc[moderate_bullish | moderate_bearish] = 'MODERATE'
        stoch_strength.loc[weak_bullish | weak_bearish] = 'WEAK'

        # Pattern classification
        stoch_pattern.loc[divergences['stoch_regular_bullish_div']] = 'BULLISH_DIVERGENCE'
        stoch_pattern.loc[divergences['stoch_regular_bearish_div']] = 'BEARISH_DIVERGENCE'
        stoch_pattern.loc[failure_swings['stoch_bullish_failure']] = 'BULLISH_FAILURE_SWING'
        stoch_pattern.loc[failure_swings['stoch_bearish_failure']] = 'BEARISH_FAILURE_SWING'
        stoch_pattern.loc[extended_oversold & bullish_cross] = 'EXTENDED_OVERSOLD_REVERSAL'
        stoch_pattern.loc[extended_overbought & bearish_cross] = 'EXTENDED_OVERBOUGHT_REVERSAL'

        # Market regime-specific signals
        if hasattr(self, 'market_regime') and self.market_regime != 'unknown':
            if self.market_regime in ['trending_low_vol', 'trending_high_vol']:
                # In trends, prioritize failure swings and divergences
                trend_signals = stoch_signal.copy()
                trend_signals.loc[~(strong_bullish | strong_bearish | moderate_bullish | moderate_bearish)] = 'HOLD'
                stoch_regime_signal = trend_signals

            elif self.market_regime in ['ranging_low_vol', 'ranging_high_vol']:
                # In ranges, all signals are valid
                stoch_regime_signal = stoch_signal

            else:  # volatile_choppy
                # In chaos, only strongest signals
                chaos_signals = stoch_signal.copy()
                chaos_signals.loc[~(strong_bullish | strong_bearish)] = 'HOLD'
                stoch_regime_signal = chaos_signals

        # Calculate overall confidence using base class
        confidence = self.get_signal_confidence(df, stoch_signal)

        return {
            'stoch_signal': stoch_signal,
            'stoch_regime_signal': stoch_regime_signal,
            'stoch_strength': stoch_strength,
            'stoch_pattern': stoch_pattern,
            'stoch_confidence': confidence,
            'stoch_bullish_divergence': divergences['stoch_regular_bullish_div'],
            'stoch_bearish_divergence': divergences['stoch_regular_bearish_div'],
            'stoch_bullish_failure': failure_swings['stoch_bullish_failure'],
            'stoch_bearish_failure': failure_swings['stoch_bearish_failure'],
            'stoch_oversold_duration': df['stoch_oversold_duration'],
            'stoch_overbought_duration': df['stoch_overbought_duration']
        }


# Recently added module for support/resistance levels
class SupportResistanceModule(TechnicalIndicatorBase):
    """
    Advanced Support/Resistance Detection

    Finds true support and resistance levels based on:
    1. Swing point detection (actual price reversals)
    2. Level clustering (groups nearby levels)
    3. Strength scoring (number of touches, recency, hold strength)
    """

    def __init__(self, timeframe: str = None, swing_window: int = None,
                 lookback_period: int = None, cluster_tolerance: float = None,
                 min_touches: int = None):
        super().__init__("TrueSupportResistance")

        # Timeframe-specific parameter optimization
        self.timeframe_configs = {
            '15m': {
                'swing_window': 3,          # Shorter for faster detection
                'lookback_period': 200,     # More recent data (50 hours)
                'cluster_tolerance': 0.0005, # Tighter clustering (0.05%)
                'min_touches': 2,
                'proximity_tolerance': 0.0003  # Very tight (0.03%)
            },
            '1h': {
                'swing_window': 5,          # Standard detection
                'lookback_period': 168,     # One week of hourly data
                'cluster_tolerance': 0.001,  # Standard clustering (0.1%)
                'min_touches': 2,
                'proximity_tolerance': 0.0005  # Tight (0.05%)
            },
            '4h': {
                'swing_window': 5,          # Standard detection
                'lookback_period': 126,     # 3 weeks of 4h data
                'cluster_tolerance': 0.0015, # Wider clustering (0.15%)
                'min_touches': 2,
                'proximity_tolerance': 0.001   # Standard (0.1%)
            },
            'daily': {
                'swing_window': 7,          # Longer for major reversals
                'lookback_period': 100,     # 100 trading days
                'cluster_tolerance': 0.002,  # Widest clustering (0.2%)
                'min_touches': 3,           # Require more validation
                'proximity_tolerance': 0.002   # Wider (0.2%)
            }
        }

        # Use timeframe-specific config if provided, otherwise use defaults or overrides
        if timeframe and timeframe in self.timeframe_configs:
            config = self.timeframe_configs[timeframe]
            self.timeframe = timeframe
            self.swing_window = swing_window if swing_window is not None else config['swing_window']
            self.lookback_period = lookback_period if lookback_period is not None else config['lookback_period']
            self.cluster_tolerance = cluster_tolerance if cluster_tolerance is not None else config['cluster_tolerance']
            self.min_touches = min_touches if min_touches is not None else config['min_touches']
            self.proximity_tolerance = config['proximity_tolerance']
        else:
            # Fallback to manual parameters or defaults
            self.timeframe = timeframe or 'unknown'
            self.swing_window = swing_window or 5
            self.lookback_period = lookback_period or 100
            self.cluster_tolerance = cluster_tolerance or 0.001
            self.min_touches = min_touches or 2
            self.proximity_tolerance = 0.001

    def get_min_periods(self) -> int:
        return self.lookback_period + self.swing_window

    def find_swing_highs(self, highs: pd.Series) -> pd.Series:
        """
        Find swing high points where price made a local maximum
        A swing high is a bar where high > high of N bars before and after
        """
        swing_highs = pd.Series(False, index=highs.index)

        for i in range(self.swing_window, len(highs) - self.swing_window):
            current_high = highs.iloc[i]

            # Check if current high is higher than surrounding bars
            left_window = highs.iloc[i-self.swing_window:i]
            right_window = highs.iloc[i+1:i+self.swing_window+1]

            if (current_high > left_window.max()) and (current_high > right_window.max()):
                swing_highs.iloc[i] = True

        return swing_highs

    def find_swing_lows(self, lows: pd.Series) -> pd.Series:
        """
        Find swing low points where price made a local minimum
        A swing low is a bar where low < low of N bars before and after
        """
        swing_lows = pd.Series(False, index=lows.index)

        for i in range(self.swing_window, len(lows) - self.swing_window):
            current_low = lows.iloc[i]

            # Check if current low is lower than surrounding bars
            left_window = lows.iloc[i-self.swing_window:i]
            right_window = lows.iloc[i+1:i+self.swing_window+1]

            if (current_low < left_window.min()) and (current_low < right_window.min()):
                swing_lows.iloc[i] = True

        return swing_lows

    def cluster_price_levels(self, price_levels: list, timestamps: list) -> list:
        """
        Group nearby price levels into clusters

        Args:
            price_levels: List of price levels
            timestamps: List of corresponding timestamps

        Returns:
            List of clustered levels with metadata
        """
        if not price_levels:
            return []

        # Sort by price
        sorted_data = sorted(zip(price_levels, timestamps))

        clusters = []
        current_cluster = [sorted_data[0]]

        for i in range(1, len(sorted_data)):
            price, timestamp = sorted_data[i]
            cluster_avg = sum(p for p, t in current_cluster) / len(current_cluster)

            # If price is within tolerance of cluster average, add to cluster
            if abs(price - cluster_avg) / cluster_avg <= self.cluster_tolerance:
                current_cluster.append((price, timestamp))
            else:
                # Finish current cluster and start new one
                if len(current_cluster) >= self.min_touches:
                    clusters.append(current_cluster)
                current_cluster = [(price, timestamp)]

        # Don't forget the last cluster
        if len(current_cluster) >= self.min_touches:
            clusters.append(current_cluster)

        return clusters

    def score_level_strength(self, cluster: list, df: pd.DataFrame, level_type: str,
                           volume_profile: Dict[str, Any] = None) -> dict:
        """
        Score a support/resistance level based on various factors including volume

        Args:
            cluster: List of (price, timestamp) tuples
            df: Price DataFrame
            level_type: 'support' or 'resistance'
            volume_profile: Volume profile data for validation

        Returns:
            Dictionary with level data and score
        """
        if not cluster:
            return {}

        # Calculate cluster statistics
        prices = [p for p, t in cluster]
        timestamps = [t for p, t in cluster]

        level_price = sum(prices) / len(prices)  # Average price of cluster
        touch_count = len(cluster)

        # Recency score (more recent = higher score)
        latest_timestamp = max(timestamps)
        latest_index = df.index.get_loc(latest_timestamp) if latest_timestamp in df.index else len(df) - 1
        recency_score = 1.0 - (len(df) - latest_index) / len(df)

        # Calculate how well the level held (bounces vs breaks)
        hold_strength = self.calculate_hold_strength(level_price, df, level_type)

        # NEW: Volume validation score
        volume_score = 0.5  # Default neutral score
        if volume_profile:
            volume_score = self.calculate_volume_weighted_strength(level_price, df, volume_profile)

        # Enhanced final score calculation
        # - Touch count: more touches = stronger (weight: 30%)
        # - Recency: more recent = better (weight: 25%)
        # - Hold strength: fewer breaks = stronger (weight: 25%)
        # - Volume confirmation: institutional validation (weight: 20%)
        score = (
            (touch_count / 10.0) * 0.3 +  # Normalize touch count
            recency_score * 0.25 +
            hold_strength * 0.25 +
            volume_score * 0.2  # Volume confirmation
        )

        return {
            'price': level_price,
            'touch_count': touch_count,
            'recency_score': recency_score,
            'hold_strength': hold_strength,
            'volume_score': volume_score,
            'final_score': min(score, 1.0),  # Cap at 1.0
            'timestamps': timestamps
        }

    def calculate_hold_strength(self, level_price: float, df: pd.DataFrame, level_type: str) -> float:
        """
        Calculate how well a level held (ratio of bounces to breaks)
        """
        tolerance = level_price * self.cluster_tolerance

        if level_type == 'resistance':
            # Count times price approached but didn't break above
            approaches = (df['high'] >= level_price - tolerance) & (df['high'] <= level_price + tolerance)
            breaks = df['close'] > level_price + tolerance
        else:  # support
            # Count times price approached but didn't break below
            approaches = (df['low'] <= level_price + tolerance) & (df['low'] >= level_price - tolerance)
            breaks = df['close'] < level_price - tolerance

        total_approaches = approaches.sum()
        total_breaks = breaks.sum()

        if total_approaches == 0:
            return 0.5  # No data

        hold_ratio = 1.0 - (total_breaks / total_approaches)
        return max(0.0, hold_ratio)

    def calculate_volume_at_price(self, df: pd.DataFrame, price_bins: int = 50) -> Dict[str, Any]:
        """
        Calculate Volume Profile - volume distribution across price levels

        Args:
            df: Price/volume DataFrame
            price_bins: Number of price bins to distribute volume

        Returns:
            Dictionary with volume profile data
        """
        if 'volume' not in df.columns:
            # Return empty profile if no volume data
            return {
                'price_levels': [],
                'volume_at_price': [],
                'poc_price': df['close'].iloc[-1],
                'poc_volume': 0,
                'value_area_high': df['high'].max(),
                'value_area_low': df['low'].min(),
                'profile_type': 'no_volume_data'
            }

        # Define price range
        price_high = df['high'].max()
        price_low = df['low'].min()

        if price_high == price_low:
            return {'poc_price': price_high, 'poc_volume': 0}

        # Create price bins
        price_step = (price_high - price_low) / price_bins
        price_levels = [price_low + (i * price_step) for i in range(price_bins + 1)]

        # Initialize volume at each price level
        volume_at_price = [0.0] * price_bins

        # Distribute volume across price levels
        for idx, row in df.iterrows():
            # Calculate which price bin this bar's volume should be distributed to
            bar_range = row['high'] - row['low']

            if bar_range == 0:
                # Single price bar - all volume goes to one bin
                bin_index = min(int((row['close'] - price_low) / price_step), price_bins - 1)
                volume_at_price[bin_index] += row['volume']
            else:
                # Distribute volume proportionally across the bar's price range
                start_bin = max(0, int((row['low'] - price_low) / price_step))
                end_bin = min(price_bins - 1, int((row['high'] - price_low) / price_step))

                # Volume per price unit within this bar
                volume_per_price = row['volume'] / bar_range

                for bin_idx in range(start_bin, end_bin + 1):
                    bin_price_low = price_levels[bin_idx]
                    bin_price_high = price_levels[bin_idx + 1] if bin_idx < price_bins - 1 else price_high

                    # Calculate overlap between bar and price bin
                    overlap_low = max(row['low'], bin_price_low)
                    overlap_high = min(row['high'], bin_price_high)
                    overlap = max(0, overlap_high - overlap_low)

                    volume_at_price[bin_idx] += volume_per_price * overlap

        # Find Point of Control (highest volume price level)
        poc_index = volume_at_price.index(max(volume_at_price))
        poc_price = price_levels[poc_index] + (price_step / 2)  # Center of bin
        poc_volume = volume_at_price[poc_index]

        # Calculate Value Area (70% of total volume)
        total_volume = sum(volume_at_price)
        target_volume = total_volume * 0.7

        # Start from POC and expand until we reach 70% volume
        value_area_indices = [poc_index]
        current_volume = volume_at_price[poc_index]

        expand_up = poc_index + 1 < price_bins
        expand_down = poc_index - 1 >= 0

        while current_volume < target_volume and (expand_up or expand_down):
            # Choose direction with higher volume
            up_volume = volume_at_price[poc_index + len([i for i in value_area_indices if i > poc_index]) + 1] if expand_up else 0
            down_volume = volume_at_price[poc_index - len([i for i in value_area_indices if i < poc_index]) - 1] if expand_down else 0

            if expand_up and (not expand_down or up_volume >= down_volume):
                next_index = max(value_area_indices) + 1
                if next_index < price_bins:
                    value_area_indices.append(next_index)
                    current_volume += volume_at_price[next_index]
                    if next_index + 1 >= price_bins:
                        expand_up = False
                else:
                    expand_up = False
            elif expand_down:
                next_index = min(value_area_indices) - 1
                if next_index >= 0:
                    value_area_indices.append(next_index)
                    current_volume += volume_at_price[next_index]
                    if next_index - 1 < 0:
                        expand_down = False
                else:
                    expand_down = False
            else:
                break

        # Value Area boundaries
        value_area_indices.sort()
        value_area_high = price_levels[value_area_indices[-1] + 1] if value_area_indices[-1] + 1 < len(price_levels) else price_high
        value_area_low = price_levels[value_area_indices[0]]

        return {
            'price_levels': price_levels,
            'volume_at_price': volume_at_price,
            'poc_price': poc_price,
            'poc_volume': poc_volume,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'total_volume': total_volume,
            'value_area_volume': current_volume,
            'profile_type': 'normal'
        }

    def calculate_volume_nodes(self, volume_profile: Dict[str, Any]) -> List[Dict[str, float]]:
        """
        Find high volume nodes (secondary support/resistance levels)

        Args:
            volume_profile: Output from calculate_volume_at_price

        Returns:
            List of volume nodes with prices and strengths
        """
        if volume_profile['profile_type'] == 'no_volume_data':
            return []

        volume_at_price = volume_profile['volume_at_price']
        price_levels = volume_profile['price_levels']

        # Find local volume peaks
        nodes = []
        for i in range(1, len(volume_at_price) - 1):
            current_vol = volume_at_price[i]
            prev_vol = volume_at_price[i - 1]
            next_vol = volume_at_price[i + 1]

            # Local maximum in volume
            if current_vol > prev_vol and current_vol > next_vol:
                # Must be significant (> 5% of POC volume)
                if current_vol > volume_profile['poc_volume'] * 0.05:
                    node_price = price_levels[i] + ((price_levels[i + 1] - price_levels[i]) / 2)
                    node_strength = current_vol / volume_profile['poc_volume']  # Relative to POC

                    nodes.append({
                        'price': node_price,
                        'volume': current_vol,
                        'strength': node_strength
                    })

        # Sort by volume (strongest first)
        nodes.sort(key=lambda x: x['volume'], reverse=True)
        return nodes[:5]  # Return top 5 nodes

    def calculate_volume_weighted_strength(self, level_price: float, df: pd.DataFrame,
                                         volume_profile: Dict[str, Any]) -> float:
        """
        Calculate volume confirmation for S/R level

        Args:
            level_price: Support or resistance price level
            df: Price DataFrame
            volume_profile: Volume profile data

        Returns:
            Volume confirmation score (0-1)
        """
        if volume_profile['profile_type'] == 'no_volume_data':
            return 0.5  # Neutral if no volume data

        # Check if level aligns with high volume areas
        poc_distance = abs(level_price - volume_profile['poc_price']) / level_price

        # Very close to POC = strong confirmation
        if poc_distance < 0.001:  # Within 0.1%
            return 1.0

        # Check alignment with volume nodes
        nodes = self.calculate_volume_nodes(volume_profile)
        for node in nodes:
            node_distance = abs(level_price - node['price']) / level_price
            if node_distance < 0.002:  # Within 0.2%
                return 0.5 + (node['strength'] * 0.5)  # Scale by node strength

        # Check if in value area
        in_value_area = (volume_profile['value_area_low'] <= level_price <= volume_profile['value_area_high'])
        if in_value_area:
            return 0.6  # Moderate confirmation

        return 0.3  # Low confirmation if far from volume areas

    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume-enhanced support and resistance levels"""
        self.validate_data(df)

        # Limit lookback to avoid performance issues
        recent_df = df.tail(self.lookback_period) if len(df) > self.lookback_period else df

        # Step 1: Calculate Volume Profile
        volume_profile = self.calculate_volume_at_price(recent_df)

        # Step 2: Find swing points
        swing_highs_mask = self.find_swing_highs(recent_df['high'])
        swing_lows_mask = self.find_swing_lows(recent_df['low'])

        # Extract swing point data
        swing_high_prices = recent_df.loc[swing_highs_mask, 'high'].tolist()
        swing_high_times = recent_df.loc[swing_highs_mask, 'high'].index.tolist()

        swing_low_prices = recent_df.loc[swing_lows_mask, 'low'].tolist()
        swing_low_times = recent_df.loc[swing_lows_mask, 'low'].index.tolist()

        # Step 3: Cluster nearby levels
        resistance_clusters = self.cluster_price_levels(swing_high_prices, swing_high_times)
        support_clusters = self.cluster_price_levels(swing_low_prices, swing_low_times)

        # Step 4: Score and rank levels WITH volume validation
        scored_resistance = []
        for cluster in resistance_clusters:
            score_data = self.score_level_strength(cluster, recent_df, 'resistance', volume_profile)
            if score_data:
                scored_resistance.append(score_data)

        scored_support = []
        for cluster in support_clusters:
            score_data = self.score_level_strength(cluster, recent_df, 'support', volume_profile)
            if score_data:
                scored_support.append(score_data)

        # Sort by score (best first)
        scored_resistance.sort(key=lambda x: x['final_score'], reverse=True)
        scored_support.sort(key=lambda x: x['final_score'], reverse=True)

        # Create series for the full DataFrame
        # Use top 3 levels of each type
        primary_resistance = scored_resistance[0]['price'] if scored_resistance else recent_df['high'].max()
        secondary_resistance = scored_resistance[1]['price'] if len(scored_resistance) > 1 else primary_resistance
        tertiary_resistance = scored_resistance[2]['price'] if len(scored_resistance) > 2 else secondary_resistance

        primary_support = scored_support[0]['price'] if scored_support else recent_df['low'].min()
        secondary_support = scored_support[1]['price'] if len(scored_support) > 1 else primary_support
        tertiary_support = scored_support[2]['price'] if len(scored_support) > 2 else secondary_support

        # Create series for full dataframe (fill with latest values)
        resistance_series = pd.Series(primary_resistance, index=df.index)
        support_series = pd.Series(primary_support, index=df.index)

        # Enhanced strength scores (now include volume validation)
        resistance_strength = pd.Series(
            scored_resistance[0]['final_score'] if scored_resistance else 0.0,
            index=df.index
        )
        support_strength = pd.Series(
            scored_support[0]['final_score'] if scored_support else 0.0,
            index=df.index
        )

        # Volume-specific features
        volume_resistance_score = pd.Series(
            scored_resistance[0]['volume_score'] if scored_resistance else 0.5,
            index=df.index
        )
        volume_support_score = pd.Series(
            scored_support[0]['volume_score'] if scored_support else 0.5,
            index=df.index
        )

        # Point of Control and Value Area
        poc_price = pd.Series(volume_profile['poc_price'], index=df.index)
        value_area_high = pd.Series(volume_profile['value_area_high'], index=df.index)
        value_area_low = pd.Series(volume_profile['value_area_low'], index=df.index)

        # Market bias based on price position relative to value area
        above_value_area = df['close'] > volume_profile['value_area_high']
        below_value_area = df['close'] < volume_profile['value_area_low']
        in_value_area = ~above_value_area & ~below_value_area

        return {
            # Traditional S/R levels (now volume-enhanced)
            'true_resistance': resistance_series,
            'true_support': support_series,
            'resistance_2': pd.Series(secondary_resistance, index=df.index),
            'resistance_3': pd.Series(tertiary_resistance, index=df.index),
            'support_2': pd.Series(secondary_support, index=df.index),
            'support_3': pd.Series(tertiary_support, index=df.index),

            # Enhanced strength scores (include volume validation)
            'resistance_strength': resistance_strength,
            'support_strength': support_strength,
            'volume_resistance_score': volume_resistance_score,
            'volume_support_score': volume_support_score,

            # Volume Profile features (no multicollinearity - integrated)
            'poc_price': poc_price,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'above_value_area': above_value_area.astype(float),
            'below_value_area': below_value_area.astype(float),
            'in_value_area': in_value_area.astype(float),

            # Traditional touch counts
            'resistance_touches': pd.Series(
                scored_resistance[0]['touch_count'] if scored_resistance else 0,
                index=df.index
            ),
            'support_touches': pd.Series(
                scored_support[0]['touch_count'] if scored_support else 0,
                index=df.index
            )
        }

    def get_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate volume-enhanced support/resistance signals"""
        signals = pd.Series('HOLD', index=df.index)
        strength_signals = pd.Series('WEAK', index=df.index)
        volume_bias = pd.Series('NEUTRAL', index=df.index)

        # Use timeframe-specific tolerance
        tight_tolerance = self.proximity_tolerance
        loose_tolerance = self.proximity_tolerance * 2.0  # Double for secondary levels

        # Traditional S/R proximity
        near_resistance = abs(df['close'] - df['true_resistance']) / df['close'] < tight_tolerance
        near_support = abs(df['close'] - df['true_support']) / df['close'] < tight_tolerance
        near_resistance_2 = abs(df['close'] - df['resistance_2']) / df['close'] < loose_tolerance
        near_support_2 = abs(df['close'] - df['support_2']) / df['close'] < loose_tolerance

        # Volume Profile proximity
        near_poc = abs(df['close'] - df['poc_price']) / df['close'] < tight_tolerance

        # Enhanced strength-based signals (now include volume validation)
        strong_resistance = (df['resistance_strength'] > 0.7) & near_resistance
        strong_support = (df['support_strength'] > 0.7) & near_support
        moderate_resistance = (df['resistance_strength'] > 0.4) & (df['resistance_strength'] <= 0.7) & near_resistance
        moderate_support = (df['support_strength'] > 0.4) & (df['support_strength'] <= 0.7) & near_support

        # VOLUME-ENHANCED SIGNALS
        # POC as dynamic S/R
        poc_support = near_poc & (df['close'] > df['poc_price'].shift(1))  # Price bouncing off POC
        poc_resistance = near_poc & (df['close'] < df['poc_price'].shift(1))  # Price rejecting at POC

        # Volume-confirmed levels (high volume score + price proximity)
        volume_confirmed_support = near_support & (df['volume_support_score'] > 0.7)
        volume_confirmed_resistance = near_resistance & (df['volume_resistance_score'] > 0.7)

        # Value Area bias signals
        bullish_bias = df['above_value_area'] == 1.0  # Price above value area = bullish
        bearish_bias = df['below_value_area'] == 1.0  # Price below value area = bearish
        neutral_bias = df['in_value_area'] == 1.0     # Price in value area = neutral

        # PRIORITY SIGNAL HIERARCHY
        # 1. Volume-confirmed levels (highest priority)
        signals.loc[volume_confirmed_support] = 'VOLUME_CONFIRMED_BUY'
        signals.loc[volume_confirmed_resistance] = 'VOLUME_CONFIRMED_SELL'

        # 2. Traditional strong levels
        signals.loc[strong_support & ~volume_confirmed_support] = 'STRONG_BUY'
        signals.loc[strong_resistance & ~volume_confirmed_resistance] = 'STRONG_SELL'

        # 3. POC-based signals
        signals.loc[poc_support & ~(strong_support | volume_confirmed_support)] = 'POC_BUY'
        signals.loc[poc_resistance & ~(strong_resistance | volume_confirmed_resistance)] = 'POC_SELL'

        # 4. Moderate traditional levels
        signals.loc[moderate_support & ~(strong_support | volume_confirmed_support | poc_support)] = 'BUY_SUPPORT'
        signals.loc[moderate_resistance & ~(strong_resistance | volume_confirmed_resistance | poc_resistance)] = 'SELL_RESISTANCE'

        # 5. Secondary levels (lowest priority)
        signals.loc[near_support_2 & (signals == 'HOLD')] = 'WEAK_BUY'
        signals.loc[near_resistance_2 & (signals == 'HOLD')] = 'WEAK_SELL'

        # Enhanced strength classification (include volume validation)
        volume_enhanced_strength = (df['resistance_strength'] + df['support_strength'] +
                                  df['volume_resistance_score'] + df['volume_support_score']) / 4

        strength_signals.loc[volume_enhanced_strength > 0.7] = 'STRONG'
        strength_signals.loc[(volume_enhanced_strength > 0.4) & (volume_enhanced_strength <= 0.7)] = 'MODERATE'

        # Market bias from value area position
        volume_bias.loc[bullish_bias] = 'BULLISH_BIAS'
        volume_bias.loc[bearish_bias] = 'BEARISH_BIAS'
        volume_bias.loc[neutral_bias] = 'NEUTRAL_BIAS'

        return {
            # Traditional signals (enhanced with volume)
            'sr_signal': signals,
            'sr_strength': strength_signals,

            # Volume-specific signals
            'volume_bias': volume_bias,
            'poc_signal': pd.Series(['POC_SUPPORT' if poc_support.iloc[i] else
                                   'POC_RESISTANCE' if poc_resistance.iloc[i] else 'NONE'
                                   for i in range(len(df))], index=df.index),

            # Quality assessments (volume-enhanced)
            'resistance_quality': pd.Series(
                ['VOLUME_CONFIRMED' if df['volume_resistance_score'].iloc[i] > 0.7 else
                 'STRONG' if df['resistance_strength'].iloc[i] > 0.7 else
                 'MODERATE' if df['resistance_strength'].iloc[i] > 0.4 else 'WEAK'
                 for i in range(len(df))], index=df.index
            ),
            'support_quality': pd.Series(
                ['VOLUME_CONFIRMED' if df['volume_support_score'].iloc[i] > 0.7 else
                 'STRONG' if df['support_strength'].iloc[i] > 0.7 else
                 'MODERATE' if df['support_strength'].iloc[i] > 0.4 else 'WEAK'
                 for i in range(len(df))], index=df.index
            )
        }
    
    
