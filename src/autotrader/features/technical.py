"""Technical indicators (all operate on pandas Series/DataFrames, NO lookahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Trend / Moving Averages
# ---------------------------------------------------------------------------


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------------
# RSI (Wilder's smoothing)
# ---------------------------------------------------------------------------


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing method.

    Returns a Series in range [0, 100].  The first *period* values will be NaN.
    """
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing is equivalent to EWM with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    _rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss == 0 (no losses at all), RSI is 100
    _rsi = _rsi.where(avg_loss != 0, 100.0)

    # Enforce NaN for the initial warm-up window
    _rsi.iloc[:period] = np.nan

    return _rsi


# ---------------------------------------------------------------------------
# ADX / ATR
# ---------------------------------------------------------------------------


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index.

    Uses Wilder's smoothing for +DI, -DI, and the ADX itself.
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = (high - prev_high).clip(lower=0.0)
    minus_dm = (prev_low - low).clip(lower=0.0)

    # Zero out whichever is smaller
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)

    _atr = atr(high, low, close, period)

    alpha = 1.0 / period

    smoothed_plus_dm = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * smoothed_plus_dm / _atr
    minus_di = 100.0 * smoothed_minus_dm / _atr

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)

    _adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # Warm-up: need 2 * period - 1 bars before ADX is meaningful
    _adx.iloc[: 2 * period - 1] = np.nan

    return _adx


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Returns (upper, middle, lower).
    """
    middle = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def bb_width(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger Band width as a percentage of the middle band.

    width = (upper - lower) / middle
    """
    upper, middle, lower = bollinger_bands(close, period, num_std)
    width = (upper - lower) / middle
    # Sanitize inf values that arise when middle band is zero
    width = width.replace([np.inf, -np.inf], np.nan)
    return width


def bb_width_percentile(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
    lookback: int = 100,
) -> pd.Series:
    """Rolling percentile rank of BB width over *lookback* bars.

    Returns values in [0, 1].
    """
    width = bb_width(close, period, num_std)

    def _percentile_rank(window: pd.Series) -> float:
        if len(window) < 2:
            return np.nan
        current = window.iloc[-1]
        count_below = (window.iloc[:-1] < current).sum()
        return count_below / (len(window) - 1)

    return width.rolling(window=lookback, min_periods=lookback).apply(_percentile_rank, raw=False)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator.

    Returns (macd_line, signal_line, histogram).
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Moving Average Slope
# ---------------------------------------------------------------------------


def ma_slope(series: pd.Series, period: int = 20, lookback: int = 5) -> pd.Series:
    """Slope of the SMA over *lookback* bars.

    Computed as (sma_now - sma_{lookback_ago}) / lookback.
    """
    _sma = sma(series, period)
    return (_sma - _sma.shift(lookback)) / lookback


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def realized_vol(close: pd.Series, period: int = 20) -> pd.Series:
    """Annualized realized volatility from log returns.

    Annualization factor assumes ~365 trading days (crypto markets).
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=period, min_periods=period).std(ddof=1) * np.sqrt(365)


def parkinson_vol(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """Parkinson volatility estimator.

    Uses the high-low range to estimate volatility, annualized for crypto (365 days).
    Formula: sqrt( (1 / (4 * n * ln2)) * sum(ln(H/L)^2) )
    """
    log_hl = np.log(high / low)
    sq = log_hl**2
    factor = 1.0 / (4.0 * np.log(2.0))
    variance = factor * sq.rolling(window=period, min_periods=period).mean()
    return np.sqrt(variance) * np.sqrt(365)


# ---------------------------------------------------------------------------
# Hurst Exponent
# ---------------------------------------------------------------------------


def hurst_exponent(close: pd.Series, max_lag: int = 20) -> pd.Series:
    """Rolling Hurst exponent estimate using the simplified R/S method.

    * H < 0.5 => mean-reverting
    * H ~ 0.5 => random walk
    * H > 0.5 => trending

    Computation is done on a rolling window of size *max_lag* bars.
    The method fits a line to log(R/S) vs log(n) for sub-divisions of the window.
    """
    log_ret = np.log(close / close.shift(1))

    window_size = max_lag

    def _rs_hurst(window: np.ndarray) -> float:
        """Compute Hurst exponent for a single window of log returns."""
        if len(window) < 8:
            return np.nan

        n = len(window)
        # Use several lag sizes (powers of 2 that fit within the window)
        lags = []
        lag = 4
        while lag <= n:
            lags.append(lag)
            lag *= 2

        if len(lags) < 2:
            return np.nan

        rs_values = []
        for lag_size in lags:
            num_blocks = n // lag_size
            if num_blocks == 0:
                continue

            rs_list = []
            for i in range(num_blocks):
                block = window[i * lag_size : (i + 1) * lag_size]
                mean_block = np.mean(block)
                deviations = np.cumsum(block - mean_block)
                r = np.max(deviations) - np.min(deviations)
                s = np.std(block, ddof=1)
                if s > 1e-12:
                    rs_list.append(r / s)

            if rs_list:
                rs_values.append((lag_size, np.mean(rs_list)))

        if len(rs_values) < 2:
            return np.nan

        log_lags = np.log([v[0] for v in rs_values])
        log_rs = np.log([v[1] for v in rs_values])

        # Linear regression: log(R/S) = H * log(n) + c
        coeffs = np.polyfit(log_lags, log_rs, 1)
        hurst = coeffs[0]

        # Clamp to [0, 1]
        return float(np.clip(hurst, 0.0, 1.0))

    result = log_ret.rolling(window=window_size, min_periods=window_size).apply(_rs_hurst, raw=True)

    return result


# ---------------------------------------------------------------------------
# Mean-Reversion Half-Life (Ornstein–Uhlenbeck)
# ---------------------------------------------------------------------------


def mean_reversion_half_life(close: pd.Series, lookback: int = 50) -> pd.Series:
    """Rolling estimate of the mean-reversion half-life using an AR(1) model.

    The half-life is defined as ``-ln(2) / ln(beta)`` where *beta* is the
    OLS slope of ``Δy_t`` on ``y_{t-1}`` (de-meaned price).  A smaller
    half-life indicates faster mean reversion.

    Returns ``NaN`` when *beta* >= 0 (no mean reversion detected) or when
    insufficient data is available.
    """
    log_ret = np.log(close / close.shift(1))

    def _half_life(window: np.ndarray) -> float:
        y = window[1:]
        y_lag = window[:-1]
        if len(y) < 10:
            return np.nan
        y_lag_dm = y_lag - np.mean(y_lag)
        dy = y - y_lag
        denom = np.dot(y_lag_dm, y_lag_dm)
        if denom < 1e-15:
            return np.nan
        beta = np.dot(y_lag_dm, dy) / denom
        if beta >= 0:
            return np.nan  # not mean-reverting
        hl = -np.log(2) / np.log(1 + beta)
        return max(1.0, min(hl, 500.0))  # clamp to sensible range

    return close.rolling(window=lookback, min_periods=lookback).apply(_half_life, raw=True)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average of volume."""
    return sma(volume, period)


# ---------------------------------------------------------------------------
# Candle Shape
# ---------------------------------------------------------------------------


def wick_ratio(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Wick ratio: (upper_wick + lower_wick) / body.

    * body = abs(close - open)
    * upper_wick = high - max(open, close)
    * lower_wick = min(open, close) - low

    When body is zero (doji), the ratio is set to NaN.
    """
    body = (close - open_).abs()
    upper = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower = pd.concat([open_, close], axis=1).min(axis=1) - low

    total_wick = upper + lower

    ratio = total_wick / body
    # Doji / zero-body candles -> NaN
    ratio = ratio.where(body > 1e-12, np.nan)

    return ratio
