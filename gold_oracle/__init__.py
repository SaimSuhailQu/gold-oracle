"""Gold Oracle — probability model for gold (XAUUSD) direction from macro factors.

Pipeline: data.py (fetch) -> factors.py (z-scored features, bullish-positive)
-> model.py (composite -> calibrated probability) -> backtest.py / signal.py
"""
