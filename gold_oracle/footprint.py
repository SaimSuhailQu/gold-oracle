"""
Footprint & Delta Engine
Computes:
1. Bid-Ask Order Flow Aggregation:
   - Volume at Ask (aggressive buyers lifting the offer)
   - Volume at Bid (aggressive sellers hitting the bid)
   - Imbalance detection (e.g. 300% / 3:1 diagonal imbalance)
2. Delta Metrics:
   - Bar Delta (Ask Volume - Bid Volume)
   - Cumulative Volume Delta (CVD)
   - Delta Divergences (Price making higher highs with decreasing or negative CVD)
   - Unfinished auctions (Zero bid at low or zero ask at high)
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class FootprintDeltaEngine:
    def __init__(self, imbalance_ratio: float = 3.0, min_imbalance_vol: float = 10.0):
        """
        :param imbalance_ratio: Diagonal imbalance multiplier (default 3x or 300%)
        :param min_imbalance_vol: Minimum volume difference to qualify as institutional imbalance
        """
        self.imbalance_ratio = imbalance_ratio
        self.min_imbalance_vol = min_imbalance_vol

    def compute_candle_footprint(
        self,
        ticks_df: pd.DataFrame,
        price_col: str = "price",
        volume_col: str = "volume",
        side_col: str = "side",
        tick_size: float = 0.1
    ) -> Dict[str, Any]:
        """
        Processes raw ticks (with side: 'buy'/'sell' or 1/-1) into a single candle's Footprint matrix.
        """
        if ticks_df.empty:
            return {}

        df = ticks_df.copy()
        df["bucket"] = (df[price_col] / tick_size).round() * tick_size
        
        # Split buy (ask lifted) vs sell (bid hit)
        df["is_buy"] = df[side_col].astype(str).str.lower().isin(["buy", "ask", "1"])
        
        buy_vol = df[df["is_buy"]].groupby("bucket")[volume_col].sum()
        sell_vol = df[~df["is_buy"]].groupby("bucket")[volume_col].sum()

        all_prices = sorted(list(set(buy_vol.index).union(set(sell_vol.index))))
        matrix = []

        total_ask_vol = 0.0
        total_bid_vol = 0.0

        for p in all_prices:
            ask_v = float(buy_vol.get(p, 0.0))
            bid_v = float(sell_vol.get(p, 0.0))
            delta_v = ask_v - bid_v

            total_ask_vol += ask_v
            total_bid_vol += bid_v

            matrix.append({
                "price": round(p, 4),
                "bid_vol": bid_v,
                "ask_vol": ask_v,
                "delta": delta_v,
                "total": ask_v + bid_v
            })

        matrix_df = pd.DataFrame(matrix)
        
        # Diagonal Imbalance Analysis:
        # Ask at price [i] vs Bid at price [i-1]
        imbalances = []
        if len(matrix_df) > 1:
            for i in range(len(matrix_df)):
                # Buy imbalance (Ask[i] >> Bid[i-1])
                if i > 0:
                    prev_bid = matrix_df.loc[i - 1, "bid_vol"]
                    curr_ask = matrix_df.loc[i, "ask_vol"]
                    if prev_bid > 0 and (curr_ask / prev_bid >= self.imbalance_ratio) and (curr_ask - prev_bid >= self.min_imbalance_vol):
                        imbalances.append({
                            "type": "BUY_IMBALANCE",
                            "price": matrix_df.loc[i, "price"],
                            "ratio": round(curr_ask / prev_bid, 2)
                        })
                # Sell imbalance (Bid[i] >> Ask[i+1])
                if i < len(matrix_df) - 1:
                    next_ask = matrix_df.loc[i + 1, "ask_vol"]
                    curr_bid = matrix_df.loc[i, "bid_vol"]
                    if next_ask > 0 and (curr_bid / next_ask >= self.imbalance_ratio) and (curr_bid - next_ask >= self.min_imbalance_vol):
                        imbalances.append({
                            "type": "SELL_IMBALANCE",
                            "price": matrix_df.loc[i, "price"],
                            "ratio": round(curr_bid / next_ask, 2)
                        })

        bar_delta = total_ask_vol - total_bid_vol
        total_vol = total_ask_vol + total_bid_vol
        delta_pct = (bar_delta / total_vol * 100.0) if total_vol > 0 else 0.0

        # Unfinished auction check (bid at high or ask at low)
        highest_row = matrix_df.iloc[-1]
        lowest_row = matrix_df.iloc[0]
        unfinished_high = highest_row["bid_vol"] > 0 and highest_row["ask_vol"] > 0
        unfinished_low = lowest_row["bid_vol"] > 0 and lowest_row["ask_vol"] > 0

        return {
            "matrix": matrix,
            "bar_delta": round(bar_delta, 2),
            "delta_pct": round(delta_pct, 2),
            "total_volume": round(total_vol, 2),
            "imbalances": imbalances,
            "unfinished_auction_high": bool(unfinished_high),
            "unfinished_auction_low": bool(unfinished_low)
        }

    def compute_cvd_series(self, ohlcv_with_delta: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Cumulative Volume Delta (CVD) and flags divergences.
        """
        df = ohlcv_with_delta.copy()
        if "delta" not in df.columns:
            # Approximation if tick data isn't present: buy/sell volume split from bar direction & wick
            df["delta"] = df["volume"] * np.where(df["close"] >= df["open"], 1, -1)

        df["cvd"] = df["delta"].cumsum()
        
        # Rolling divergence detection (5-period lookback)
        df["price_high_5"] = df["close"].rolling(5).max()
        df["cvd_high_5"] = df["cvd"].rolling(5).max()
        df["price_low_5"] = df["close"].rolling(5).min()
        df["cvd_low_5"] = df["cvd"].rolling(5).min()

        # Bearish divergence: Price makes higher high while CVD makes lower high
        df["bearish_cvd_divergence"] = (df["close"] >= df["price_high_5"]) & (df["cvd"] < df["cvd_high_5"])
        # Bullish divergence: Price makes lower low while CVD makes higher low
        df["bullish_cvd_divergence"] = (df["close"] <= df["price_low_5"]) & (df["cvd"] > df["cvd_low_5"])

        return df
