"""
Volume Weighted Average Price (VWAP) Engine
Calculates:
- Anchored VWAP (Daily Session, Weekly, Monthly, or arbitrary timestamp)
- Rolling VWAP
- Dynamic Standard Deviation Bands (±1.0σ, ±2.0σ, ±3.0σ)
- VWAP slope & momentum metrics
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


class VWAPEngine:
    def __init__(self, anchor: str = "D"):
        """
        :param anchor: 'D' for Daily session, 'W' for Weekly, 'M' for Monthly,
                       or 'event' for timestamp-specific anchoring.
        """
        self.anchor = anchor

    def compute(
        self,
        df: pd.DataFrame,
        price_col: str = "typical_price",
        volume_col: str = "volume",
        timestamp_col: str = "datetime",
        custom_anchor_mask: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Computes VWAP and 3 Standard Deviation bands.
        Expects df to have price (or High, Low, Close) and volume.
        """
        data = df.copy()
        
        # Ensure typical price exists
        if price_col not in data.columns:
            if all(col in data.columns for col in ["high", "low", "close"]):
                data["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3.0
                price_col = "typical_price"
            elif "close" in data.columns:
                data["typical_price"] = data["close"]
                price_col = "typical_price"
            else:
                raise ValueError("DataFrame must contain 'typical_price' or 'high','low','close' columns.")

        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
            data[timestamp_col] = pd.to_datetime(data[timestamp_col])

        # Define anchoring groups
        if custom_anchor_mask is not None:
            # Each True starts a new group
            group_ids = custom_anchor_mask.cumsum()
        elif self.anchor == "D":
            group_ids = data[timestamp_col].dt.floor("D")
        elif self.anchor == "W":
            group_ids = data[timestamp_col].dt.to_period("W").apply(lambda r: r.start_time)
        elif self.anchor == "M":
            group_ids = data[timestamp_col].dt.to_period("M").apply(lambda r: r.start_time)
        else:
            group_ids = pd.Series(0, index=data.index)

        pv = data[price_col] * data[volume_col]
        cum_pv = pv.groupby(group_ids).cumsum()
        cum_vol = data[volume_col].groupby(group_ids).cumsum()

        # Prevent division by zero
        cum_vol_safe = cum_vol.replace(0, np.nan)
        vwap = cum_pv / cum_vol_safe

        # Calculate standard deviation bands
        squared_diff = data[volume_col] * ((data[price_col] - vwap) ** 2)
        cum_sq_diff = squared_diff.groupby(group_ids).cumsum()
        variance = (cum_sq_diff / cum_vol_safe).clip(lower=0)
        std_dev = np.sqrt(variance)

        data["vwap"] = vwap
        data["vwap_upper_1"] = vwap + (1.0 * std_dev)
        data["vwap_lower_1"] = vwap - (1.0 * std_dev)
        data["vwap_upper_2"] = vwap + (2.0 * std_dev)
        data["vwap_lower_2"] = vwap - (2.0 * std_dev)
        data["vwap_upper_3"] = vwap + (3.0 * std_dev)
        data["vwap_lower_3"] = vwap - (3.0 * std_dev)
        data["vwap_std"] = std_dev
        
        # Distance metric (in standard deviations / z-score from VWAP)
        data["vwap_zscore"] = np.where(std_dev > 0, (data[price_col] - vwap) / std_dev, 0.0)

        return data
