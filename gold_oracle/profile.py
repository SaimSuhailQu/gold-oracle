"""
Auction Market Theory Engine
Implements:
1. Volume Profile (Volume at Price distribution)
   - Point of Control (VPOC)
   - Value Area High (VAH) & Value Area Low (VAL) [default 70% distribution]
   - High Volume Nodes (HVN) & Low Volume Nodes (LVN)
2. Market Profile (Time Price Opportunity - TPO)
   - Initial Balance (IB High / IB Low)
   - Single Prints (Unaccepted auction price discovery)
   - Profile Shape Classification (Normal, P-shape, b-shape, Neutral)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class ProfileEngine:
    def __init__(self, tick_size: float = 0.1, value_area_pct: float = 0.70):
        """
        :param tick_size: Minimum price bucket step (e.g. 0.1 for Gold / XAUUSD)
        :param value_area_pct: Percentage of auction volume to consider as fair value (typically 70%)
        """
        self.tick_size = tick_size
        self.value_area_pct = value_area_pct

    def compute_volume_profile(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        volume_col: str = "volume"
    ) -> Dict[str, Any]:
        """
        Calculates the Volume Profile distribution across price buckets.
        """
        if df.empty or volume_col not in df.columns:
            return {}

        # Round prices to bucket steps
        bucketed_prices = (df[price_col] / self.tick_size).round() * self.tick_size
        profile = df.groupby(bucketed_prices)[volume_col].sum().sort_index()

        if profile.empty:
            return {}

        total_volume = profile.sum()
        target_va_vol = total_volume * self.value_area_pct

        # Volume Point of Control (VPOC) - price level with maximum volume
        vpoc_price = profile.idxmax()
        vpoc_vol = profile.max()

        # Calculate Value Area (70% expansion from POC)
        poc_idx = profile.index.get_loc(vpoc_price)
        upper_idx = poc_idx
        lower_idx = poc_idx
        accumulated_vol = vpoc_vol

        prices = profile.index.values
        vols = profile.values
        n = len(prices)

        while accumulated_vol < target_va_vol and (lower_idx > 0 or upper_idx < n - 1):
            next_upper_vol = vols[upper_idx + 1] if upper_idx + 1 < n else 0
            next_lower_vol = vols[lower_idx - 1] if lower_idx - 1 >= 0 else 0

            if next_upper_vol >= next_lower_vol and upper_idx + 1 < n:
                upper_idx += 1
                accumulated_vol += next_upper_vol
            elif lower_idx - 1 >= 0:
                lower_idx -= 1
                accumulated_vol += next_lower_vol
            elif upper_idx + 1 < n:
                upper_idx += 1
                accumulated_vol += next_upper_vol
            else:
                break

        val = prices[lower_idx]
        vah = prices[upper_idx]

        # Identify High Volume Nodes (HVNs) and Low Volume Nodes (LVNs)
        vol_mean = profile.mean()
        vol_std = profile.std() if len(profile) > 1 else 1.0

        hvns = profile[profile >= (vol_mean + 0.8 * vol_std)].index.tolist()
        lvns = profile[profile <= max(0, vol_mean - 0.8 * vol_std)].index.tolist()

        return {
            "poc": float(vpoc_price),
            "vah": float(vah),
            "val": float(val),
            "total_volume": float(total_volume),
            "profile_distribution": {float(k): float(v) for k, v in profile.items()},
            "hvns": [float(p) for p in hvns],
            "lvns": [float(p) for p in lvns],
        }

    def compute_tpo_profile(
        self,
        df: pd.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        timestamp_col: str = "datetime",
        period_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Builds Time-Price-Opportunity (TPO) Market Profile structure.
        Tracks Initial Balance (first 1 hour / 2 periods) and profile skewness.
        """
        if df.empty:
            return {}

        data = df.sort_values(timestamp_col).copy()
        
        # Partition into 30m TPO brackets (A, B, C, ...)
        tpo_counts: Dict[float, int] = {}
        bracket_labels = [chr(ord('A') + i) if i < 26 else f"Z{i}" for i in range(100)]
        
        # Group by 30-min brackets
        grouped = data.groupby(pd.Grouper(key=timestamp_col, freq=f"{period_minutes}min"))
        
        ib_high = None
        ib_low = None
        bracket_num = 0

        for _, group in grouped:
            if group.empty:
                continue
            
            p_high = group[high_col].max()
            p_low = group[low_col].min()

            # Initial Balance: First 2 brackets (first 60 minutes)
            if bracket_num == 0:
                ib_high = p_high
                ib_low = p_low
            elif bracket_num == 1:
                ib_high = max(ib_high, p_high)
                ib_low = min(ib_low, p_low)

            # Mark TPO ticks
            min_tick = int(np.floor(p_low / self.tick_size))
            max_tick = int(np.ceil(p_high / self.tick_size))

            for t in range(min_tick, max_tick + 1):
                level = round(t * self.tick_size, 4)
                tpo_counts[level] = tpo_counts.get(level, 0) + 1

            bracket_num += 1

        if not tpo_counts:
            return {}

        tpo_series = pd.Series(tpo_counts).sort_index()
        tpo_poc = tpo_series.idxmax()

        # Classify profile shape:
        # P-shape: POC near top (short covering / buying exhaustion at tops)
        # b-shape: POC near bottom (long liquidation / supply absorption)
        # D-shape: POC centered (balanced auction)
        profile_range = tpo_series.index.max() - tpo_series.index.min()
        if profile_range > 0:
            poc_relative_pos = (tpo_poc - tpo_series.index.min()) / profile_range
            if poc_relative_pos > 0.65:
                profile_type = "P_SHAPE_BULLISH_UP_EXTREME"
            elif poc_relative_pos < 0.35:
                profile_type = "b_SHAPE_BEARISH_DOWN_EXTREME"
            else:
                profile_type = "D_SHAPE_BALANCED"
        else:
            profile_type = "FLAT"

        return {
            "tpo_poc": float(tpo_poc),
            "ib_high": float(ib_high) if ib_high is not None else None,
            "ib_low": float(ib_low) if ib_low is not None else None,
            "profile_type": profile_type,
            "tpo_levels": {float(k): int(v) for k, v in tpo_series.items()}
        }
