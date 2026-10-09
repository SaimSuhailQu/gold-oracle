"""
Order Book Heatmap & Depth Engine
Simulates and tracks Level 2 / DOM (Depth of Market) resting liquidity:
1. Resting Limit Order Concentrations (Support & Resistance Heatmap)
2. Spoofing & Liquidity Pulling Detection:
   - Tracks cancellation rate vs filled rate of large resting size
3. Bid/Ask Depth Imbalance:
   - Book Pressure Ratio (Cumulative Bids / Cumulative Asks within N ticks)
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class HeatmapEngine:
    def __init__(self, depth_levels: int = 20, tick_size: float = 0.1):
        self.depth_levels = depth_levels
        self.tick_size = tick_size

    def compute_book_metrics(
        self,
        bids: List[Dict[str, float]],  # [{'price': float, 'volume': float}, ...]
        asks: List[Dict[str, float]],  # [{'price': float, 'volume': float}, ...]
        spot_price: float
    ) -> Dict[str, Any]:
        """
        Analyzes a snapshot of resting liquidity.
        """
        if not bids or not asks:
            return {}

        bids_df = pd.DataFrame(bids).sort_values("price", ascending=False).head(self.depth_levels)
        asks_df = pd.DataFrame(asks).sort_values("price", ascending=True).head(self.depth_levels)

        total_bid_depth = float(bids_df["volume"].sum())
        total_ask_depth = float(asks_df["volume"].sum())

        total_liquidity = total_bid_depth + total_ask_depth
        book_skew = (total_bid_depth - total_ask_depth) / total_liquidity if total_liquidity > 0 else 0.0

        # Liquidity concentration nodes (Hot spots on the heatmap)
        mean_bid = bids_df["volume"].mean()
        mean_ask = asks_df["volume"].mean()

        bid_walls = bids_df[bids_df["volume"] > 2.0 * mean_bid]
        ask_walls = asks_df[asks_df["volume"] > 2.0 * mean_ask]

        return {
            "spot_price": spot_price,
            "total_bid_depth": total_bid_depth,
            "total_ask_depth": total_ask_depth,
            "book_skew": round(book_skew, 4),  # > 0: Bid heavy, < 0: Ask heavy
            "resting_bid_walls": bid_walls.to_dict(orient="records"),
            "resting_ask_walls": ask_walls.to_dict(orient="records"),
            "spread": round(asks_df.iloc[0]["price"] - bids_df.iloc[0]["price"], 4)
        }

    def detect_liquidity_pulls(
        self,
        prev_depth_snapshot: pd.DataFrame,
        curr_depth_snapshot: pd.DataFrame,
        pull_threshold_pct: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Identifies potential spoofing / flash cancellations where large size disappears
        without price trading through that level.
        """
        merged = pd.merge(
            prev_depth_snapshot,
            curr_depth_snapshot,
            on="price",
            suffixes=("_prev", "_curr")
        )

        pulled_levels = []
        for _, row in merged.iterrows():
            vol_prev = row["volume_prev"]
            vol_curr = row["volume_curr"]
            if vol_prev > 50 and vol_curr < vol_prev * (1.0 - pull_threshold_pct):
                pulled_levels.append({
                    "price": float(row["price"]),
                    "prev_volume": float(vol_prev),
                    "curr_volume": float(vol_curr),
                    "cancelled_volume": float(vol_prev - vol_curr),
                    "type": "LIQUIDITY_PULL_SPOOF"
                })

        return pulled_levels
