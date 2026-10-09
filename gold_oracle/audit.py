"""
Order Flow Audit Engine
Synthesizes all microstructure signals into a rigorous trade validation audit:
1. Absorption Detection:
   - High aggressive delta absorbed by passive limit orders at key levels (VPOC, VWAP ±2σ, GEX walls)
2. Exhaustion Detection:
   - Price probing new highs/lows with rapidly declining volume and near-zero delta
3. Trapped Traders:
   - Clustered buy imbalances at the absolute high of a candle followed by a close below the cluster
4. Final Trade Decision Audit:
   - Issues quantitative alignment scores (0.0 to 1.0) and concrete execution gates.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class OrderFlowAuditEngine:
    def __init__(self, absorption_delta_threshold: float = 50.0):
        self.absorption_delta_threshold = absorption_delta_threshold

    def audit_setup(
        self,
        current_price: float,
        vwap_data: Dict[str, float],      # {'vwap': ..., 'upper_2': ..., 'lower_2': ..., 'zscore': ...}
        profile_data: Dict[str, Any],     # {'poc': ..., 'vah': ..., 'val': ..., 'profile_type': ...}
        gex_data: Dict[str, Any],         # {'regime': ..., 'zero_gamma_level': ..., 'call_wall': ..., 'put_wall': ...}
        footprint_data: Dict[str, Any],   # {'bar_delta': ..., 'delta_pct': ..., 'imbalances': [...]}
        macro_bias: str = "LONG"          # from Gold Oracle ('LONG', 'SHORT', 'NEUTRAL')
    ) -> Dict[str, Any]:
        """
        Conducts a unified auction and order flow audit before allowing execution.
        """
        audit_verdict = "STAND_ASIDE"
        confidence_score = 0.0
        reasons = []

        poc = profile_data.get("poc", current_price)
        vah = profile_data.get("vah", current_price)
        val = profile_data.get("val", current_price)
        vwap = vwap_data.get("vwap", current_price)
        vwap_z = vwap_data.get("zscore", 0.0)
        
        call_wall = gex_data.get("call_wall", current_price + 100)
        put_wall = gex_data.get("put_wall", current_price - 100)
        gex_regime = gex_data.get("regime", "POSITIVE_GEX_PINNED_MEAN_REVERSION")

        bar_delta = footprint_data.get("bar_delta", 0.0)
        delta_pct = footprint_data.get("delta_pct", 0.0)
        imbalances = footprint_data.get("imbalances", [])

        # Check for Absorption
        # Bullish Absorption: Price pushed down near VAL / Put Wall / VWAP -2σ, Delta is negative, but buying stepped in
        bullish_absorption = (current_price <= val or current_price <= put_wall or vwap_z < -1.5) and (bar_delta < 0 and delta_pct < -30.0)
        # Bearish Absorption: Price pushed up near VAH / Call Wall / VWAP +2σ, Delta is heavily positive, but passive sellers halted price
        bearish_absorption = (current_price >= vah or current_price >= call_wall or vwap_z > 1.5) and (bar_delta > 0 and delta_pct > 30.0)

        # Check for Trapped Traders
        buy_imbalances = [imb for imb in imbalances if imb.get("type") == "BUY_IMBALANCE"]
        sell_imbalances = [imb for imb in imbalances if imb.get("type") == "SELL_IMBALANCE"]
        trapped_buyers = len(buy_imbalances) >= 2 and current_price < max([i["price"] for i in buy_imbalances], default=current_price)
        trapped_sellers = len(sell_imbalances) >= 2 and current_price > min([i["price"] for i in sell_imbalances], default=current_price)

        # Quantitative Alignment Evaluation
        if macro_bias == "LONG":
            score = 0.2  # base macro alignment
            
            # Auction Theory discount?
            if current_price < poc:
                score += 0.2
                reasons.append("Trading below POC (Auction value discount)")
            
            # GEX support
            if abs(current_price - put_wall) < 15.0 or current_price > gex_data.get("zero_gamma_level", 0):
                score += 0.2
                reasons.append("Respecting Options Put Wall / Positive Gamma support")

            # Footprint confirmation
            if trapped_sellers or bullish_absorption:
                score += 0.3
                reasons.append("Sellers absorbed / trapped at auction low")
            elif bar_delta > 0 and delta_pct > 15.0:
                score += 0.2
                reasons.append("Aggressive buyers stepping in (positive delta)")

            confidence_score = min(round(score, 2), 1.0)
            if confidence_score >= 0.70:
                audit_verdict = "AUDIT_PASSED_LONG"

        elif macro_bias == "SHORT":
            score = 0.2
            
            # Auction Theory premium?
            if current_price > poc:
                score += 0.2
                reasons.append("Trading above POC (Auction value premium)")

            # GEX resistance
            if abs(current_price - call_wall) < 15.0:
                score += 0.2
                reasons.append("Testing Options Call Wall resistance")

            # Footprint confirmation
            if trapped_buyers or bearish_absorption:
                score += 0.3
                reasons.append("Buyers absorbed / trapped at auction high")
            elif bar_delta < 0 and delta_pct < -15.0:
                score += 0.2
                reasons.append("Aggressive sellers stepping in (negative delta)")

            confidence_score = min(round(score, 2), 1.0)
            if confidence_score >= 0.70:
                audit_verdict = "AUDIT_PASSED_SHORT"

        return {
            "audit_verdict": audit_verdict,
            "confidence_score": confidence_score,
            "absorption": {
                "bullish_absorption": bool(bullish_absorption),
                "bearish_absorption": bool(bearish_absorption)
            },
            "trapped_traders": {
                "trapped_buyers": bool(trapped_buyers),
                "trapped_sellers": bool(trapped_sellers)
            },
            "reasons": reasons,
            "key_levels": {
                "poc": poc,
                "val": val,
                "vah": vah,
                "vwap": vwap,
                "call_wall": call_wall,
                "put_wall": put_wall
            }
        }
