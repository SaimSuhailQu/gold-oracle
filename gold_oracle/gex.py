"""
Options Gamma Exposure (GEX) Engine
Calculates:
1. Per-Strike Gamma Exposure (GEX) for Market Makers:
   - Call GEX = Spot * Spot Gamma * Call Open Interest * Contract Multiplier * 100
   - Put GEX = -Spot * Spot Gamma * Put Open Interest * Contract Multiplier * 100
   - Net GEX = Call GEX + Put GEX
2. Zero Gamma Flip Level:
   - The inflection price where market flips from Positive GEX (mean reversion)
     to Negative GEX (accelerated momentum / dealer short gamma hedging).
3. Major Options Walls:
   - Call Resistance Wall (highest Call Open Interest / Gamma concentration)
   - Put Support Wall (highest Put Open Interest / Gamma concentration)
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class GammaExposureEngine:
    def __init__(self, contract_multiplier: float = 100.0):
        self.contract_multiplier = contract_multiplier

    def compute_bs_gamma(
        self,
        spot: float,
        strike: float,
        time_to_exp: float,
        iv: float,
        risk_free_rate: float = 0.04
    ) -> float:
        """
        Black-Scholes analytical gamma calculation.
        """
        if time_to_exp <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
            return 0.0

        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * iv ** 2) * time_to_exp) / (iv * np.sqrt(time_to_exp))
        # Standard normal PDF
        phi_d1 = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1 ** 2)
        gamma = phi_d1 / (spot * iv * np.sqrt(time_to_exp))
        return float(gamma)

    def calculate_gex_profile(
        self,
        options_chain: pd.DataFrame,
        spot_price: float,
        risk_free_rate: float = 0.04
    ) -> Dict[str, Any]:
        """
        Takes options chain with columns:
        ['strike', 'option_type' (call/put), 'open_interest', 'implied_volatility', 'dte_days']
        Returns GEX structure, Zero-Gamma flip point, Call Wall, and Put Wall.
        """
        if options_chain.empty:
            return {}

        df = options_chain.copy()
        df["t_years"] = df["dte_days"].clip(lower=0.5) / 365.0

        # Calculate gamma per row
        gammas = []
        for _, row in df.iterrows():
            g = self.compute_bs_gamma(
                spot=spot_price,
                strike=row["strike"],
                time_to_exp=row["t_years"],
                iv=row["implied_volatility"],
                risk_free_rate=risk_free_rate
            )
            gammas.append(g)
        df["gamma"] = gammas

        # Market maker positioning assumption:
        # Long Call OI -> MM is short call -> negative gamma if unhedged, but net market dealer long gamma
        # Standard dealer convention:
        # Call GEX = + (Spot * Gamma * Call_OI * Multiplier)
        # Put GEX = - (Spot * Gamma * Put_OI * Multiplier)
        df["is_call"] = df["option_type"].str.lower() == "call"
        
        df["dollar_gamma"] = np.where(
            df["is_call"],
            spot_price * df["gamma"] * df["open_interest"] * self.contract_multiplier,
            -spot_price * df["gamma"] * df["open_interest"] * self.contract_multiplier
        )

        # Aggregate by strike
        by_strike = df.groupby("strike").agg(
            call_oi=pd.NamedAgg(column="open_interest", aggfunc=lambda s: s[df.loc[s.index, "is_call"]].sum()),
            put_oi=pd.NamedAgg(column="open_interest", aggfunc=lambda s: s[~df.loc[s.index, "is_call"]].sum()),
            net_gex=pd.NamedAgg(column="dollar_gamma", aggfunc="sum")
        ).sort_index()

        total_net_gex = by_strike["net_gex"].sum()
        regime = "POSITIVE_GEX_PINNED_MEAN_REVERSION" if total_net_gex > 0 else "NEGATIVE_GEX_ACCELERATED_VOLATILITY"

        # Call Wall (strike with highest call OI) & Put Wall (highest put OI)
        call_wall = by_strike["call_oi"].idxmax() if not by_strike["call_oi"].empty else spot_price
        put_wall = by_strike["put_oi"].idxmax() if not by_strike["put_oi"].empty else spot_price

        # Zero Gamma Flip Point (strike where cumulative GEX crosses zero)
        by_strike["cum_gex"] = by_strike["net_gex"].cumsum()
        zero_flip = spot_price
        crosses = by_strike[by_strike["net_gex"] * by_strike["net_gex"].shift(1) < 0]
        if not crosses.empty:
            zero_flip = float(crosses.index[0])

        return {
            "spot_price": spot_price,
            "total_net_gex": float(total_net_gex),
            "regime": regime,
            "zero_gamma_level": float(zero_flip),
            "call_wall": float(call_wall),
            "put_wall": float(put_wall),
            "gex_by_strike": {
                float(k): {
                    "net_gex": float(v["net_gex"]),
                    "call_oi": int(v["call_oi"]),
                    "put_oi": int(v["put_oi"])
                }
                for k, v in by_strike.iterrows()
            }
        }
