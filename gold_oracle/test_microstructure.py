"""
Unified Test Runner for Order Flow, Auction Theory, VWAP & GEX Engine
Verifies all components and produces an end-to-end simulated audit.
"""

import numpy as np
import pandas as pd
from gold_oracle.vwap import VWAPEngine
from gold_oracle.profile import ProfileEngine
from gold_oracle.footprint import FootprintDeltaEngine
from gold_oracle.gex import GammaExposureEngine
from gold_oracle.heatmap import HeatmapEngine
from gold_oracle.audit import OrderFlowAuditEngine


def run_pipeline():
    print("=== Testing VWAP Engine ===")
    dates = pd.date_range("2026-10-01 09:00", periods=100, freq="15min")
    prices = 2650.0 + np.cumsum(np.random.normal(0, 1.5, 100))
    volumes = np.random.randint(50, 500, 100)
    df_bars = pd.DataFrame({
        "datetime": dates,
        "open": prices - 0.5,
        "high": prices + 1.2,
        "low": prices - 1.2,
        "close": prices,
        "volume": volumes
    })

    vwap_eng = VWAPEngine(anchor="D")
    df_vwap = vwap_eng.compute(df_bars)
    latest_vwap = df_vwap.iloc[-1]
    print(f"Latest VWAP: {latest_vwap['vwap']:.2f} | Upper 2-std: {latest_vwap['vwap_upper_2']:.2f} | Lower 2-std: {latest_vwap['vwap_lower_2']:.2f} | Z: {latest_vwap['vwap_zscore']:.2f}")

    print("\n=== Testing Volume & Market Profile Engine ===")
    prof_eng = ProfileEngine(tick_size=0.5)
    vp = prof_eng.compute_volume_profile(df_bars)
    tpo = prof_eng.compute_tpo_profile(df_bars)
    print(f"Volume POC: {vp['poc']} | VAH: {vp['vah']} | VAL: {vp['val']}")
    print(f"TPO POC: {tpo['tpo_poc']} | Profile Shape: {tpo['profile_type']}")

    print("\n=== Testing Footprint & Delta Engine ===")
    ticks_data = []
    base_p = 2650.0
    for _ in range(300):
        side = "buy" if np.random.rand() > 0.45 else "sell"
        p = base_p + np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
        v = np.random.randint(1, 25)
        ticks_data.append({"price": p, "volume": v, "side": side})
    ticks_df = pd.DataFrame(ticks_data)
    
    fp_eng = FootprintDeltaEngine(imbalance_ratio=3.0)
    fp = fp_eng.compute_candle_footprint(ticks_df, tick_size=0.5)
    print(f"Bar Delta: {fp['bar_delta']} ({fp['delta_pct']}%) | Total Vol: {fp['total_volume']}")
    print(f"Imbalances detected: {len(fp['imbalances'])}")

    print("\n=== Testing Gamma Exposure (GEX) Engine ===")
    strikes = [2600, 2620, 2640, 2650, 2660, 2680, 2700]
    calls = []
    puts = []
    for s in strikes:
        calls.append({"strike": s, "option_type": "call", "open_interest": np.random.randint(500, 5000), "implied_volatility": 0.16, "dte_days": 14})
        puts.append({"strike": s, "option_type": "put", "open_interest": np.random.randint(500, 5000), "implied_volatility": 0.17, "dte_days": 14})
    chain = pd.DataFrame(calls + puts)

    gex_eng = GammaExposureEngine()
    gex_res = gex_eng.calculate_gex_profile(chain, spot_price=2650.0)
    print(f"Regime: {gex_res['regime']} | Call Wall: {gex_res['call_wall']} | Put Wall: {gex_res['put_wall']}")

    print("\n=== Testing Order Book Heatmap Engine ===")
    heat_eng = HeatmapEngine()
    bids = [{"price": 2650.0 - i * 0.5, "volume": float(np.random.randint(10, 100))} for i in range(1, 10)]
    asks = [{"price": 2650.0 + i * 0.5, "volume": float(np.random.randint(10, 100))} for i in range(1, 10)]
    book = heat_eng.compute_book_metrics(bids, asks, spot_price=2650.0)
    print(f"Book Skew: {book['book_skew']} | Total Bid Depth: {book['total_bid_depth']} | Total Ask Depth: {book['total_ask_depth']}")

    print("\n=== Testing Order Flow Audit Engine ===")
    audit_eng = OrderFlowAuditEngine()
    audit_verdict = audit_eng.audit_setup(
        current_price=2648.5,
        vwap_data={"vwap": float(latest_vwap["vwap"]), "upper_2": float(latest_vwap["vwap_upper_2"]), "lower_2": float(latest_vwap["vwap_lower_2"]), "zscore": float(latest_vwap["vwap_zscore"])},
        profile_data=vp,
        gex_data=gex_res,
        footprint_data=fp,
        macro_bias="LONG"
    )
    print(f"Audit Result: {audit_verdict['audit_verdict']} (Confidence: {audit_verdict['confidence_score']})")
    print("Audit Reasons:", audit_verdict["reasons"])


if __name__ == "__main__":
    run_pipeline()
