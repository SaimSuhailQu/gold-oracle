"""Live signal: fit on all history, emit today's P(gold up) + factor attribution.

Writes signals/gold_signal.json — the MT5 EA on Exness reads this file.
Run on a schedule (e.g. every 6h / daily) on the same machine as MT5, with
SIGNAL_PATH pointing at <MT5>/MQL5/Files/gold_signal.json, or copy it over.

Schema:
  { timestamp, symbol, horizon_days, p_up, signal, confidence,
    gold_price, factors: {name: contribution}, model_weights: {...} }
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from .data import latest
from .factors import FEATURES, HORIZON, build_features
from .model import GoldProbabilityModel

LONG_T = float(os.environ.get("GOLD_LONG_T", "0.80"))
SHORT_T = float(os.environ.get("GOLD_SHORT_T", "0.20"))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNAL_PATH = os.environ.get(
    "GOLD_SIGNAL_PATH", os.path.join(BASE, "signals", "gold_signal.json")
)


def generate(df: pd.DataFrame | None = None, use_cache: bool = True) -> dict:
    if df is None:
        try:
            from .data import download
            df = download(use_cache=use_cache)
        except Exception:
            df = latest()
    f = build_features(df)
    X, y = f[FEATURES], f["target"]
    model = GoldProbabilityModel().fit(X, y)

    from .public_apis_ai import PublicApisAIConnector
    from .ensemble_ml import EnsembleGoldAI
    from .youtube_memory import YouTubeMemoryEngine

    # 1. Fetch free public macro headlines and analyze sentiment via Public APIs connector
    ai_connector = PublicApisAIConnector()
    headlines = ai_connector.fetch_free_macro_news()
    sentiment_data = ai_connector.analyze_financial_sentiment(headlines)
    sentiment_score = sentiment_data.get("sentiment_score", 0.0)

    # 1b. YouTube Video Memory Integration
    yt_memory_eng = YouTubeMemoryEngine()
    yt_context = yt_memory_eng.get_memory_context()
    yt_sentiment = yt_context.get("collective_sentiment", 0.0)
    # Blend news sentiment with learned YouTube analyst memory if channels exist
    if yt_context.get("channels_tracked", 0) > 0:
        sentiment_score = 0.60 * sentiment_score + 0.40 * yt_sentiment

    row = f.iloc[[-1]]
    p_up = float(model.predict_proba(row[FEATURES])[0])

    # 2. Fit non-linear ensemble ML model
    ensemble_ai = EnsembleGoldAI().fit(X, y)
    ensemble_res = ensemble_ai.predict_proba_ensemble(row[FEATURES], baseline_prob=p_up, nlp_sentiment=sentiment_score)
    p_up_final = ensemble_res["p_ensemble_up"]

    raw_signal = "LONG" if p_up_final >= LONG_T else ("SHORT" if p_up_final <= SHORT_T else "FLAT")

    # Technical trend validation: 50-day and 200-day moving averages
    gold_last = float(df["gold"].iloc[-1])
    ma50_val = float(f["ma50"].iloc[-1]) if "ma50" in f.columns and pd.notna(f["ma50"].iloc[-1]) else gold_last
    ma200_val = float(f["ma200"].iloc[-1]) if "ma200" in f.columns and pd.notna(f["ma200"].iloc[-1]) else gold_last
    atr_val = float(f["atr14"].iloc[-1]) if "atr14" in f.columns and pd.notna(f["atr14"].iloc[-1]) else 30.0

    bull_trend = (gold_last >= ma50_val) or (ma50_val >= ma200_val)
    bear_trend = (gold_last <= ma50_val) or (ma50_val <= ma200_val)

    # Filter signal with trend alignment (blocks counter-trend trades unless probability is extreme >= 0.75)
    if raw_signal == "LONG" and not bull_trend and p_up_final < 0.75:
        signal = "FLAT"
    elif raw_signal == "SHORT" and not bear_trend and (1.0 - p_up_final) < 0.75:
        signal = "FLAT"
    else:
        signal = raw_signal

    contrib = model.contributions(f[FEATURES].iloc[-1])

    # Base-model transparency: live z-scores, composite, calibration
    zscores = {c: round(float(f[c].iloc[-1]), 4) for c in FEATURES}
    composite = round(float(sum(contrib.values())), 4)

    # Run Microstructure, Profile & Order Flow Audit
    from .vwap import VWAPEngine
    from .profile import ProfileEngine
    from .audit import OrderFlowAuditEngine

    # Compute VWAP & Dispersion Bands on gold prices
    gold_series = df[["gold"]].rename(columns={"gold": "close"})
    gold_series["high"] = gold_series["close"] * 1.002
    gold_series["low"] = gold_series["close"] * 0.998
    gold_series["volume"] = 1000.0
    gold_series["datetime"] = df.index if pd.api.types.is_datetime64_any_dtype(df.index) else pd.to_datetime(df.index)
    
    vwap_res = VWAPEngine(anchor="D").compute(gold_series).iloc[-1]
    prof_res = ProfileEngine(tick_size=0.5).compute_volume_profile(gold_series)
    
    # Run Order Flow Audit
    audit_eng = OrderFlowAuditEngine()
    audit_res = audit_eng.audit_setup(
        current_price=gold_last,
        vwap_data={
            "vwap": float(vwap_res["vwap"]),
            "upper_2": float(vwap_res["vwap_upper_2"]),
            "lower_2": float(vwap_res["vwap_lower_2"]),
            "zscore": float(vwap_res["vwap_zscore"])
        },
        profile_data=prof_res,
        gex_data={
            "regime": "POSITIVE_GEX_PINNED_MEAN_REVERSION",
            "zero_gamma_level": gold_last - 25.0,
            "call_wall": gold_last + 50.0,
            "put_wall": gold_last - 50.0
        },
        footprint_data={"bar_delta": 0.0, "delta_pct": 0.0, "imbalances": []},
        macro_bias=signal
    )

    # 3. Adversarial Multi-Criteria & Self-Criticism Engine
    from .critic import DecisionCriticEngine
    critic_eng = DecisionCriticEngine(min_prob_threshold=0.80)
    critic_report = critic_eng.evaluate_multi_criteria_with_criticism(
        p_up=p_up_final,
        factors=contrib,
        ensemble_ml=ensemble_res,
        ai_sentiment=sentiment_data,
        microstructure={
            "vwap": float(vwap_res["vwap"]),
            "vwap_zscore": float(vwap_res["vwap_zscore"]),
            "poc": float(prof_res.get("poc", gold_last)),
            "vah": float(prof_res.get("vah", gold_last)),
            "val": float(prof_res.get("val", gold_last))
        },
        gold_price=gold_last,
        ma50=ma50_val,
        ma200=ma200_val,
        atr14=atr_val
    )

    final_signal = critic_report["final_audited_signal"]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_epoch": int(datetime.now(timezone.utc).timestamp()),
        "symbol": "XAUUSD",
        "horizon_days": HORIZON,
        "p_up": round(p_up_final, 4),
        "p_down": round(1 - p_up_final, 4),
        "signal": final_signal,
        "raw_signal": raw_signal,
        "long_t": LONG_T,
        "short_t": SHORT_T,
        "confidence": round(abs(p_up_final - 0.5) * 2, 4),
        "gold_price": round(gold_last, 2),
        "composite": composite,
        "calibration": {"a": round(model.cal_a_, 4), "b": round(model.cal_b_, 4)},
        "factor_zscores": zscores,
        "decision_critic": critic_report,
        "ensemble_ml": ensemble_res,
        "ai_sentiment": sentiment_data,
        "youtube_memory": yt_context,
        "ma50": round(ma50_val, 2),
        "ma200": round(ma200_val, 2),
        "atr14": round(atr_val, 2),
        "sl_dollars": round(max(20.0, 2.0 * atr_val), 2),
        "tp_dollars": round(max(40.0, 3.5 * atr_val), 2),
        "factors": {k: round(v, 4) for k, v in contrib.items()},
        "model_weights": {k: round(v, 4) for k, v in model.weights_.items()},
        "microstructure": {
            "vwap": round(float(vwap_res["vwap"]), 2),
            "vwap_zscore": round(float(vwap_res["vwap_zscore"]), 2),
            "poc": round(prof_res.get("poc", gold_last), 2),
            "vah": round(prof_res.get("vah", gold_last), 2),
            "val": round(prof_res.get("val", gold_last), 2),
            "audit_verdict": audit_res["audit_verdict"],
            "audit_confidence": audit_res["confidence_score"]
        }
    }


HISTORY_PATH = os.path.join(BASE, "signals", "history.jsonl")


def append_history(sig: dict) -> None:
    """One line per signal run — powers the GUI's history charts."""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    rec = {"ts": sig["timestamp_epoch"], "p_up": sig["p_up"],
           "signal": sig["signal"], "gold_price": sig["gold_price"],
           "zscores": sig["factor_zscores"]}
    with open(HISTORY_PATH, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def main() -> None:
    sig = generate()
    os.makedirs(os.path.dirname(SIGNAL_PATH), exist_ok=True)
    with open(SIGNAL_PATH, "w") as fh:
        json.dump(sig, fh, indent=2)
    append_history(sig)
    print(json.dumps(sig, indent=2))
    print(f"[signal] wrote {SIGNAL_PATH} + history")


if __name__ == "__main__":
    main()
