"""
Multi-Criteria Decision Analysis (MCDA) & Adversarial Self-Criticism Engine
Evaluates trading setups across independent orthogonal criteria:
1. Macro Fundamentals (Rates, Yields, Dollar, Fed Stance, Reserves)
2. Machine Learning Non-Linear Model Consensus
3. Free Public AI Live News NLP Sentiment
4. Auction Market Theory (Value Area, POC, Distribution)
5. Microstructure & Order Flow (Footprint Delta, Imbalance, VWAP Z-Score)
6. Options Gamma Exposure (GEX Regime, Call/Put Walls)

Adversarial Self-Criticism (The 'Devil's Advocate' Engine):
- Actively seeks reasons why the trade could fail
- Tests for trap risk, fakeouts, volatility compression, and macro regime contradictions
- Enforces tiered clearance: >= 70% win probability + MCDA >= 0.70 -> FULL size;
  >= 60% + MCDA >= 0.50 -> QUARTER-size exploratory; else FLAT.
  (Thresholds recalibrated 2026-10-10 from walk-forward evidence: the
  ensemble's blended probability maxes at ~0.76, so the old 80%/65% bars
  were unreachable — 5 trades in 7 years. 70%/60% matches what the model
  can actually produce.)
"""

from typing import Dict, List, Any, Optional
import numpy as np


class DecisionCriticEngine:
    """Tiered sizing critic.

    - FULL (1.0x):  directional P >= full_prob (0.70) and MCDA >= full_mcda (0.70),
      no fatal flaws — full-sized position.
    - QUARTER (0.25x): directional P >= q_prob (0.60) and MCDA >= q_mcda (0.50),
      no fatal flaws — exploratory position that gathers out-of-sample data
      without heavy risk.
    - FLAT (0.0x): anything else, or any fatal flaw.
    Thresholds overridable via constructor (env GOLD_FULL_P / GOLD_Q_P).
    Recalibrated 2026-10-10 from walk-forward evidence (old 0.80/0.65 bars
    were unreachable: ensemble max ~0.76, 5 trades/7yr).
    """

    def __init__(self, min_prob_threshold: float = 0.70,
                 full_mcda: float = 0.70,
                 q_prob: float = 0.60, q_mcda: float = 0.50):
        import os
        self.min_prob_threshold = float(os.environ.get("GOLD_FULL_P", min_prob_threshold))
        self.full_mcda = full_mcda
        self.q_prob = float(os.environ.get("GOLD_Q_P", q_prob))
        self.q_mcda = q_mcda

    def evaluate_multi_criteria_with_criticism(
        self,
        p_up: float,
        factors: Dict[str, float],
        ensemble_ml: Dict[str, float],
        ai_sentiment: Dict[str, Any],
        microstructure: Dict[str, Any],
        gold_price: float,
        ma50: float,
        ma200: float,
        atr14: float,
        technicals: Optional[Dict[str, Any]] = None,
        news_blocked: bool = False,
        news_event: Optional[str] = None,
        rr_ratio: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Runs full MCDA scoring, adversarial self-criticism, and trade clearance.
        """
        technicals = technicals or {}
        p_down = 1.0 - p_up
        raw_direction = "LONG" if p_up >= 0.50 else "SHORT"
        prob_directional = p_up if raw_direction == "LONG" else p_down

        # 1. Multi-Criteria Scoring (0.0 to 1.0 each)
        scores = {}

        # Criterion A: Macro Consistency
        bullish_macro = sum(1 for v in factors.values() if v > 0)
        total_macro = len(factors) if factors else 1
        macro_ratio = bullish_macro / total_macro
        scores["macro_alignment"] = macro_ratio if raw_direction == "LONG" else (1.0 - macro_ratio)

        # Criterion B: ML Consensus
        p_linear = ensemble_ml.get("p_linear_logistic", 0.5)
        p_boosted = ensemble_ml.get("p_boosted_ml", 0.5)
        ml_agreement = 1.0 - abs(p_linear - p_boosted)
        scores["ml_consensus"] = round(ml_agreement, 3)

        # Criterion C: AI News Sentiment
        sent_score = ai_sentiment.get("sentiment_score", 0.0)
        scores["news_sentiment_alignment"] = max(0.0, (sent_score if raw_direction == "LONG" else -sent_score) + 0.5)

        # Criterion D: Technical Trend Structure
        if raw_direction == "LONG":
            trend_score = 1.0 if (gold_price >= ma50 and ma50 >= ma200) else (0.5 if gold_price >= ma50 else 0.1)
        else:
            trend_score = 1.0 if (gold_price <= ma50 and ma50 <= ma200) else (0.5 if gold_price <= ma50 else 0.1)
        scores["technical_trend"] = trend_score

        # Criterion E: Auction Profile & VWAP Location
        poc = microstructure.get("poc", gold_price)
        val = microstructure.get("val", gold_price)
        vah = microstructure.get("vah", gold_price)
        vwap_z = microstructure.get("vwap_zscore", 0.0)

        if raw_direction == "LONG":
            auction_score = 0.8 if gold_price <= poc else 0.4
            if vwap_z < -1.5:  # oversold at lower band
                auction_score += 0.2
        else:
            auction_score = 0.8 if gold_price >= poc else 0.4
            if vwap_z > 1.5:   # overbought at upper band
                auction_score += 0.2
        scores["auction_location"] = min(1.0, round(auction_score, 2))

        # Composite Multi-Criteria Score
        weights = {
            "macro_alignment": 0.25,
            "ml_consensus": 0.20,
            "news_sentiment_alignment": 0.15,
            "technical_trend": 0.20,
            "auction_location": 0.20
        }
        mcda_score = sum(scores[k] * weights[k] for k in scores)

        # 2. Adversarial Self-Criticism (The 'Devil's Advocate')
        criticisms: List[str] = []
        fatal_flaws: List[str] = []

        # Critic 1: Check Probability Threshold
        if prob_directional < self.min_prob_threshold:
            fatal_flaws.append(
                f"INSUFFICIENT PROBABILITY: Calculated win probability is {prob_directional*100:.1f}%, which fails the mandatory >= {self.min_prob_threshold*100:.0f}% threshold."
            )

        # Critic 2: Macro vs Momentum Contradiction
        if raw_direction == "LONG" and factors.get("nom_yield", 0) < -0.30:
            criticisms.append(
                "CRITIC WARNING: Nominal 10Y Yields are surging strongly, acting as a powerful macro headwind against gold."
            )
        if raw_direction == "LONG" and factors.get("dollar", 0) < -0.20:
            criticisms.append(
                "CRITIC WARNING: US Dollar index (DXY/UUP) is in an upward impulse, creating currency drag."
            )
        if raw_direction == "SHORT" and factors.get("cb_regime", 0) > 0.0:
            criticisms.append(
                "CRITIC WARNING: Structural central-bank reserve accumulation creates sticky long-term price support on dips."
            )

        # Critic 3: Trend Disconnect
        if raw_direction == "LONG" and gold_price < ma50:
            criticisms.append("CRITIC WARNING: Counter-trend trade: Gold is trading beneath its 50-day moving average.")
        elif raw_direction == "SHORT" and gold_price > ma50:
            criticisms.append("CRITIC WARNING: Counter-trend trade: Gold is trading above its 50-day moving average.")

        # Critic 4: Microstructure Overextension
        if raw_direction == "LONG" and vwap_z > 2.0:
            fatal_flaws.append("FATAL MICROSTRUCTURE FLAW: Price is overextended at +2.0 standard deviations above VWAP; high risk of mean reversion.")
        elif raw_direction == "SHORT" and vwap_z < -2.0:
            fatal_flaws.append("FATAL MICROSTRUCTURE FLAW: Price is overextended at -2.0 standard deviations below VWAP; high risk of short squeeze.")

        # Critic 5: Model Disagreement
        if abs(p_linear - p_boosted) > 0.25:
            criticisms.append(f"CRITIC WARNING: Model divergence detected (Linear Logistic={p_linear:.2f} vs Boosted ML={p_boosted:.2f}). Market regime may be transitioning.")

        # Critic 6: Economic news blackout (fatal — never enter into NFP/CPI/FOMC)
        if news_blocked:
            fatal_flaws.append(
                f"NEWS BLACKOUT: {news_event or 'high-impact event'} within ±30 min — no new entries into event volatility."
            )

        # Critic 7: Session filter (trade London/New York only)
        sess = (technicals.get("session") or {}).get("session", "UNKNOWN")
        if not (technicals.get("session") or {}).get("trade_ok", True):
            fatal_flaws.append(
                f"SESSION FILTER: current session is {sess} — entries only during London/New York."
            )

        # Critic 8: Risk-to-reward filter (need ≥ 1.5:1)
        if rr_ratio and rr_ratio < 1.5:
            fatal_flaws.append(
                f"POOR R:R — planned reward:risk is {rr_ratio:.2f}:1, below the 1.5:1 minimum."
            )

        # Critic 9: Technical alignment (multi-condition entry confirmation)
        tech_score = float(technicals.get("tech_score") or 0.0)
        if raw_direction == "LONG" and tech_score < -0.30:
            criticisms.append(
                f"CRITIC WARNING: technical stack is bearish (tech_score={tech_score}) against a LONG — "
                f"MTF={technicals.get('mtf_score')}, RSI={technicals.get('rsi14')}, structure={technicals.get('market_structure', {}).get('state')}."
            )
        elif raw_direction == "SHORT" and tech_score > 0.30:
            criticisms.append(
                f"CRITIC WARNING: technical stack is bullish (tech_score={tech_score}) against a SHORT."
            )

        # 3. Tiered clearance: FULL / QUARTER / FLAT
        fatal = len(fatal_flaws) > 0
        if not fatal and prob_directional >= self.min_prob_threshold and mcda_score >= self.full_mcda:
            size_tier, size_mult = "FULL", 1.0
            final_signal = raw_direction
            audit_summary = "TRADE APPROVED — FULL SIZE (all criteria & adversarial checks passed)"
        elif not fatal and prob_directional >= self.q_prob and mcda_score >= self.q_mcda:
            size_tier, size_mult = "QUARTER", 0.25
            final_signal = raw_direction
            audit_summary = ("TRADE APPROVED — QUARTER SIZE (exploratory: gathers out-of-sample "
                             "data at 0.25x risk)")
        else:
            size_tier, size_mult = "FLAT", 0.0
            final_signal = "FLAT"
            if fatal:
                audit_summary = "TRADE REJECTED / STAND ASIDE (adversarial critic found fatal flaw)"
            else:
                audit_summary = ("TRADE REJECTED / STAND ASIDE (below quarter-tier bars: "
                                 f"P={prob_directional:.0%} < {self.q_prob:.0%} or MCDA={mcda_score:.2f} < {self.q_mcda})")

        return {
            "is_cleared_to_trade": size_mult > 0,
            "size_tier": size_tier,
            "size_mult": size_mult,
            "final_audited_signal": final_signal,
            "evaluated_direction": raw_direction,
            "directional_probability": round(prob_directional, 4),
            "min_required_probability": self.min_prob_threshold,
            "quarter_probability": self.q_prob,
            "mcda_score": round(mcda_score, 3),
            "mcda_criteria_scores": scores,
            "fatal_flaws": fatal_flaws,
            "self_criticisms": criticisms,
            "audit_verdict_summary": audit_summary
        }
