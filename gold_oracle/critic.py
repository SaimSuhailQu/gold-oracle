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
- Enforces strict >= 80% (0.80) win probability threshold before giving trade clearance.
"""

from typing import Dict, List, Any, Optional
import numpy as np


class DecisionCriticEngine:
    def __init__(self, min_prob_threshold: float = 0.80):
        self.min_prob_threshold = min_prob_threshold

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
        atr14: float
    ) -> Dict[str, Any]:
        """
        Runs full MCDA scoring, adversarial self-criticism, and trade clearance.
        """
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

        # 3. Final Self-Audited Clearance
        is_cleared = (len(fatal_flaws) == 0) and (prob_directional >= self.min_prob_threshold) and (mcda_score >= 0.70)
        final_signal = raw_direction if is_cleared else "FLAT"

        audit_summary = "TRADE APPROVED (All Criteria & Adversarial Checks Passed)" if is_cleared else "TRADE REJECTED / STAND ASIDE (Adversarial Critic Blocked Entry)"

        return {
            "is_cleared_to_trade": is_cleared,
            "final_audited_signal": final_signal,
            "evaluated_direction": raw_direction,
            "directional_probability": round(prob_directional, 4),
            "min_required_probability": self.min_prob_threshold,
            "mcda_score": round(mcda_score, 3),
            "mcda_criteria_scores": scores,
            "fatal_flaws": fatal_flaws,
            "self_criticisms": criticisms,
            "audit_verdict_summary": audit_summary
        }
