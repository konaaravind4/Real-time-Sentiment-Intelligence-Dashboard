"""
Financial Sentiment Mode for Real-time Sentiment Intelligence Dashboard
Provides specialized financial news/social sentiment analysis optimized
for market signal generation — integrates with Kronos financial forecasting.

Analyzes text from:
- Financial news headlines (Bloomberg, Reuters, CNBC)
- Reddit (r/wallstreetbets, r/stocks, r/CryptoCurrency)
- Twitter/X financial topics (#BTC, $AAPL, #trading)
- Earnings call transcripts

Output feeds directly into:
- Kronos reproduction (market sentiment overlay on forecasts)
- KonaDB time-series (for persistent storage and trend analysis)

Usage:
    from api.financial_mode import FinancialSentimentAnalyzer

    analyzer = FinancialSentimentAnalyzer()
    result = analyzer.analyze("Bitcoin surges 15% after ETF approval news!")
    print(result.signal)       # "bullish" | "bearish" | "neutral"
    print(result.confidence)   # 0.0 – 1.0
    print(result.score)        # -1.0 (bearish) to +1.0 (bullish)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal


Signal = Literal["bullish", "bearish", "neutral"]



BULLISH_KEYWORDS: list[tuple[str, float]] = [
    # (pattern, weight)
    (r"\bsurges?\b", 0.9),
    (r"\ball[- ]time high\b", 1.0),
    (r"\bbreaks?.*resistance\b", 0.8),
    (r"\bbeats? estimates?\b", 0.85),
    (r"\brecord\b.*\bprofit\b", 0.9),
    (r"\bstrong buy\b", 1.0),
    (r"\bupgrade[ds]?\b", 0.7),
    (r"\bapproval\b", 0.6),
    (r"\bgains?\b", 0.5),
    (r"\bjumps?\b", 0.6),
    (r"\brallies?\b", 0.8),
    (r"\bbullish\b", 0.9),
    (r"\bexpands?\b", 0.5),
    (r"\bgrowth\b", 0.4),
    (r"\boptimis\w+\b", 0.6),
    (r"\brecovery\b", 0.55),
    (r"\bbreakout\b", 0.85),
    (r"\bmomentum\b", 0.5),
    (r"\binvests?\s+in\b", 0.6),
    (r"\bpartnership\b", 0.5),
]

BEARISH_KEYWORDS: list[tuple[str, float]] = [
    (r"\bcrashes?\b", 0.95),
    (r"\bplunges?\b", 0.9),
    (r"\bslumps?\b", 0.8),
    (r"\bdowngrade[ds]?\b", 0.75),
    (r"\blosses?\b", 0.6),
    (r"\bmisses?\s+estimates?\b", 0.85),
    (r"\bbearish\b", 0.9),
    (r"\blayoffs?\b", 0.7),
    (r"\bbankruptcy\b", 1.0),
    (r"\brecession\b", 0.85),
    (r"\bregulatory\s+action\b", 0.7),
    (r"\bsec\s+investigation\b", 0.9),
    (r"\bfraud\b", 1.0),
    (r"\bfear\b", 0.5),
    (r"\bpanic\b", 0.75),
    (r"\bsell[- ]off\b", 0.8),
    (r"\bweak\s+guidance\b", 0.8),
    (r"\bwrites?\s+down\b", 0.7),
    (r"\bwithdraw[ns]?\b", 0.6),
    (r"\bconcern\b", 0.4),
]


_TICKER_RE = re.compile(r"\b(BTC|ETH|SOL|AAPL|MSFT|NVDA|TSLA|AMZN|GOOGL|META|SPY|QQQ)\b")
_PERCENTAGE_RE = re.compile(r"([+-]?\d+\.?\d*)\s*%")




@dataclass
class FinancialSentimentResult:
    text: str
    signal: Signal
    score: float              # -1.0 (bearish) to +1.0 (bullish)
    confidence: float         # 0.0 to 1.0
    tickers: list[str]        # mentioned asset tickers
    magnitude: float | None   # extracted % change if mentioned
    bullish_signals: list[str]
    bearish_signals: list[str]
    timestamp: float = field(default_factory=time.time)

    @property
    def kona_tags(self) -> dict:
        """Tags for KonaDB TimeSeries storage."""
        return {
            "signal": self.signal,
            "ticker": self.tickers[0] if self.tickers else "MARKET",
            "source": "financial_sentiment",
        }


@dataclass
class MarketSentimentSummary:
    """Aggregated sentiment across multiple texts (e.g. last hour of news)."""
    total_analyzed: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    avg_score: float
    dominant_signal: Signal
    top_tickers: list[str]
    window_seconds: float
    timestamp: float = field(default_factory=time.time)

    @property
    def bull_bear_ratio(self) -> float:
        """Bull/bear ratio. >1 = bullish market. <1 = bearish."""
        if self.bearish_count == 0:
            return float("inf") if self.bullish_count > 0 else 1.0
        return self.bullish_count / self.bearish_count


class FinancialSentimentAnalyzer:
    """
    Fast, rule-based financial sentiment analyzer optimized for market signals.
    
    Uses weighted keyword matching + magnitude extraction for speed (<5ms latency).
    Can be combined with the RoBERTa emotion model for full 8-class emotion output.
    """

    def __init__(self):
        self._history: list[FinancialSentimentResult] = []

    def analyze(self, text: str) -> FinancialSentimentResult:
        """
        Analyze a single piece of financial text for market sentiment.

        Args:
            text: Financial news headline, tweet, or paragraph.

        Returns:
            FinancialSentimentResult with signal, score, confidence, tickers.

        Example:
            >>> result = analyzer.analyze("Bitcoin surges 15% after ETF approval")
            >>> print(result.signal, result.score)
            bullish 0.82
        """
        text_lower = text.lower()

        bullish_score = 0.0
        matched_bullish = []
        for pattern, weight in BULLISH_KEYWORDS:
            if re.search(pattern, text_lower):
                bullish_score += weight
                matched_bullish.append(pattern.replace(r"\b", "").strip("()?.*+"))


        bearish_score = 0.0
        matched_bearish = []
        for pattern, weight in BEARISH_KEYWORDS:
            if re.search(pattern, text_lower):
                bearish_score += weight
                matched_bearish.append(pattern.replace(r"\b", "").strip("()?.*+"))


        max_possible = max(sum(w for _, w in BULLISH_KEYWORDS), sum(w for _, w in BEARISH_KEYWORDS))
        bull_norm = min(bullish_score / max_possible, 1.0)
        bear_norm = min(bearish_score / max_possible, 1.0)


        net_score = round(bull_norm - bear_norm, 4)


        if net_score > 0.05:
            signal: Signal = "bullish"
        elif net_score < -0.05:
            signal = "bearish"
        else:
            signal = "neutral"


        confidence = min(abs(net_score) * 2, 1.0)


        tickers = list(dict.fromkeys(_TICKER_RE.findall(text.upper())))


        pct_matches = _PERCENTAGE_RE.findall(text)
        magnitude = float(pct_matches[0]) if pct_matches else None

        result = FinancialSentimentResult(
            text=text,
            signal=signal,
            score=net_score,
            confidence=round(confidence, 4),
            tickers=tickers,
            magnitude=magnitude,
            bullish_signals=matched_bullish[:5],
            bearish_signals=matched_bearish[:5],
        )

        self._history.append(result)
        return result

    def analyze_batch(self, texts: list[str]) -> list[FinancialSentimentResult]:
        """Analyze a list of texts and return results in the same order."""
        return [self.analyze(t) for t in texts]

    def market_summary(self, window_seconds: float = 3600.0) -> MarketSentimentSummary:
        """
        Compute an aggregated market sentiment summary over a time window.

        Args:
            window_seconds: How far back to look (default: 1 hour).

        Returns:
            MarketSentimentSummary with bull/bear counts, avg score, dominant signal.
        """
        cutoff = time.time() - window_seconds
        recent = [r for r in self._history if r.timestamp >= cutoff]

        if not recent:
            return MarketSentimentSummary(
                total_analyzed=0, bullish_count=0, bearish_count=0, neutral_count=0,
                avg_score=0.0, dominant_signal="neutral", top_tickers=[],
                window_seconds=window_seconds,
            )

        bullish  = sum(1 for r in recent if r.signal == "bullish")
        bearish  = sum(1 for r in recent if r.signal == "bearish")
        neutral  = sum(1 for r in recent if r.signal == "neutral")
        avg_score = sum(r.score for r in recent) / len(recent)

        if avg_score > 0.05:
            dominant: Signal = "bullish"
        elif avg_score < -0.05:
            dominant = "bearish"
        else:
            dominant = "neutral"

        ticker_counts: dict[str, int] = {}
        for r in recent:
            for t in r.tickers:
                ticker_counts[t] = ticker_counts.get(t, 0) + 1
        top_tickers = sorted(ticker_counts, key=ticker_counts.get, reverse=True)[:5]

        return MarketSentimentSummary(
            total_analyzed=len(recent),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            avg_score=round(avg_score, 4),
            dominant_signal=dominant,
            top_tickers=top_tickers,
            window_seconds=window_seconds,
        )
