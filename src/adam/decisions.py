from collections import Counter
from math import isclose, isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Signal = Literal["BUY", "HOLD", "SELL"]
Agent = Literal["market", "forecast", "news", "technical", "fundamentals"]


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1)
    agent: Agent
    signal: Signal
    probabilities: dict[Signal, float]
    summary: str
    status: Literal["completed", "failed", "pending"]
    reliable: bool = True
    stale: bool = False

    @model_validator(mode="after")
    def validate_scores(self):
        scores = self.probabilities
        if (set(scores) != {"BUY", "HOLD", "SELL"}
                or any(not isfinite(v) or v < 0 or v > 1 for v in scores.values())
                or not isclose(sum(scores.values()), 1.0, abs_tol=1e-6)):
            raise ValueError("Scores must contain BUY, HOLD and SELL and sum to one")
        return self


class Consensus(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal: Signal | None
    counts: dict[Signal, int]
    disagreement: bool
    abstained: tuple[Agent, ...]


def consensus(decisions: list[Decision], weights: dict[Agent, float] | None = None, *, margin: float = 0.0) -> Consensus:
    """Directional soft vote: weighted mean probabilities decide between BUY and SELL.

    Categorical HOLD votes no longer block the call; the consensus goes with whichever
    direction the agents lean toward on average. HOLD is returned only when the lean is
    balanced within margin or nobody votes. Counts and disagreement still report the
    agents' categorical signals for display.
    """
    if not isfinite(margin) or isinstance(margin, bool) or not 0 <= margin < 1:
        raise ValueError("margin must be between 0 and 1")
    if len({d.agent for d in decisions}) != len(decisions):
        raise ValueError("Only one decision per agent is allowed")
    if len({d.ticker for d in decisions}) > 1:
        raise ValueError("Decisions must belong to one ticker")
    available = [d for d in decisions if d.status == "completed" and d.reliable and not d.stale]
    if weights is not None and any(d.agent not in weights or not isfinite(weights[d.agent]) or weights[d.agent] <= 0 for d in available):
        raise ValueError("Each voting agent needs a positive finite weight")
    counts = Counter(d.signal for d in available)
    weight_total = sum((weights[d.agent] if weights is not None else 1) for d in available)
    means = {signal: sum((weights[d.agent] if weights is not None else 1) * d.probabilities[signal]
                         for d in available) / weight_total
             for signal in ("BUY", "HOLD", "SELL")} if available else None
    if means is None:
        signal = None
    elif means["BUY"] - means["SELL"] > margin:
        signal = "BUY"
    elif means["SELL"] - means["BUY"] > margin:
        signal = "SELL"
    else:
        signal = "HOLD"
    return Consensus(signal=signal, counts={s: counts[s] for s in ("BUY", "HOLD", "SELL")},
                     disagreement=len(counts) > 1,
                     abstained=tuple(d.agent for d in decisions if d not in available))
