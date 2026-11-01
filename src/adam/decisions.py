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


def consensus(decisions: list[Decision], weights: dict[Agent, float] | None = None) -> Consensus:
    if len({d.agent for d in decisions}) != len(decisions):
        raise ValueError("Only one decision per agent is allowed")
    if len({d.ticker for d in decisions}) > 1:
        raise ValueError("Decisions must belong to one ticker")
    available = [d for d in decisions if d.status == "completed" and d.reliable and not d.stale]
    if weights is not None and any(d.agent not in weights or not isfinite(weights[d.agent]) or weights[d.agent] <= 0 for d in available):
        raise ValueError("Each voting agent needs a positive finite weight")
    counts = Counter(d.signal for d in available)
    totals = {signal: sum((weights[d.agent] if weights is not None else 1) for d in available if d.signal == signal)
              for signal in ("BUY", "HOLD", "SELL")}
    highest = max(totals.values())
    winners = [signal for signal, total in totals.items() if total == highest]
    signal = None if not available else winners[0] if len(winners) == 1 else "HOLD"
    return Consensus(signal=signal, counts={s: counts[s] for s in totals},
                     disagreement=len(counts) > 1,
                     abstained=tuple(d.agent for d in decisions if d not in available))
