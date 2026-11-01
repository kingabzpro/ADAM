from datetime import datetime, timezone
from math import isfinite
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat

from adam.decisions import Consensus

NonNegative = Annotated[FiniteFloat, Field(ge=0)]
Fraction = Annotated[FiniteFloat, Field(ge=0, le=1)]


class RiskEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    market_time: AwareDatetime | None = None
    volatility: NonNegative | None = None
    downside: Fraction | None = None
    position_fraction: Fraction | None = None
    concentration: Fraction | None = None
    daily_dollar_volume: NonNegative | None = None
    correlated_exposure: Fraction | None = None
    reward_risk: NonNegative | None = None
    transaction_cost_fraction: Fraction | None = None
    loss_fraction: Fraction | None = None


class RiskLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_age_seconds: NonNegative
    max_volatility: NonNegative
    max_downside: Fraction
    max_position_fraction: Fraction
    max_concentration: Fraction
    min_daily_dollar_volume: NonNegative
    max_correlated_exposure: Fraction
    min_reward_risk: NonNegative
    max_transaction_cost_fraction: Fraction
    max_loss_fraction: Fraction


class RiskResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["pass", "warning", "fail", "insufficient_evidence"]
    action: Literal["BUY", "HOLD", "SELL", "NO_TRADE"]
    reasons: tuple[str, ...]


def derive_risk_evidence(supplied: RiskEvidence, market: dict | None, forecast: dict | None) -> RiskEvidence:
    """Combine observed data with user inputs using the more conservative value."""
    values = supplied.model_dump()

    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)

    def combine(name, observed, choose):
        prior = values[name]
        values[name] = None if observed is None else observed if prior is None else choose(prior, observed)

    if market is not None:
        try:
            timestamp = datetime.fromisoformat(market.get('as_of', ''))
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                timestamp = None
        except (ValueError, TypeError):
            timestamp = None
        combine('market_time', timestamp, min)
        volatility = market.get('annualized_volatility')
        combine('volatility', float(volatility) if number(volatility) and volatility >= 0 else None, max)
        price, volume = market.get('last_close'), market.get('last_volume')
        liquidity = price * volume if number(price) and number(volume) and price > 0 and volume >= 0 else None
        combine('daily_dollar_volume', float(liquidity) if liquidity is not None and isfinite(liquidity) else None, min)
    if forecast is not None:
        lower = forecast.get('lower_returns')
        downside = min(1.0, max(0.0, -min(lower))) if isinstance(lower, list) and lower and all(number(x) for x in lower) else None
        combine('downside', downside, max)
    return RiskEvidence.model_validate(values)


def evaluate_risk(vote: Consensus, evidence: RiskEvidence, limits: RiskLimits, *, now: datetime | None = None) -> RiskResult:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    missing = [name for name in RiskEvidence.model_fields if getattr(evidence, name) is None]
    if vote.signal is None:
        missing.append("consensus")
    failures = []
    if evidence.market_time is not None:
        age = (now - evidence.market_time).total_seconds()
        if age < 0 or age > limits.max_age_seconds:
            failures.append("market_time")
    checks = (
        ("volatility", limits.max_volatility, "max"),
        ("downside", limits.max_downside, "max"),
        ("position_fraction", limits.max_position_fraction, "max"),
        ("concentration", limits.max_concentration, "max"),
        ("daily_dollar_volume", limits.min_daily_dollar_volume, "min"),
        ("correlated_exposure", limits.max_correlated_exposure, "max"),
        ("reward_risk", limits.min_reward_risk, "min"),
        ("transaction_cost_fraction", limits.max_transaction_cost_fraction, "max"),
        ("loss_fraction", limits.max_loss_fraction, "max"),
    )
    for name, threshold, direction in checks:
        value = getattr(evidence, name)
        if value is not None and ((direction == "max" and value > threshold) or (direction == "min" and value < threshold)):
            failures.append(name)
    if failures:
        return RiskResult(status="fail", action="NO_TRADE", reasons=tuple(failures + missing))
    if missing:
        return RiskResult(status="insufficient_evidence", action="NO_TRADE", reasons=tuple(missing))
    if vote.disagreement:
        return RiskResult(status="warning", action=vote.signal, reasons=("model_disagreement",))
    return RiskResult(status="pass", action=vote.signal, reasons=())


def decide(decisions, evidence: RiskEvidence, limits: RiskLimits, *, now: datetime | None = None) -> RiskResult:
    from adam.decisions import consensus

    return evaluate_risk(consensus(decisions), evidence, limits, now=now)
