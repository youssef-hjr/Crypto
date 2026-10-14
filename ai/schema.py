"""
What Claude must answer with (PDF section 51): TRADE or NO TRADE, plus
reasoning, confidence, expected R, invalidation, market regime and
portfolio impact.

The same shape is sent to the API as a JSON schema (structured output), and
the answer is validated again here before anything uses it. Anything that
fails validation is treated as NO TRADE.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from models import MarketRegime


class Verdict(str, Enum):
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"


class AIDecision(BaseModel):
    decision: Verdict
    reasoning: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0)
    expected_r: float = Field(ge=-10.0, le=20.0)
    invalidation: str = Field(max_length=1000)
    market_regime: MarketRegime
    portfolio_impact: str = Field(max_length=1000)
    conflicting_signals: list[str] = Field(default_factory=list, max_length=10)
    missing_information: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("reasoning")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reasoning is empty")
        return v.strip()

    @property
    def is_trade(self) -> bool:
        return self.decision == Verdict.TRADE

    @classmethod
    def no_trade(cls, why: str) -> "AIDecision":
        """The safe answer used whenever Claude can't give a valid one."""
        return cls(decision=Verdict.NO_TRADE, reasoning=why, confidence=0.0, expected_r=0.0,
                   invalidation="", market_regime=MarketRegime.SIDEWAYS, portfolio_impact="")


def _strings(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


# Structured-output schema. Bounds (0..1 and so on) are checked by the
# pydantic model above, after the answer comes back.
DECISION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": [v.value for v in Verdict]},
        "reasoning": {"type": "string",
                      "description": "Why, in a few sentences, citing the numbers that matter"},
        "confidence": {"type": "number", "description": "0 to 1"},
        "expected_r": {"type": "number",
                       "description": "Expected result in R after costs; negative if the odds are against it"},
        "invalidation": {"type": "string",
                         "description": "What would prove the idea wrong, besides the stop"},
        "market_regime": {"type": "string", "enum": [r.value for r in MarketRegime]},
        "portfolio_impact": {"type": "string",
                             "description": "How it changes the portfolio's risk and concentration"},
        "conflicting_signals": _strings("Data that argues against the trade"),
        "missing_information": _strings("Data you would want but were not given"),
    },
    "required": ["decision", "reasoning", "confidence", "expected_r", "invalidation",
                 "market_regime", "portfolio_impact", "conflicting_signals",
                 "missing_information"],
    "additionalProperties": False,
}
