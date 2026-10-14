"""Builds the advisor for the entry points (main.py, paper_trade.py)."""
from __future__ import annotations

import sys
from typing import Optional

from ai.advisor import ClaudeAdvisor
from ai.llm import AnthropicLLM, MissingAPIKey
from config.settings import AppSettings


def build_advisor(settings: AppSettings, mode: str, engine=None) -> Optional[ClaudeAdvisor]:
    """Claude, if --ai or AI_ENABLED=true; else None. Exits with a clear
    message if the API key is missing. With a database engine, every
    decision is stored in ai_decisions."""
    if not settings.ai.enabled:
        return None
    try:
        llm = AnthropicLLM(settings.ai)
    except MissingAPIKey as e:
        sys.exit(str(e))
    journal = None
    if engine is not None:
        from database import AIDecisionRepository
        journal = AIDecisionRepository(engine)
    return ClaudeAdvisor(llm, settings.ai, mode=mode, journal=journal)
