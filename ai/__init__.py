"""Claude as a reasoning layer (Milestone 4): reviews trade candidates and
answers TRADE or NO TRADE. The Risk Manager still decides; nothing executes."""
from ai.advisor import AIReview, ClaudeAdvisor
from ai.briefing import Candidate, MarketSnapshot, build_brief, risk_state
from ai.llm import AnthropicLLM, LLMResponse, ScriptedLLM
from ai.schema import AIDecision, Verdict

__all__ = ["AIDecision", "AIReview", "AnthropicLLM", "Candidate", "ClaudeAdvisor",
           "LLMResponse", "MarketSnapshot", "ScriptedLLM", "Verdict", "build_brief",
           "risk_state"]
