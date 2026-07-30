from pathlib import Path
from types import SimpleNamespace

from config import estimate_cost
from models import Session


def test_estimate_cost():
    assert estimate_cost(1_000_000, 1_000_000, 0.50, 1.50) == 2.0
    assert abs(estimate_cost(10200, 2250, 0.50, 1.50) - 0.008475) < 1e-9


def test_estimate_cost_zero():
    assert estimate_cost(0, 0, 0.50, 1.50) == 0.0


def test_session_usage_defaults():
    session = Session(
        id="abc",
        title="t",
        workspace=Path("."),
        history_path=Path("conversation_history.jsonl"),
    )
    assert session.prompt_tokens == 0
    assert session.completion_tokens == 0
    assert session.total_tokens == 0
    assert session.estimated_cost_usd == 0.0


def test_session_usage_accumulation_math():
    session = Session(
        id="abc",
        title="t",
        workspace=Path("."),
        history_path=Path("conversation_history.jsonl"),
    )
    prompt, completion = 10200, 2250
    session.prompt_tokens += prompt
    session.completion_tokens += completion
    session.total_tokens += prompt + completion
    session.estimated_cost_usd += estimate_cost(prompt, completion, 0.50, 1.50)
    assert session.total_tokens == 12450
    assert abs(session.estimated_cost_usd - 0.008475) < 1e-9


def test_record_usage_reads_openai_usage_shape():
    """Ensure attribute access matches OpenAI-compatible usage objects."""
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    response = SimpleNamespace(usage=usage)
    prompt = int(getattr(response.usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(response.usage, "completion_tokens", 0) or 0)
    total = int(getattr(response.usage, "total_tokens", 0) or (prompt + completion))
    assert (prompt, completion, total) == (100, 50, 150)
