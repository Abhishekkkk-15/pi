"""App config, provider builtins, and auth.json helpers."""

from __future__ import annotations

from dotenv import load_dotenv
import os
from typing import Any, Optional

from memory import Memory

load_dotenv()

# Built-in OpenAI-compatible providers
BUILTIN_PROVIDERS: dict[str, dict[str, str]] = {
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
}

ACTIVE_PROVIDER_KEY = "active_provider"
CUSTOM_MARKER = "is_custom"


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_price: float,
    output_price: float,
) -> float:
    """Estimate USD cost from token counts and per-million-token prices."""
    return (
        (prompt_tokens / 1_000_000) * input_price
        + (completion_tokens / 1_000_000) * output_price
    )


def auth_path() -> Any:
    return Memory.root / "auth.json"


def load_auth() -> dict[str, Any]:
    return Memory.read_from_json(auth_path()) or {}


def save_auth(data: dict[str, Any]) -> None:
    Memory.write_to_json(auth_path(), data)


def _ensure_llm_settings(auth: dict[str, Any]) -> dict[str, Any]:
    """Normalize auth.json into per-provider llm_settings shape (migrates legacy)."""
    llm = auth.get("llm_settings")
    if not isinstance(llm, dict):
        llm = {}

    # Legacy: credentials.api_key + llm_settings.provider
    legacy_creds = auth.get("credentials") if isinstance(auth.get("credentials"), dict) else {}
    legacy_key = legacy_creds.get("api_key") if legacy_creds else None
    legacy_provider = llm.get("provider") if isinstance(llm.get("provider"), str) else None

    active = llm.get(ACTIVE_PROVIDER_KEY)
    if not active:
        active = (legacy_provider or os.getenv("LLM_PROVIDER", "mistral")).lower()
        llm[ACTIVE_PROVIDER_KEY] = active

    # Ensure built-in provider buckets exist
    for name, meta in BUILTIN_PROVIDERS.items():
        bucket = llm.get(name)
        if not isinstance(bucket, dict):
            bucket = {}
            llm[name] = bucket
        bucket.setdefault("base_url", meta["base_url"])
        bucket.setdefault("model", meta["default_model"])
        bucket.setdefault("api_key", "")

    # Migrate legacy flat api_key into active provider if empty
    active_bucket = llm.get(active)
    if not isinstance(active_bucket, dict):
        active_bucket = {}
        llm[active] = active_bucket
    if legacy_key and not active_bucket.get("api_key"):
        active_bucket["api_key"] = legacy_key

    auth["llm_settings"] = llm
    return auth


def list_provider_names(auth: Optional[dict[str, Any]] = None) -> list[str]:
    """Built-ins first, then custom provider names."""
    data = _ensure_llm_settings(auth if auth is not None else load_auth())
    llm = data["llm_settings"]
    names = list(BUILTIN_PROVIDERS.keys())
    for key, val in llm.items():
        if key in (ACTIVE_PROVIDER_KEY, "provider"):
            continue
        if key in names:
            continue
        if isinstance(val, dict):
            names.append(key)
    return names


def get_provider_settings(provider: str, auth: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    data = _ensure_llm_settings(auth if auth is not None else load_auth())
    llm = data["llm_settings"]
    bucket = llm.get(provider)
    if not isinstance(bucket, dict):
        bucket = {}
    builtin = BUILTIN_PROVIDERS.get(provider, {})
    return {
        "api_key": bucket.get("api_key") or "",
        "model": bucket.get("model") or builtin.get("default_model") or "",
        "base_url": bucket.get("base_url") or builtin.get("base_url") or "",
        "is_custom": bool(bucket.get(CUSTOM_MARKER)) or provider not in BUILTIN_PROVIDERS,
    }


def set_active_provider(provider: str, auth: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    data = _ensure_llm_settings(auth if auth is not None else load_auth())
    data["llm_settings"][ACTIVE_PROVIDER_KEY] = provider
    # Drop legacy flat provider field to avoid confusion
    data["llm_settings"].pop("provider", None)
    save_auth(data)
    return data


def upsert_provider_settings(
    provider: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    is_custom: Optional[bool] = None,
    make_active: bool = False,
    auth: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    data = _ensure_llm_settings(auth if auth is not None else load_auth())
    llm = data["llm_settings"]
    bucket = llm.get(provider)
    if not isinstance(bucket, dict):
        bucket = {}
        llm[provider] = bucket

    if api_key is not None:
        bucket["api_key"] = api_key
    if model is not None:
        bucket["model"] = model
    if base_url is not None:
        bucket["base_url"] = base_url
    if is_custom is not None:
        bucket[CUSTOM_MARKER] = is_custom
    elif provider not in BUILTIN_PROVIDERS:
        bucket[CUSTOM_MARKER] = True

    # Keep built-in defaults filled
    if provider in BUILTIN_PROVIDERS:
        bucket.setdefault("base_url", BUILTIN_PROVIDERS[provider]["base_url"])
        bucket.setdefault("model", BUILTIN_PROVIDERS[provider]["default_model"])

    if make_active:
        llm[ACTIVE_PROVIDER_KEY] = provider
        llm.pop("provider", None)

    # Keep legacy credentials in sync for older readers
    credentials = data.get("credentials") if isinstance(data.get("credentials"), dict) else {}
    active = llm.get(ACTIVE_PROVIDER_KEY, provider)
    active_bucket = llm.get(active) if isinstance(llm.get(active), dict) else {}
    if active_bucket.get("api_key"):
        credentials["api_key"] = active_bucket["api_key"]
        data["credentials"] = credentials

    save_auth(data)
    return data


class Config:
    tavily_api_key: str | None = None
    is_dev: bool = True
    model: str = "mistral-large-latest"
    max_history_messages: int = 50
    autonomous_risk: bool = False
    input_price_per_mtok: float = 0.50
    output_price_per_mtok: float = 1.50
    api_key: str | None = None
    provider: str = "mistral"
    base_url: str = BUILTIN_PROVIDERS["mistral"]["base_url"]

    def __init__(self):
        auth = _ensure_llm_settings(load_auth())
        llm = auth["llm_settings"]

        self.provider = str(llm.get(ACTIVE_PROVIDER_KEY) or "mistral").lower()
        settings = get_provider_settings(self.provider, auth)

        env_key = os.getenv("LLM_KEY")
        self.api_key = settings["api_key"] or env_key or None
        self.base_url = settings["base_url"] or BUILTIN_PROVIDERS["mistral"]["base_url"]
        self.model = (
            os.getenv("LLM_MODEL")
            or settings["model"]
            or BUILTIN_PROVIDERS.get(self.provider, {}).get("default_model")
            or "mistral-large-latest"
        )

        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

        env_mode = os.getenv("ENVIRONMENT") or os.getenv("enviroment") or os.getenv("ENV")
        self.is_dev = (
            env_mode.lower() in ("dev", "development", "true", "1") if env_mode else True
        )

        max_hist = os.getenv("MAX_HISTORY_MESSAGES")
        if max_hist and max_hist.isdigit():
            self.max_history_messages = int(max_hist)

        auto_risk = os.getenv("AUTONOMOUS_RISK")
        if auto_risk:
            self.autonomous_risk = auto_risk.lower() in ("true", "1", "yes")

        try:
            self.input_price_per_mtok = float(os.getenv("INPUT_PRICE_PER_MTOK", "0.50"))
        except ValueError:
            self.input_price_per_mtok = 0.50

        try:
            self.output_price_per_mtok = float(os.getenv("OUTPUT_PRICE_PER_MTOK", "1.50"))
        except ValueError:
            self.output_price_per_mtok = 1.50

    def reload_from_auth(self) -> None:
        """Refresh provider/model/api_key/base_url from auth.json."""
        auth = _ensure_llm_settings(load_auth())
        llm = auth["llm_settings"]
        self.provider = str(llm.get(ACTIVE_PROVIDER_KEY) or self.provider).lower()
        settings = get_provider_settings(self.provider, auth)
        self.api_key = settings["api_key"] or os.getenv("LLM_KEY") or None
        self.base_url = settings["base_url"] or self.base_url
        self.model = settings["model"] or self.model
