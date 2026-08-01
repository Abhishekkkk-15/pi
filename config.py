"""App config, provider builtins, and auth.json helpers."""

from __future__ import annotations

from dotenv import load_dotenv
import os
from typing import Any, Optional

from memory import Memory, is_development

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
MAX_KEYS_PER_PROVIDER = 2


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


def _normalize_provider_keys(bucket: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy api_key -> api_keys[] and clamp active_key_index."""
    keys_raw = bucket.get("api_keys")
    keys: list[str] = []
    if isinstance(keys_raw, list):
        keys = [str(k).strip() for k in keys_raw if k and str(k).strip()]

    legacy = bucket.get("api_key")
    if legacy and str(legacy).strip():
        legacy_s = str(legacy).strip()
        if legacy_s not in keys:
            keys.insert(0, legacy_s)

    keys = keys[:MAX_KEYS_PER_PROVIDER]

    try:
        idx = int(bucket.get("active_key_index", 0) or 0)
    except (TypeError, ValueError):
        idx = 0
    if not keys:
        idx = 0
    else:
        idx = max(0, min(idx, len(keys) - 1))

    bucket["api_keys"] = keys
    bucket["active_key_index"] = idx
    bucket["api_key"] = keys[idx] if keys else ""
    return bucket


def _sync_legacy_credentials(data: dict[str, Any]) -> None:
    llm = data.get("llm_settings") if isinstance(data.get("llm_settings"), dict) else {}
    active = llm.get(ACTIVE_PROVIDER_KEY, "mistral")
    bucket = llm.get(active) if isinstance(llm.get(active), dict) else {}
    _normalize_provider_keys(bucket)
    credentials = data.get("credentials") if isinstance(data.get("credentials"), dict) else {}
    if bucket.get("api_key"):
        credentials["api_key"] = bucket["api_key"]
        data["credentials"] = credentials


def _ensure_llm_settings(auth: dict[str, Any]) -> dict[str, Any]:
    """Normalize auth.json into per-provider llm_settings shape (migrates legacy)."""
    llm = auth.get("llm_settings")
    if not isinstance(llm, dict):
        llm = {}

    legacy_creds = auth.get("credentials") if isinstance(auth.get("credentials"), dict) else {}
    legacy_key = legacy_creds.get("api_key") if legacy_creds else None
    legacy_provider = llm.get("provider") if isinstance(llm.get("provider"), str) else None

    active = llm.get(ACTIVE_PROVIDER_KEY)
    if not active:
        active = (legacy_provider or os.getenv("LLM_PROVIDER", "mistral")).lower()
        llm[ACTIVE_PROVIDER_KEY] = active

    for name, meta in BUILTIN_PROVIDERS.items():
        bucket = llm.get(name)
        if not isinstance(bucket, dict):
            bucket = {}
            llm[name] = bucket
        bucket.setdefault("base_url", meta["base_url"])
        bucket.setdefault("model", meta["default_model"])
        _normalize_provider_keys(bucket)

    active_bucket = llm.get(active)
    if not isinstance(active_bucket, dict):
        active_bucket = {}
        llm[active] = active_bucket
    _normalize_provider_keys(active_bucket)
    if legacy_key and not active_bucket.get("api_keys"):
        active_bucket["api_keys"] = [str(legacy_key).strip()]
        _normalize_provider_keys(active_bucket)

    for key, val in list(llm.items()):
        if key in (ACTIVE_PROVIDER_KEY, "provider"):
            continue
        if isinstance(val, dict):
            _normalize_provider_keys(val)

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
    _normalize_provider_keys(bucket)
    builtin = BUILTIN_PROVIDERS.get(provider, {})
    keys = list(bucket.get("api_keys") or [])
    idx = int(bucket.get("active_key_index", 0) or 0)
    return {
        "api_key": keys[idx] if keys else "",
        "api_keys": keys,
        "active_key_index": idx,
        "key_count": len(keys),
        "model": bucket.get("model") or builtin.get("default_model") or "",
        "base_url": bucket.get("base_url") or builtin.get("base_url") or "",
        "is_custom": bool(bucket.get(CUSTOM_MARKER)) or provider not in BUILTIN_PROVIDERS,
    }


def set_active_provider(provider: str, auth: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    data = _ensure_llm_settings(auth if auth is not None else load_auth())
    data["llm_settings"][ACTIVE_PROVIDER_KEY] = provider
    data["llm_settings"].pop("provider", None)
    _sync_legacy_credentials(data)
    save_auth(data)
    return data


def get_active_api_key(provider: str, auth: Optional[dict[str, Any]] = None) -> str:
    """Return the currently active API key string for a provider (may be empty)."""
    settings = get_provider_settings(provider, auth)
    return str(settings.get("api_key") or "")


def set_provider_key(
    provider: str,
    slot: int,
    api_key: str,
    *,
    make_active_slot: bool = True,
    auth: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Set Primary (0) or Secondary (1) API key for a provider."""
    if slot not in (0, 1):
        raise ValueError("slot must be 0 (Primary) or 1 (Secondary)")
    key = (api_key or "").strip()
    if not key:
        raise ValueError("api_key cannot be empty")

    data = _ensure_llm_settings(auth if auth is not None else load_auth())
    llm = data["llm_settings"]
    bucket = llm.get(provider)
    if not isinstance(bucket, dict):
        bucket = {}
        llm[provider] = bucket
    _normalize_provider_keys(bucket)

    existing = list(bucket.get("api_keys") or [])
    primary = existing[0] if len(existing) >= 1 else ""
    secondary = existing[1] if len(existing) >= 2 else ""

    if slot == 0:
        primary = key
    else:
        if not primary:
            # No primary yet: store as the sole (primary) key
            primary = key
            secondary = ""
            slot = 0
        else:
            secondary = key

    keys: list[str] = []
    if primary:
        keys.append(primary)
    if secondary:
        keys.append(secondary)

    bucket["api_keys"] = keys[:MAX_KEYS_PER_PROVIDER]
    if make_active_slot:
        if slot == 1 and len(bucket["api_keys"]) > 1:
            bucket["active_key_index"] = 1
        else:
            bucket["active_key_index"] = 0
    _normalize_provider_keys(bucket)
    _sync_legacy_credentials(data)
    save_auth(data)
    return data


def rotate_provider_key(provider: str, auth: Optional[dict[str, Any]] = None) -> Optional[str]:
    """
    Flip active_key_index when a second key exists.
    Returns the new active key, or None if rotation is not possible.
    """
    data = _ensure_llm_settings(auth if auth is not None else load_auth())
    llm = data["llm_settings"]
    bucket = llm.get(provider)
    if not isinstance(bucket, dict):
        return None
    _normalize_provider_keys(bucket)
    keys = list(bucket.get("api_keys") or [])
    if len(keys) < 2:
        return None
    idx = int(bucket.get("active_key_index", 0) or 0)
    bucket["active_key_index"] = 1 - idx
    _normalize_provider_keys(bucket)
    _sync_legacy_credentials(data)
    save_auth(data)
    return bucket.get("api_key") or None


def upsert_provider_settings(
    provider: str,
    *,
    api_key: Optional[str] = None,
    key_slot: int = 0,
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
        set_provider_key(
            provider,
            key_slot,
            api_key,
            make_active_slot=True,
            auth=data,
        )
        data = _ensure_llm_settings(load_auth())
        llm = data["llm_settings"]
        bucket = llm.get(provider) if isinstance(llm.get(provider), dict) else {}

    if model is not None:
        bucket["model"] = model
    if base_url is not None:
        bucket["base_url"] = base_url
    if is_custom is not None:
        bucket[CUSTOM_MARKER] = is_custom
    elif provider not in BUILTIN_PROVIDERS:
        bucket[CUSTOM_MARKER] = True

    if provider in BUILTIN_PROVIDERS:
        bucket.setdefault("base_url", BUILTIN_PROVIDERS[provider]["base_url"])
        bucket.setdefault("model", BUILTIN_PROVIDERS[provider]["default_model"])

    _normalize_provider_keys(bucket)

    if make_active:
        llm[ACTIVE_PROVIDER_KEY] = provider
        llm.pop("provider", None)

    _sync_legacy_credentials(data)
    save_auth(data)
    return data


def get_tavily_api_key(auth: Optional[dict[str, Any]] = None) -> str:
    """Return Tavily API key from auth.json credentials, falling back to env."""
    data = auth if auth is not None else load_auth()
    creds = data.get("credentials") if isinstance(data.get("credentials"), dict) else {}
    key = (creds.get("tavily_api_key") or "").strip()
    if key:
        return key
    return (os.getenv("TAVILY_API_KEY") or "").strip()


def set_tavily_api_key(api_key: str, auth: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Persist Tavily API key under credentials.tavily_api_key in auth.json."""
    key = (api_key or "").strip()
    if not key:
        raise ValueError("tavily_api_key cannot be empty")
    data = auth if auth is not None else load_auth()
    if not isinstance(data, dict):
        data = {}
    creds = data.get("credentials") if isinstance(data.get("credentials"), dict) else {}
    creds["tavily_api_key"] = key
    data["credentials"] = creds
    save_auth(data)
    return data


class Config:
    tavily_api_key: str | None = None
    is_dev: bool = True
    model: str = "mistral-large-latest"
    max_history_messages: int = 32
    autonomous_risk: bool = False
    input_price_per_mtok: float = 0.50
    output_price_per_mtok: float = 1.50
    api_key: str | None = None
    provider: str = "mistral"
    base_url: str = BUILTIN_PROVIDERS["mistral"]["base_url"]
    active_key_index: int = 0
    key_count: int = 0

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
        self.active_key_index = int(settings.get("active_key_index", 0) or 0)
        self.key_count = int(settings.get("key_count", 0) or 0)

        self.tavily_api_key = get_tavily_api_key(auth) or None

        self.is_dev = is_development()

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
        self.active_key_index = int(settings.get("active_key_index", 0) or 0)
        self.key_count = int(settings.get("key_count", 0) or 0)
        self.tavily_api_key = get_tavily_api_key(auth) or None
