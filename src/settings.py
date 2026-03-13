from __future__ import annotations

from dataclasses import asdict, dataclass

from src.utils import DATA_DIR, env_flag, env_float, env_int, env_str, get_secret, load_json, save_json


SETTINGS_FILE = DATA_DIR / "ui_settings.json"


@dataclass
class RuntimeSettings:
    """Session-friendly runtime settings for providers, generation, and RAG behavior."""

    provider: str = "openai"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 220
    top_p: float = 0.9
    retrieval_k: int = 10
    final_retrieval_k: int = 4
    rerank_enabled: bool = True
    general_fallback_enabled: bool = True

    def normalized_provider(self) -> str:
        """Return the normalized provider key."""

        provider = self.provider.strip().lower()
        return provider if provider in {"ollama", "openai", "test"} else "ollama"

    def safe_for_persistence(self) -> dict:
        """Return a JSON-safe payload without secret values."""

        payload = asdict(self)
        payload["openai_api_key"] = ""
        return payload


def default_runtime_settings() -> RuntimeSettings:
    """Build runtime settings from environment defaults."""

    return RuntimeSettings(
        provider=env_str("LLM_PROVIDER", "openai"),
        ollama_base_url=env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=env_str("OLLAMA_MODEL", "llama3"),
        openai_api_key=get_secret("OPENAI_API_KEY", ""),
        openai_model=env_str("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=env_float("LLM_TEMPERATURE", 0.2),
        max_tokens=env_int("LLM_MAX_TOKENS", 220),
        top_p=env_float("LLM_TOP_P", 0.9),
        retrieval_k=env_int("RAG_INITIAL_RETRIEVAL_K", 10),
        final_retrieval_k=env_int("RAG_FINAL_RETRIEVAL_K", 4),
        rerank_enabled=env_flag("RAG_ENABLE_RERANK", True),
        general_fallback_enabled=env_flag("RAG_ENABLE_GENERAL_FALLBACK", True),
    )


def load_persisted_settings() -> dict:
    """Load locally persisted non-secret UI settings."""

    if not SETTINGS_FILE.exists():
        return {}
    try:
        data = load_json(SETTINGS_FILE)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_runtime_settings() -> RuntimeSettings:
    """Return runtime settings using persisted non-secret values over env defaults."""

    settings = default_runtime_settings()
    persisted = load_persisted_settings()
    for key, value in persisted.items():
        if hasattr(settings, key) and key != "openai_api_key":
            setattr(settings, key, value)
    return settings


def persist_runtime_settings(settings: RuntimeSettings) -> None:
    """Persist non-secret settings for later local sessions."""

    save_json(settings.safe_for_persistence(), SETTINGS_FILE)
