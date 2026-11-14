import os
from typing import Optional, List, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Priority credential loading logic
# 1. Runtime Environment Variable (GEMINI_API_KEY already present)
# 2. Streamlit Secrets (st.secrets["GEMINI_API_KEY"])
# 3. Local .env (backend/.env or .env)
# ---------------------------------------------------------------------------

gemini_key = os.environ.get("GEMINI_API_KEY")
credential_source = "environment_variable" if gemini_key else "not_configured"

# Check Streamlit secrets
if not gemini_key:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            gemini_key = str(st.secrets["GEMINI_API_KEY"])
            credential_source = "streamlit_secrets"
    except Exception:
        pass

# Check local .env
if not gemini_key:
    try:
        from dotenv import load_dotenv
        # Check backend/.env first, then root .env
        if os.path.exists("backend/.env"):
            load_dotenv("backend/.env")
        elif os.path.exists(".env"):
            load_dotenv(".env")
        
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            credential_source = ".env"
    except ImportError:
        pass


class Settings(BaseSettings):
    # Gemini Configuration
    gemini_api_key: Optional[str] = None
    credential_source: str = "not_configured"
    gemini_model: str = "gemini-3.5-flash"
    gemini_temperature: float = 0.2
    gemini_top_p: float = 0.95
    gemini_timeout: int = 180  # Default client request timeout in seconds

    # Failover & Resiliency Settings
    max_api_retries: int = 5
    backoff_multiplier: float = 2.0
    fallback_models: Any = "gemini-2.5-flash,gemini-2.0-flash"

    # Circuit Breaker Configuration
    circuit_failure_threshold: int = 3
    circuit_recovery_time: float = 60.0  # seconds

    # Public-deployment protection (leave app_api_key unset for open local dev)
    app_api_key: Optional[str] = None      # shared secret the Streamlit frontend sends as X-API-Key
    rate_limit_per_minute: int = 10        # global cap on /generate calls (protects Gemini quota); 0 = off

    # Agentic Mode (tool-using writer + LLM planner + LLM supervisor advisor)
    # Every agentic component degrades gracefully to the deterministic pipeline.
    agentic_mode: bool = True
    agent_max_tool_steps: int = 8          # hard cap on tool-loop iterations per generation
    agent_step_timeout: int = 60           # seconds per LLM call inside agent loops
    llm_planner_enabled: bool = True
    llm_supervisor_enabled: bool = True

    # Thresholds and Retry Limits (Correction Loop)
    max_retries: int = 3
    quality_threshold: float = 85.0    # Out of 100
    hallucination_threshold: float = 100.0  # Out of 100

    # API Configuration
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    allowed_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    # Outputs and Prompts Paths
    outputs_dir: str = "./outputs"
    prompts_dir: str = "./app/prompts"

    @field_validator('fallback_models', mode='before')
    @classmethod
    def parse_fallback_models(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_file_path_behavior="optional",  # Don't crash if .env is absent
        extra="ignore"
    )


# ---------------------------------------------------------------------------
# Instantiate settings.
# ---------------------------------------------------------------------------
settings = Settings()

# Apply the loaded key and source
settings.gemini_api_key = gemini_key
settings.credential_source = credential_source

# Overlay other st.secrets for Streamlit Cloud deployment if present
for _key, _attr in [
    ("GEMINI_MODEL",    "gemini_model"),
    ("BACKEND_HOST",    "backend_host"),
    ("BACKEND_PORT",    "backend_port"),
    ("ALLOWED_ORIGINS", "allowed_origins"),
]:
    try:
        import streamlit as st
        _val = st.secrets.get(_key)
        if _val:
            _current = getattr(settings, _attr)
            try:
                setattr(settings, _attr, type(_current)(_val))
            except Exception:
                setattr(settings, _attr, _val)
    except Exception:
        pass

# Ensure output/prompt directories exist
os.makedirs(settings.outputs_dir, exist_ok=True)
os.makedirs(settings.prompts_dir, exist_ok=True)
