import os
from google import genai
from app.core.config import settings
from app.core.logger import logger

class GeminiService:
    _client = None

    @classmethod
    def initialize(cls):
        """Initializes the single shared Gemini client using the configured settings."""
        if not settings.gemini_api_key:
            err_msg = (
                "Gemini API key not configured.\n\n"
                "Expected one of:\n"
                "• Environment Variable (GEMINI_API_KEY)\n"
                "• Streamlit Secrets\n"
                "• backend/.env"
            )
            raise ValueError(err_msg)
            
        clean_key = "".join(c for c in str(settings.gemini_api_key) if c.isprintable()).strip()
        if not clean_key:
            raise ValueError("Gemini API key is configured but empty.")
            
        cls._client = genai.Client(api_key=clean_key)
        logger.info("Shared Gemini Client successfully initialized.")

    @classmethod
    def get_client(cls) -> genai.Client:
        """Returns the single shared Gemini client, initializing it if necessary."""
        if cls._client is None:
            cls.initialize()
        return cls._client

    @classmethod
    def is_connected(cls) -> bool:
        """Returns True if the shared Gemini client has been initialized."""
        return cls._client is not None
