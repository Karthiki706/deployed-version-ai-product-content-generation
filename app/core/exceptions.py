class ContentPlatformError(Exception):
    """Base exception class for all errors in the Content Generation Platform."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class GeminiUnavailableError(ContentPlatformError):
    """Raised when the Gemini API is unresponsive or returns transient errors (e.g. 503, 500, 429)."""
    pass

class GeminiTimeoutError(ContentPlatformError):
    """Raised when the request to Gemini API exceeds the configured client-side timeout limit."""
    pass

class InvalidJSONResponseError(ContentPlatformError):
    """Raised when Gemini returns an output that does not match valid JSON or Pydantic schemas."""
    pass

class RetryLimitExceededError(ContentPlatformError):
    """Raised when all retry attempts and fallback models have been exhausted without success."""
    pass

class ConfigurationError(ContentPlatformError):
    """Raised when environment variables or dynamic credentials are invalid or missing."""
    pass

class CircuitBreakerOpenError(ContentPlatformError):
    """Raised when a request is blocked because the Circuit Breaker is in an OPEN state due to previous failures."""
    pass
