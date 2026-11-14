import time
import random
import concurrent.futures
from typing import Dict, Any, Optional, Tuple, List
from google import genai
from google.genai import types
from google.genai import errors

from app.core.config import settings
from app.core.logger import logger
from app.core.circuit_breaker import CircuitBreaker
from app.core.request_logger import RequestLogger
from app.core.exceptions import (
    GeminiUnavailableError,
    GeminiTimeoutError,
    CircuitBreakerOpenError,
    RetryLimitExceededError
)

class GeminiContentGenerator:
    """Interfaces with Google Gemini API with retries, timeout cancellations, fallbacks, and circuit breaker checks."""
    
    def __init__(self, api_key: Optional[str] = None):
        from app.core.gemini_service import GeminiService
        self.client = GeminiService.get_client()
        
    def generate(
        self, 
        prompt: str, 
        req_logger: RequestLogger,
        model_override: Optional[str] = None, 
        temperature: Optional[float] = None, 
        top_p: Optional[float] = None
    ) -> str:
        """
        Generates content from the Gemini API. Coordinates timeouts, retries, and model fallbacks.
        
        Returns:
            str: Raw JSON response text from Gemini.
        """
        primary_model = model_override or settings.gemini_model
        
        # Build the chain of models to try: Primary -> Fallback 1 -> Fallback 2 -> ...
        models_chain = [primary_model]
        for m in settings.fallback_models:
            if m not in models_chain:
                models_chain.append(m)
                
        active_temp = temperature if temperature is not None else settings.gemini_temperature
        active_topp = top_p if top_p is not None else settings.gemini_top_p
        
        last_exception = None
        
        for model_idx, model_id in enumerate(models_chain):
            is_fallback = model_idx > 0
            
            # Instantiate model-specific Circuit Breaker (independent states)
            cb = CircuitBreaker(
                model_name=model_id,
                failure_threshold=settings.circuit_failure_threshold,
                recovery_time=settings.circuit_recovery_time
            )
            
            # Pre-flight check: skip model entirely if its circuit is open
            if not cb.can_execute():
                elapsed = time.time() - cb.last_state_change
                cooldown_remaining = max(0.0, cb.recovery_time - elapsed)
                logger.warning(
                    f"Skipping model '{model_id}' - Circuit Breaker is OPEN. "
                    f"Cooldown Remaining: {cooldown_remaining:.2f}s | Fallback Occurred: {is_fallback}"
                )
                last_exception = CircuitBreakerOpenError(f"Circuit Breaker is OPEN for model '{model_id}'.")
                continue
                
            if is_fallback:
                logger.warning(f"Cascading to Fallback Model: '{model_id}' (Fallback {model_idx}/{len(models_chain)-1})")
                req_logger.fallback_model_used = model_id
                req_logger.add_warning(f"Primary model failed. Switched to fallback: {model_id}")
            else:
                req_logger.model_used = model_id
                
            # Perform retries with exponential backoff on this active model
            for attempt in range(settings.max_api_retries):
                # Check circuit breaker before each attempt (handles trip during retries)
                if not cb.can_execute():
                    elapsed = time.time() - cb.last_state_change
                    cooldown_remaining = max(0.0, cb.recovery_time - elapsed)
                    logger.error(
                        f"Circuit Breaker tripped to OPEN for '{model_id}' during retry attempts. "
                        f"Cooldown Remaining: {cooldown_remaining:.2f}s. Switching to fallback."
                    )
                    last_exception = CircuitBreakerOpenError(f"Circuit Breaker tripped to OPEN for model '{model_id}'.")
                    break
                
                attempt_str = f"{attempt + 1}/{settings.max_api_retries}"
                
                # Log required parameters for every attempt
                elapsed_since_change = time.time() - cb.last_state_change
                cooldown_remaining = max(0.0, cb.recovery_time - elapsed_since_change) if cb.state == "OPEN" else 0.0
                
                logger.info(
                    f"[Request Execution] Model: {model_id} | Circuit State: {cb.state} | "
                    f"Retry Count: {attempt} | Cooldown Remaining: {cooldown_remaining:.2f}s | "
                    f"Fallback Occurred: {is_fallback}"
                )
                
                # Setup ThreadPool for request timeout cancellation
                start_time = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.client.models.generate_content,
                        model=model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=active_temp,
                            top_p=active_topp,
                            response_mime_type="application/json"
                        )
                    )
                    
                    try:
                        # Timeout Check & Execution
                        response = future.result(timeout=settings.gemini_timeout)
                        duration = time.time() - start_time
                        
                        # Add duration to request logs
                        req_logger.gemini_duration += duration
                        
                        if not response.text:
                            raise GeminiUnavailableError("Empty text response from Gemini API.")
                            
                        # Success: Reset Circuit Breaker for this model and return
                        cb.record_success()
                        return response.text
                        
                    except concurrent.futures.TimeoutError as te:
                        duration = time.time() - start_time
                        req_logger.gemini_duration += duration
                        logger.error(f"Request timeout exceeded {settings.gemini_timeout}s on attempt {attempt_str} for model '{model_id}'.")
                        last_exception = GeminiTimeoutError(f"Request timed out on model '{model_id}' after {settings.gemini_timeout}s.")
                        
                        cb.record_failure()
                        
                    except Exception as e:
                        duration = time.time() - start_time
                        req_logger.gemini_duration += duration
                        
                        is_transient, reason = self._is_transient_error(e)
                        logger.error(f"API Error on attempt {attempt_str} for model '{model_id}': {str(e)} (Transient={is_transient}, Reason='{reason}')")
                        
                        last_exception = GeminiUnavailableError(
                            f"Gemini API returned error for model '{model_id}': {str(e)}",
                            details={"code": getattr(e, "code", None), "reason": reason}
                        )
                        
                        cb.record_failure()
                        
                        # If not transient, do not retry this model. Failover to next model.
                        if not is_transient:
                            break
                            
                # Jittered Exponential Backoff Delay
                if attempt < settings.max_api_retries - 1:
                    delay = (settings.backoff_multiplier ** attempt) + random.uniform(0.0, 1.0)
                    logger.info(f"Retry {attempt + 1}/{settings.max_api_retries}. Reason: {str(last_exception)}. Waiting {delay:.2f} seconds...")
                    req_logger.increment_retries()
                    time.sleep(delay)
                    
        # If all models in the chain failed
        raise RetryLimitExceededError(
            f"All models ({', '.join(models_chain)}) and API retry limits exhausted. Last failure: {str(last_exception)}"
        )

    def _is_transient_error(self, e: Exception) -> Tuple[bool, str]:
        """
        Determines if the exception represents a transient/recoverable API failure.
        
        Returns:
            Tuple[bool, str]: (is_transient, reason_string)
        """
        # Catch standard SDK APIErrors
        if isinstance(e, errors.APIError):
            code = getattr(e, "code", None)
            if code in [429, 500, 502, 503, 504]:
                return True, f"HTTP {code} API Error"
            return False, f"HTTP {code} API Error (Permanent)"
            
        err_str = str(e).lower()
        
        # Check standard transient error codes in message
        for code in ["429", "500", "502", "503", "504"]:
            if code in err_str:
                return True, f"HTTP {code} found in message"
                
        # Connection timeouts, DNS resolution issues, or Socket shutdowns
        transient_phrases = [
            "timeout", "timed out", "connection reset", "connection refused", 
            "host unreachable", "dns", "broken pipe", "service unavailable", "overloaded"
        ]
        for phrase in transient_phrases:
            if phrase in err_str:
                return True, f"Network phrase: '{phrase}'"
                
        return False, "Unknown Permanent Error"
