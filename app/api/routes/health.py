import socket
from fastapi import APIRouter, Header
from typing import Optional
from app.core.config import settings
from app.core.circuit_breaker import CircuitBreaker

router = APIRouter()

def check_dns_connectivity() -> bool:
    """Checks if the system can resolve the Gemini API endpoint host."""
    try:
        socket.gethostbyname("generativelanguage.googleapis.com")
        return True
    except Exception:
        return False

@router.get("", summary="Get system health, failover config, and API circuit status")
async def get_health():
    from app.core.gemini_service import GeminiService
    
    cb = CircuitBreaker(
        model_name=settings.gemini_model,
        failure_threshold=settings.circuit_failure_threshold,
        recovery_time=settings.circuit_recovery_time
    )
    
    dns_online = check_dns_connectivity()
    api_key_set = bool(settings.gemini_api_key)
    
    gemini_status = "connected" if GeminiService.is_connected() else "disconnected"
    credential_source = settings.credential_source
    
    # Calculate status
    if cb.state == "OPEN":
        status = "degraded"
    elif not dns_online:
        status = "degraded"
    elif not api_key_set:
        status = "unhealthy"
    else:
        status = "healthy"

    return {
        "status": status,
        "gemini": gemini_status,
        "credential_source": credential_source,
        "app_version": "1.0.0",
        "gemini_connectivity": {
            "dns_resolution_success": dns_online,
            "api_key_configured": api_key_set,
            "api_key_source": credential_source,
            "auth_status": "Connected" if gemini_status == "connected" else "Failed"
        },
        "models": {
            "active_primary_model": settings.gemini_model,
            "fallback_models_chain": settings.fallback_models
        },
        "resiliency_configuration": {
            "max_api_retries_limit": settings.max_api_retries,
            "backoff_multiplier": settings.backoff_multiplier,
            "request_timeout_seconds": settings.gemini_timeout
        },
        "circuit_breaker": {
            "state": cb.state,
            "consecutive_failures": cb.consecutive_failures,
            "failure_threshold": cb.failure_threshold,
            "cooldown_time_seconds": cb.recovery_time
        }
    }
