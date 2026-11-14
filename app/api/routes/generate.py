from fastapi import APIRouter, HTTPException
from typing import Optional
from app.models.product import Product
from app.models.response import PipelineResponse
from app.orchestrator.orchestrator import AIOrchestrator
from app.engines.normalizer import NormalizationEngine
from app.engines.validator import AttributeValidationEngine
from app.core.logger import logger

from app.core.exceptions import (
    CircuitBreakerOpenError,
    GeminiTimeoutError,
    GeminiUnavailableError,
    RetryLimitExceededError
)

router = APIRouter()

@router.post("/generate", response_model=PipelineResponse, summary="Generates and self-corrects SEO content for a product")
def generate_product_content(
    product: Product
):
    """
    Runs the complete production pipeline:
    - Input Normalization
    - Attribute Verification
    - Category Classification & Enrichment
    - Dynamic Prompt Construction
    - Gemini Model Call (with optional UI API key override)
    - JSON validation & post-processing validation checks
    - Self-correction retries if quality threshold or safety is violated
    - Reports export and finalized payload returned.
    """
    logger.info(f"Received API request /generate for product: {product.product_name}")
    import re
    
    def redact_key(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"AIzaSy[A-Za-z0-9_\-]{25,}", "[REDACTED_API_KEY]", text)

    try:
        orchestrator = AIOrchestrator()
        response = orchestrator.run_pipeline(product)
        return response
    except ValueError as ve:
        clean_err = redact_key(str(ve))
        logger.error(f"Validation error in generation pipeline: {clean_err}")
        raise HTTPException(status_code=400, detail=clean_err)
    except CircuitBreakerOpenError as cbe:
        clean_err = redact_key(str(cbe))
        logger.error(f"Circuit Breaker blocked call: {clean_err}")
        raise HTTPException(status_code=503, detail=clean_err)
    except GeminiTimeoutError as gte:
        clean_err = redact_key(str(gte))
        logger.error(f"Gemini API timed out: {clean_err}")
        raise HTTPException(status_code=504, detail=clean_err)
    except GeminiUnavailableError as gue:
        clean_err = redact_key(str(gue))
        logger.error(f"Gemini API transient failure: {clean_err}")
        raise HTTPException(status_code=502, detail=clean_err)
    except RetryLimitExceededError as rlee:
        clean_err = redact_key(str(rlee))
        logger.error(f"All retry attempts exhausted: {clean_err}")
        raise HTTPException(status_code=500, detail=clean_err)
    except Exception as e:
        clean_err = redact_key(str(e))
        logger.error(f"Unexpected error running generation pipeline: {clean_err}")
        raise HTTPException(status_code=500, detail=f"Pipeline error occurred: {clean_err}")

@router.post("/validate", summary="Normalizes and runs semantic validation on raw catalog attributes")
def validate_product_attributes(product: Product):
    """
    Executes the pre-processing validation stage only:
    - Normalizes strings and units (e.g. weight, color, features)
    - Checks mandatory requirements
    Returns validation report and normalized object.
    """
    logger.info(f"Received API request /validate for product: {product.product_name}")
    try:
        normalized = NormalizationEngine.normalize(product)
        report = AttributeValidationEngine.validate(normalized)
        return {
            "validation_report": report,
            "normalized_product": normalized
        }
    except Exception as e:
        logger.error(f"Error validating product attributes: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
