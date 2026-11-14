import json
import time
import hashlib
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.models.product import Product
from app.core.logger import logger

# Module-level import alias so tests can patch `app.engines.validator.genai`
try:
    from google import genai  # noqa: F401  (used inside _run_semantic_validation)
except ImportError:
    genai = None  # type: ignore

class SemanticWarning(BaseModel):
    attribute: str = Field(description="The attribute name that generated a warning")
    warning: str = Field(description="The explanation of the warning")

class SemanticRejection(BaseModel):
    attribute: str = Field(description="The attribute name that was rejected")
    reason: str = Field(description="The explanation of the rejection")

class ValidationResponse(BaseModel):
    accepted: List[str] = Field(description="List of attributes that are semantically valid and accepted")
    warnings: List[SemanticWarning] = Field(default_factory=list, description="List of semantic warnings for attributes")
    rejected: List[SemanticRejection] = Field(default_factory=list, description="List of rejected attributes")
    confidence: float = Field(description="Confidence score for this validation, between 0.0 and 1.0")
    confidence_reason: str = Field(description="Explanation of the confidence score, exactly 1-2 sentences")


# ─────────────────────────────────────────────────────────────────────────────
# In-memory validation cache (thread-safe, TTL = 15 minutes)
# Keys: SHA-256 of (product_name + category + sorted_features + sorted_specs)
# On cache HIT the Gemini semantic call is skipped entirely.
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_LOCK = threading.Lock()
_VALIDATION_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS: int = 900  # 15 minutes


def _cache_key(product: Product, category: str) -> str:
    """Build a stable, content-addressed cache key."""
    attrs = sorted(product.features)
    specs = sorted(f"{k}:{v}" for k, v in product.specifications.items())
    payload = json.dumps({
        "product_name": product.product_name.lower().strip(),
        "category": category.lower().strip(),
        "attrs": attrs,
        "specs": specs,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    with _CACHE_LOCK:
        entry = _VALIDATION_CACHE.get(key)
        if entry and time.time() < entry["expires_at"]:
            return dict(entry["result"])  # return a copy
        if entry:
            del _VALIDATION_CACHE[key]
    return None


def _cache_set(key: str, result: Dict[str, Any]) -> None:
    with _CACHE_LOCK:
        _VALIDATION_CACHE[key] = {
            "result": result,
            "expires_at": time.time() + CACHE_TTL_SECONDS,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AttributeValidationEngine
# ─────────────────────────────────────────────────────────────────────────────

class AttributeValidationEngine:
    """
    Two-stage validation engine:
      Stage 1 — Basic structural validation (required fields, types, bounds).
      Stage 2 — AI semantic validation via Gemini (prompt-driven, no knowledge base).

    DEGRADED fallback contract:
      If Gemini is unavailable, accepted=[], semantic_status='DEGRADED',
      needs_manual_review=True.  Unvalidated features are NEVER silently passed
      as 'accepted'.  The orchestrator decides whether to surface them as
      'unvalidated' in the prompt builder.
    """

    @staticmethod
    def validate(
        product: Product,
        category: str = "General",
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        stage_start = time.time()
        logger.info(
            f"[Validator] Starting | product='{product.product_name}' | category='{category}'"
        )

        # ── Stage 1: Basic Validation ─────────────────────────────────────
        basic = AttributeValidationEngine._run_basic_validation(product)
        if basic["status"] == "FAIL":
            return {
                "status": "FAIL",
                "stage": "basic",
                "errors": basic["errors"],
                "semantic_status": "SKIPPED",
                "accepted": [],
                "warnings": [],
                "rejected": [],
                "validation_confidence": 0.0,
                "confidence_reason": "Basic validation failed — semantic validation was skipped.",
                "needs_manual_review": True,
                "decision_log": [],
                "validation_latency_seconds": round(time.time() - stage_start, 3),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "cache_hit": False,
            }

        # ── Stage 2: AI Semantic Validation ──────────────────────────────
        semantic = AttributeValidationEngine._run_semantic_validation(
            product, category, api_key
        )
        latency = round(time.time() - stage_start, 3)

        logger.info(
            f"[Validator] Complete | semantic_status={semantic['semantic_status']} | "
            f"accepted={len(semantic.get('accepted', []))} | "
            f"warnings={len(semantic.get('warnings', []))} | "
            f"rejected={len(semantic.get('rejected', []))} | "
            f"confidence={semantic.get('confidence', 0.0):.2f} | "
            f"latency={latency}s | cache_hit={semantic.get('cache_hit', False)}"
        )

        return {
            "status": "PASS",
            "stage": "semantic",
            "errors": [],
            "semantic_status": semantic["semantic_status"],
            "accepted": semantic.get("accepted", []),
            "warnings": semantic.get("warnings", []),
            "rejected": semantic.get("rejected", []),
            "validation_confidence": semantic.get("confidence", 0.0),
            "confidence_reason": semantic.get("confidence_reason", ""),
            "needs_manual_review": semantic.get("needs_manual_review", False),
            "decision_log": AttributeValidationEngine._build_decision_log(semantic),
            "validation_latency_seconds": latency,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "cache_hit": semantic.get("cache_hit", False),
        }

    # ── Stage 1 ───────────────────────────────────────────────────────────

    @staticmethod
    def _run_basic_validation(product: Product) -> Dict[str, Any]:
        errors: List[str] = []
        if not product.product_name or len(product.product_name.strip()) < 2:
            errors.append("Product name must be at least 2 characters long.")
        if not product.brand or not product.brand.strip():
            errors.append("Brand is a mandatory field.")
        if not product.material or not product.material.strip():
            errors.append("Material is a mandatory field.")
        if not product.color or not product.color.strip():
            errors.append("Color is a mandatory field.")
        if not product.target_audience or not product.target_audience.strip():
            errors.append("Target audience is a mandatory field.")
        if product.weight_grams is None:
            errors.append("Weight is missing or could not be parsed from the provided value.")
        elif product.weight_grams <= 0:
            errors.append("Weight must be a positive number.")
        if not product.features:
            errors.append("At least one product feature is required.")
        if not product.seo_keywords:
            errors.append("At least one SEO keyword is required.")
        return {"status": "FAIL" if errors else "PASS", "errors": errors}

    # ── Stage 2 ───────────────────────────────────────────────────────────

    @staticmethod
    def _run_semantic_validation(
        product: Product,
        category: str,
        api_key: Optional[str],
    ) -> Dict[str, Any]:
        """
        Calls Gemini to semantically validate all product attributes.

        Cache contract: results are cached for CACHE_TTL_SECONDS.

        Fallback contract: on ANY Gemini failure, returns DEGRADED status.
          accepted=[] — unvalidated features are NEVER presented as accepted.
          The caller (orchestrator) decides how to handle unvalidated features.
        """
        from app.engines.validator_prompt import SEMANTIC_VALIDATION_PROMPT

        # Build attribute list: features + spec keys/values
        attributes: List[str] = list(product.features)
        for k, v in product.specifications.items():
            attributes.append(f"{k}: {v}")

        if not attributes:
            # Nothing to validate semantically
            return {
                "semantic_status": "PASS",
                "accepted": [],
                "warnings": [],
                "rejected": [],
                "confidence": 1.0,
                "confidence_reason": "No attributes provided — semantic validation not required.",
                "needs_manual_review": False,
                "cache_hit": False,
            }

        # Cache look-up
        cache_key = _cache_key(product, category)
        cached = _cache_get(cache_key)
        if cached:
            logger.info(f"[Validator] Cache HIT (key={cache_key[:16]}…)")
            cached["cache_hit"] = True
            return cached

        # Build prompt
        prompt = SEMANTIC_VALIDATION_PROMPT.format(
            category=category,
            product_name=product.product_name,
            attributes_json=json.dumps(attributes, indent=2),
        )

        try:
            from app.core.config import settings
            from app.core.gemini_service import GeminiService
            from google.genai import types

            client = GeminiService.get_client()
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,          # deterministic for validation
                    response_mime_type="application/json",
                    response_schema=ValidationResponse,
                ),
            )

            raw = response.text
            try:
                parsed = json.loads(raw)
            except Exception as parse_exc:
                logger.error(f"[Validator] JSON parsing failed for raw response: {raw}")
                raise parse_exc

            # Validate response keys
            if not isinstance(parsed, dict):
                raise ValueError("Gemini returned non-dict JSON for validation.")

            result = {
                "semantic_status": "PASS",
                "accepted": [str(a) for a in parsed.get("accepted", [])],
                "warnings": [
                    {"attribute": str(w.get("attribute", "")), "warning": str(w.get("warning", ""))}
                    for w in parsed.get("warnings", [])
                    if isinstance(w, dict)
                ],
                "rejected": [
                    {"attribute": str(r.get("attribute", "")), "reason": str(r.get("reason", ""))}
                    for r in parsed.get("rejected", [])
                    if isinstance(r, dict)
                ],
                "confidence": float(parsed.get("confidence", 0.9)),
                "confidence_reason": str(parsed.get("confidence_reason", "")),
                "needs_manual_review": False,
                "cache_hit": False,
            }

            _cache_set(cache_key, result)
            return result

        except Exception as exc:
            logger.warning(
                f"[Validator] AI semantic validation FAILED ({type(exc).__name__}: {exc}). "
                f"Returning DEGRADED status — NOT silently accepting attributes."
            )
            # DEGRADED: accepted=[] intentionally.  Do NOT default to product.features.
            return {
                "semantic_status": "DEGRADED",
                "accepted": [],
                "warnings": [{
                    "attribute": "*",
                    "warning": (
                        f"AI semantic validation is currently unavailable "
                        f"({type(exc).__name__}). All attributes remain unvalidated. "
                        f"Manual review is required before publishing."
                    ),
                }],
                "rejected": [],
                "confidence": 0.0,
                "confidence_reason": (
                    f"Semantic validation service unavailable: {str(exc)[:150]}"
                ),
                "needs_manual_review": True,
                "cache_hit": False,
            }

    # ── Decision Log Builder ──────────────────────────────────────────────

    @staticmethod
    def _build_decision_log(semantic: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Builds a structured per-attribute decision log for frontend display.
        Each entry carries: attribute, source, decision, reason.
        """
        log: List[Dict[str, Any]] = []

        if semantic.get("semantic_status") == "DEGRADED":
            log.append({
                "attribute": "All Attributes",
                "source": "AI Validator",
                "decision": "DEGRADED",
                "reason": "Semantic validation service unavailable — manual review required.",
            })
            return log

        for attr in semantic.get("accepted", []):
            log.append({
                "attribute": attr,
                "source": "User Input",
                "decision": "ACCEPTED",
                "reason": None,
            })
        for w in semantic.get("warnings", []):
            attr = w.get("attribute", "*")
            log.append({
                "attribute": attr,
                "source": "AI Validator" if attr == "*" else "User Input",
                "decision": "ACCEPTED_WITH_WARNING",
                "reason": w.get("warning", ""),
            })
        for r in semantic.get("rejected", []):
            log.append({
                "attribute": r.get("attribute", ""),
                "source": "User Input",
                "decision": "REJECTED",
                "reason": r.get("reason", ""),
            })

        return log
