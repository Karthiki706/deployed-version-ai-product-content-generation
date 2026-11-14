import json
from typing import Dict, Any, List, Optional

from app.models.product import Product
from app.models.reports import (
    HallucinationReport,
    HallucinationViolation,
    HallucinationSectionA,
    HallucinationSectionB,
    HallucinationSectionC,
    HallucinationSectionD,
    UnsupportedClaim,
)
from app.engines.comparison_engine import ComparisonEngine
from app.core.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Severity / Impact classification helpers
# ─────────────────────────────────────────────────────────────────────────────

_HIGH_SEVERITY_PATTERNS = [
    "warranty", "price", "weight", "dimension", "specification",
    "connectivity standard", "5g", "4g", "lte", "model number",
    "battery capacity", "chip", "processor", "resolution",
]
_MEDIUM_SEVERITY_PATTERNS = [
    "material", "battery", "feature", "connectivity", "sensor",
    "bluetooth", "wifi", "nfc", "gps", "display",
]
_COMPLIANCE_PATTERNS = ["warranty", "regulation", "certification", "safety", "standard"]
_TRUST_PATTERNS = ["specification", "weight", "dimension", "price", "model", "processor"]
_SEO_PATTERNS = ["keyword", "description", "title"]
_SAFETY_PATTERNS = ["safety", "regulation", "certification", "hazard"]


def _classify_severity(claim: str) -> str:
    cl = claim.lower()
    if any(p in cl for p in _HIGH_SEVERITY_PATTERNS):
        return "High"
    if any(p in cl for p in _MEDIUM_SEVERITY_PATTERNS):
        return "Medium"
    return "Low"


def _classify_impact(claim: str) -> List[str]:
    cl = claim.lower()
    impact: List[str] = []
    if any(p in cl for p in _TRUST_PATTERNS):
        impact.append("Customer Trust")
    if any(p in cl for p in _COMPLIANCE_PATTERNS):
        impact.append("Compliance")
    if any(p in cl for p in _SEO_PATTERNS):
        impact.append("SEO")
    if any(p in cl for p in _SAFETY_PATTERNS):
        impact.append("Safety")
    if not impact:
        impact.append("Content Accuracy")
    return impact


# ─────────────────────────────────────────────────────────────────────────────
# HallucinationDetectionEngine
# ─────────────────────────────────────────────────────────────────────────────

class HallucinationDetectionEngine:
    """
    Production-grade hallucination engine with a 4-section structured report.

    Section A — Category Validation context (detected category, confidence).
    Section B — Rejected user attributes (invalid input, NOT AI hallucinations).
    Section C — Unsupported AI claims (genuine LLM hallucinations, with severity & impact).
    Section D — Summary distinguishing user-input errors from AI hallucinations.
    """

    def detect(
        self,
        generated_content: Dict[str, Any],
        original_product: Product,
        validation_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logger.info(f"[Hallucination] Starting for product: {original_product.product_name}")

        # ── Aggregate generated text ─────────────────────────────────────
        text_blob = " ".join([
            str(generated_content.get("title", "")),
            str(generated_content.get("short_description", "")),
            str(generated_content.get("long_description", "")),
            " ".join(generated_content.get("bullet_points", [])),
        ])

        specs_dict = generated_content.get("specification_summary", {})
        if isinstance(specs_dict, str):
            try:
                specs_dict = json.loads(specs_dict)
            except Exception:
                specs_dict = {}
        if not isinstance(specs_dict, dict):
            specs_dict = {}

        # ── Run comparison engine ────────────────────────────────────────
        violations_raw, missing, unsupported_raw, comp_report, score = ComparisonEngine.compare(
            original_product, text_blob, specs_dict
        )

        # ── Build legacy fields (backward compat) ────────────────────────
        hallucinated_violations = [
            HallucinationViolation(field=v["field"], issue=v["issue"])
            for v in violations_raw
        ]
        confidence = score / 100.0

        # ── Section A: Category Validation context ───────────────────────
        vr = validation_report or {}
        section_a = HallucinationSectionA(
            detected_category=vr.get("_detected_category", ""),
            validation_confidence=float(vr.get("validation_confidence", 1.0)),
            validation_status=vr.get("semantic_status", "PASS") if vr else "PASS",
        )

        # ── Section B: Rejected user attributes (NOT AI hallucinations) ──
        rejected_attrs = vr.get("rejected", []) if vr else []
        section_b = HallucinationSectionB(
            rejected_attributes=[
                {"attribute": r.get("attribute", ""), "reason": r.get("reason", "")}
                for r in rejected_attrs
            ]
        )

        # ── Section C: Unsupported AI claims (genuine hallucinations) ────
        # Build a set of rejected attribute texts so we don't double-count them
        rejected_texts = {r.get("attribute", "").lower() for r in rejected_attrs}

        unsupported_structured: List[UnsupportedClaim] = []
        for raw_claim in unsupported_raw:
            # Skip if this claim matches a user-rejected attribute
            if any(rt in raw_claim.lower() for rt in rejected_texts if rt):
                continue
            severity = _classify_severity(raw_claim)
            impact = _classify_impact(raw_claim)
            unsupported_structured.append(
                UnsupportedClaim(
                    claim=raw_claim,
                    reason="This claim is not supported by any validated input attribute.",
                    severity=severity,
                    impact=impact,
                )
            )
        section_c = HallucinationSectionC(unsupported_claims=unsupported_structured)

        # ── Section D: Summary ───────────────────────────────────────────
        ai_count = len(unsupported_structured)
        high_count = sum(1 for u in unsupported_structured if u.severity == "High")
        rejected_count = len(rejected_attrs)

        if ai_count == 0 and rejected_count == 0:
            status = "PASS"
            summary = "No hallucinations detected. All AI-generated claims are fully supported by validated input attributes."
        elif ai_count == 0 and rejected_count > 0:
            status = "PASS"
            summary = (
                f"No AI hallucinations detected. "
                f"{rejected_count} invalid user attribute(s) were correctly rejected before generation — "
                f"none of them appeared in the generated output."
            )
        else:
            status = "FAIL"
            summary = (
                f"{ai_count} unsupported AI claim(s) detected "
                f"({high_count} high-severity). "
                f"{rejected_count} user attribute(s) were rejected pre-generation."
            )

        section_d = HallucinationSectionD(
            hallucination_status=status,
            summary=summary,
            rejected_input_count=rejected_count,
            ai_hallucination_count=ai_count,
            ai_hallucination_high_severity_count=high_count,
        )

        # ── Assemble final report ────────────────────────────────────────
        # Legacy unsupported_claims stays as List[str] for orchestrator compatibility
        report = HallucinationReport(
            hallucinated_fields=hallucinated_violations,
            missing_fields=missing,
            unsupported_claims=unsupported_raw,       # legacy flat list
            confidence=confidence,
            score=score,
            comparison_report=comp_report,
            section_a=section_a,
            section_b=section_b,
            section_c=section_c,
            section_d=section_d,
        )

        logger.info(
            f"[Hallucination] Complete | score={report.score} | "
            f"ai_claims={ai_count} | high_severity={high_count} | rejected_inputs={rejected_count}"
        )
        return report.model_dump()
