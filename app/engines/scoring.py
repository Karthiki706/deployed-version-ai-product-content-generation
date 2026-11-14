from typing import Dict, Any, Optional

from app.models.reports import (
    QualityReport,
    QualityBreakdown,
    ScoreComponent,
    SafetyBreakdown,
    SafetySubScores,
)
from app.core.logger import logger


def _risk_level(issue_count: int) -> str:
    if issue_count == 0:
        return "Low"
    if issue_count <= 3:
        return "Medium"
    if issue_count <= 6:
        return "High"
    return "Critical"


class QualityScoringEngine:
    """
    Aggregates individual engine scores into a weighted overall quality score.
    Produces a fully explainable breakdown with reason + penalty per component.
    Also computes:
      - validation_confidence  (from semantic validator)
      - generation_confidence  (derived from hallucination score + comparison confidence)
      - safety_score           (composite of 5 safety dimensions)
    """

    def calculate(
        self,
        h_report: Dict[str, Any],
        s_report: Dict[str, Any],
        b_report: Dict[str, Any],
        validation_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        vr = validation_report or {}

        # ── Raw component scores ─────────────────────────────────────────
        h_score = float(h_report.get("score", 100.0))
        s_score = float(s_report.get("score", 100.0))
        b_score = float(b_report.get("score", 100.0))

        # ── Validation confidence ────────────────────────────────────────
        validation_confidence = float(vr.get("validation_confidence", 1.0))
        semantic_status       = vr.get("semantic_status", "PASS")
        rejected_count        = len(vr.get("rejected", []))
        warned_count          = len(vr.get("warnings", []))
        needs_manual_review   = vr.get("needs_manual_review", False)

        # ── Generation confidence ────────────────────────────────────────
        # Derived from the comparison engine confidence (hallucination confidence)
        h_confidence = float(h_report.get("confidence", 1.0))
        # Also consider unsupported AI claims
        section_d    = h_report.get("section_d") or {}
        ai_claims    = int(section_d.get("ai_hallucination_count", len(h_report.get("hallucinated_fields", []))))
        gen_confidence_raw = h_confidence * max(0.0, 1.0 - ai_claims * 0.05)
        generation_confidence = round(min(1.0, max(0.0, gen_confidence_raw)), 3)

        # ── Breakdown: Generation Quality ────────────────────────────────
        gen_penalty = 0.0
        if ai_claims > 0:
            gen_penalty = min(30.0, ai_claims * 10.0)
        gen_score = max(0.0, round(h_score - gen_penalty, 2))
        gen_reason = (
            f"Content well-formed with no AI hallucinations detected."
            if ai_claims == 0
            else f"{ai_claims} unsupported AI claim(s) detected in generated content."
        )

        # ── Breakdown: SEO ───────────────────────────────────────────────
        seo_penalty = max(0.0, round(100.0 - s_score, 2))
        audit_summary = s_report.get("audit_summary") or {}
        missing_kw    = len(s_report.get("missing_keywords", []))
        seo_reason = (
            f"All {audit_summary.get('total', 0)} keywords optimized."
            if missing_kw == 0 and audit_summary.get("needs_improvement", 0) == 0
            else (
                f"{audit_summary.get('optimized', 0)} of {audit_summary.get('total', 0)} keywords optimized; "
                f"{audit_summary.get('needs_improvement', 0)} need improvement; "
                f"{missing_kw} missing."
            )
        )

        # ── Breakdown: Safety ────────────────────────────────────────────
        section_c        = h_report.get("section_c") or {}
        unsupported_list = section_c.get("unsupported_claims", [])
        high_sev_count   = sum(
            1 for u in unsupported_list
            if (u.get("severity") if isinstance(u, dict) else getattr(u, "severity", "")) == "High"
        )
        brand_violations = len(b_report.get("violations", []))
        total_issues     = rejected_count + ai_claims + brand_violations

        # Content safety: no unsupported claims
        content_safety   = max(0.0, 100.0 - ai_claims * 15.0)
        # Brand safety: no brand violations
        brand_safety     = b_score
        # Category safety: no rejected attributes
        category_safety  = max(0.0, 100.0 - rejected_count * 12.0)
        # Validation safety: penalise DEGRADED or high warned count
        validation_safety = (
            0.0 if semantic_status == "DEGRADED"
            else max(0.0, 100.0 - warned_count * 5.0)
        )
        # Policy safety: no high-severity hallucinations
        policy_safety    = max(0.0, 100.0 - high_sev_count * 20.0)

        safety_raw = (
            content_safety * 0.30
            + brand_safety * 0.20
            + category_safety * 0.20
            + validation_safety * 0.15
            + policy_safety * 0.15
        )
        safety_score   = round(max(0.0, min(100.0, safety_raw)), 2)
        safety_penalty = max(0.0, round(100.0 - safety_score, 2))
        safety_reason  = (
            "No safety issues detected across all five safety dimensions."
            if total_issues == 0 and semantic_status == "PASS"
            else (
                f"Issues detected — "
                f"{rejected_count} invalid attribute(s), "
                f"{ai_claims} AI claim(s), "
                f"{brand_violations} brand violation(s)."
                + (" Semantic validation DEGRADED — manual review required." if semantic_status == "DEGRADED" else "")
            )
        )
        risk = _risk_level(total_issues + (1 if semantic_status == "DEGRADED" else 0))

        # ── Breakdown: Brand Compliance ──────────────────────────────────
        brand_penalty = max(0.0, round(100.0 - b_score, 2))
        brand_reason  = (
            "No restricted terms used. Brand voice and tone compliant."
            if not brand_violations
            else f"{brand_violations} restricted term(s) found: {', '.join(b_report.get('violations', []))}."
        )

        # ── Breakdown: Validation ────────────────────────────────────────
        val_penalty  = rejected_count * 5.0 + (30.0 if semantic_status == "DEGRADED" else 0.0)
        val_score    = max(0.0, round(100.0 - val_penalty, 2))
        val_reason   = (
            f"Semantic validation DEGRADED — unvalidated attributes not filtered."
            if semantic_status == "DEGRADED"
            else (
                f"All attributes validated. "
                f"{rejected_count} rejected, {warned_count} warnings."
                if rejected_count or warned_count
                else "All attributes passed semantic validation with no warnings."
            )
        )

        # ── Breakdown: Hallucination ─────────────────────────────────────
        hal_penalty  = ai_claims * 10.0
        hal_score    = max(0.0, round(h_score, 2))
        hal_reason   = (
            "No unsupported AI-generated claims detected."
            if ai_claims == 0
            else f"{ai_claims} unsupported claim(s) found; {high_sev_count} high-severity."
        )

        # ── Overall (weighted: H=40%, S=25%, B=15%, Safety=20%) ─────────
        # Validation confidence penalty: if DEGRADED, cap overall at 70
        overall = (
            h_score    * 0.40
            + s_score  * 0.25
            + b_score  * 0.15
            + safety_score * 0.20
        )
        overall = round(overall, 2)
        if semantic_status == "DEGRADED":
            overall = min(overall, 70.0)

        logger.info(
            f"[Quality] overall={overall} | h={h_score} | seo={s_score} | brand={b_score} | "
            f"safety={safety_score} | val_conf={validation_confidence:.2f} | "
            f"gen_conf={generation_confidence:.2f} | risk={risk}"
        )

        # ── Assemble report ───────────────────────────────────────────────
        breakdown = QualityBreakdown(
            generation_quality=ScoreComponent(score=gen_score, reason=gen_reason, penalty=gen_penalty),
            seo=ScoreComponent(score=s_score, reason=seo_reason, penalty=seo_penalty),
            safety=SafetyBreakdown(
                score=safety_score,
                reason=safety_reason,
                penalty=safety_penalty,
                risk_level=risk,
                sub_scores=SafetySubScores(
                    content_safety=round(content_safety, 2),
                    brand_safety=round(brand_safety, 2),
                    category_safety=round(category_safety, 2),
                    validation_safety=round(validation_safety, 2),
                    policy_safety=round(policy_safety, 2),
                ),
            ),
            brand_compliance=ScoreComponent(score=b_score, reason=brand_reason, penalty=brand_penalty),
            validation=ScoreComponent(score=val_score, reason=val_reason, penalty=val_penalty),
            hallucination=ScoreComponent(score=hal_score, reason=hal_reason, penalty=hal_penalty),
        )

        report = QualityReport(
            seo_score=s_score,
            brand_score=b_score,
            hallucination_score=h_score,
            overall_quality=overall,
            confidence=round(h_confidence * (overall / 100.0), 3),
            safety_score=safety_score,
            validation_confidence=round(validation_confidence, 3),
            generation_confidence=generation_confidence,
            breakdown=breakdown,
            risk_level=risk,
        )

        return report.model_dump()
