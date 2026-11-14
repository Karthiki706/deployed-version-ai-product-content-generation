import re
from typing import List, Dict, Any, Optional

from app.models.reports import (
    SEOReport,
    KeywordAuditEntry,
    SEOAuditSummary,
)
from app.core.logger import logger


class SEOValidationEngine:
    """
    Evaluates SEO features: keyword coverage, density, length compliance,
    and readability.  Produces a per-keyword audit table with an honest
    summary (never claims 'all optimized' when keywords are missing or thin).
    """

    def validate(
        self,
        content: Dict[str, Any],
        target_keywords: List[str],
        target_length: int,
    ) -> Dict[str, Any]:
        logger.info(f"[SEO] Validating keywords: {target_keywords}")

        title      = content.get("title", "")
        short_desc = content.get("short_description", "")
        long_desc  = content.get("long_description", "")
        meta_desc  = content.get("meta_description", "")

        full_text  = f"{title} {short_desc} {long_desc}".lower()
        words      = re.findall(r"\b\w+\b", full_text)
        total_words = len(words) if words else 1

        # ── 1. Keyword presence + counts ─────────────────────────────────
        missing_keywords: List[str] = []
        keyword_counts: Dict[str, int] = {}
        for kw in target_keywords:
            kw_clean = kw.lower().strip()
            occurrences = len(re.findall(rf"\b{re.escape(kw_clean)}\b", full_text))
            keyword_counts[kw] = occurrences
            if occurrences == 0:
                missing_keywords.append(kw)

        # ── 2. Keyword density ────────────────────────────────────────────
        keyword_density: Dict[str, float] = {
            kw: round((cnt / total_words) * 100, 2)
            for kw, cnt in keyword_counts.items()
        }

        # ── 3. Meta / title / description lengths ────────────────────────
        meta_len       = len(meta_desc)
        title_len      = len(title)
        long_desc_words = len(re.findall(r"\b\w+\b", long_desc))

        # ── 4. Keyword stuffing ───────────────────────────────────────────
        duplicate_keywords = [kw for kw, d in keyword_density.items() if d > 4.0]

        # ── 5. Readability ────────────────────────────────────────────────
        sentences = [s for s in re.split(r"[.!?]+", long_desc) if s.strip()]
        sentence_count   = len(sentences) if sentences else 1
        avg_sentence_len = long_desc_words / sentence_count

        if avg_sentence_len <= 15:
            readability_score = 100.0
        elif avg_sentence_len > 30:
            readability_score = 60.0
        else:
            readability_score = round(100.0 - (avg_sentence_len - 15) * 2.6, 2)

        # ── 6. Per-keyword audit table ────────────────────────────────────
        keyword_audit: List[KeywordAuditEntry] = []
        for kw in target_keywords:
            density = keyword_density.get(kw, 0.0)
            if kw in missing_keywords:
                entry = KeywordAuditEntry(
                    keyword=kw,
                    density=0.0,
                    status="MISSING",
                    recommendation=f"Add '{kw}' to the long description or title.",
                )
            elif density > 4.0:
                entry = KeywordAuditEntry(
                    keyword=kw,
                    density=density,
                    status="NEEDS_IMPROVEMENT",
                    recommendation=f"Density {density}% is above 4% — reduce usage to avoid keyword stuffing.",
                )
            elif density < 1.0:
                entry = KeywordAuditEntry(
                    keyword=kw,
                    density=density,
                    status="NEEDS_IMPROVEMENT",
                    recommendation=f"Density {density}% is below the 1% target — increase usage once or twice.",
                )
            else:
                entry = KeywordAuditEntry(
                    keyword=kw,
                    density=density,
                    status="OPTIMIZED",
                    recommendation=None,
                )
            keyword_audit.append(entry)

        # ── Audit summary (never claims 'all optimized' if any are not) ──
        optimized_count     = sum(1 for e in keyword_audit if e.status == "OPTIMIZED")
        needs_improve_count = sum(1 for e in keyword_audit if e.status == "NEEDS_IMPROVEMENT")
        missing_count       = sum(1 for e in keyword_audit if e.status == "MISSING")

        audit_summary = SEOAuditSummary(
            total=len(keyword_audit),
            optimized=optimized_count,
            needs_improvement=needs_improve_count,
            missing=missing_count,
        )

        # ── 7. Score ──────────────────────────────────────────────────────
        seo_score = 100.0
        if target_keywords:
            coverage_pct = (len(target_keywords) - len(missing_keywords)) / len(target_keywords)
            seo_score -= (1.0 - coverage_pct) * 50.0
        if meta_len > 160:
            seo_score -= 10.0
        elif meta_len < 40:
            seo_score -= 5.0
        if title_len > 80 or title_len < 10:
            seo_score -= 5.0
        if total_words > 0:
            length_diff_pct = abs(long_desc_words - target_length) / max(target_length, 1)
            if length_diff_pct > 0.25:
                seo_score -= 10.0
        if duplicate_keywords:
            seo_score -= len(duplicate_keywords) * 5.0
        if readability_score < 70:
            seo_score -= 5.0
        seo_score = max(0.0, round(seo_score, 2))

        report = SEOReport(
            missing_keywords=missing_keywords,
            keyword_density=keyword_density,
            meta_description_length=meta_len,
            long_description_length=long_desc_words,
            title_length=title_len,
            duplicate_keywords=duplicate_keywords,
            readability_score=readability_score,
            score=seo_score,
            keyword_audit=keyword_audit,
            audit_summary=audit_summary,
        )

        logger.info(
            f"[SEO] Complete | score={report.score} | "
            f"optimized={optimized_count} | needs_improvement={needs_improve_count} | missing={missing_count}"
        )
        return report.model_dump()
