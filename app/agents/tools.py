"""
Tools exposed to the agentic Writer.

Each tool wraps an existing deterministic engine, so the LLM can *decide* when to
verify its own draft, while the same engines still act as the final, non-negotiable
audit gate later in the graph ("agents decide, deterministic validators verify").

Tools are plain Python callables with type hints and docstrings; the Gemini SDK turns
them into function declarations automatically. They are built per request through
`build_tools(ctx)` so every tool closes over that request's product and validation state.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.core.logger import logger
from app.engines.brand import BrandComplianceEngine
from app.engines.hallucination import HallucinationDetectionEngine
from app.engines.prompt_registry import PromptRegistry
from app.engines.seo import SEOValidationEngine
from app.models.product import Product


@dataclass
class ToolContext:
    """Per-request facts the tools need."""
    product: Product
    category: str
    validation_report: Dict[str, Any] = field(default_factory=dict)
    accepted_attributes: List[str] = field(default_factory=list)
    rejected_attributes: List[str] = field(default_factory=list)


def parse_draft(draft_json: str) -> Dict[str, Any]:
    """Parse a draft passed by the model. Tolerates markdown fences and surrounding prose."""
    text = (draft_json or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("draft_json is not valid JSON")
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("draft_json must be a JSON object")
    return data


def build_tools(ctx: ToolContext) -> List[Callable]:
    """Create the tool set bound to one request's context."""

    def get_verified_facts() -> dict:
        """Return the verified product facts the copy may use, plus attributes that are forbidden.

        Call this FIRST, before drafting. Only facts returned here may appear in the copy.
        """
        p = ctx.product
        facts = {
            "product_name": p.product_name,
            "brand": p.brand,
            "category": ctx.category,
            "material": p.material,
            "color": p.color,
            "weight_grams": p.weight_grams,
            "dimensions": p.dimensions,
            "features_allowed": ctx.accepted_attributes or p.features,
            "specifications": p.specifications,
            "package_contents": p.package_contents,
            "warranty": p.warranty,
            "target_audience": p.target_audience,
            "seo_keywords_required": p.seo_keywords,
            "brand_voice": p.brand_voice,
            "tone": p.tone,
            "target_description_words": p.description_length,
            "FORBIDDEN_attributes_never_mention": ctx.rejected_attributes,
        }
        return {k: v for k, v in facts.items() if v not in (None, "", [], {})}

    def get_category_guidance() -> dict:
        """Return the writing guidance and structure template for this product's category."""
        guidance, version = PromptRegistry.get_template(ctx.category, "v1")
        return {"category": ctx.category, "template_version": version, "guidance": guidance}

    def check_brand_rules(draft_json: str) -> dict:
        """Check a draft against brand rules: banned marketing words, tone and style.

        Args:
            draft_json: The full draft as a JSON string with keys title, short_description,
                long_description, bullet_points, meta_description, seo_keywords_used,
                specification_summary.
        """
        try:
            draft = parse_draft(draft_json)
        except ValueError as e:
            return {"error": str(e)}
        rep = BrandComplianceEngine().check(draft, ctx.product.brand_voice, ctx.product.tone)
        violations = rep.get("violations", [])
        return {
            "passed": not violations,
            "banned_words_found": violations,
            "tone_check": rep.get("tone_check"),
            "fix": "Remove or rephrase every banned word." if violations else "",
        }

    def check_seo(draft_json: str) -> dict:
        """Check that all required SEO keywords appear naturally in the draft.

        Args:
            draft_json: The full draft as a JSON string (same shape as check_brand_rules).
        """
        try:
            draft = parse_draft(draft_json)
        except ValueError as e:
            return {"error": str(e)}
        rep = SEOValidationEngine().validate(draft, ctx.product.seo_keywords, ctx.product.description_length)
        missing = rep.get("missing_keywords", [])
        return {
            "passed": not missing,
            "missing_keywords": missing,
            "keyword_density": rep.get("keyword_density"),
            "fix": "Work each missing keyword into the title or descriptions naturally." if missing else "",
        }

    def check_hallucination(draft_json: str) -> dict:
        """Check a draft for claims not supported by the verified product facts.

        Args:
            draft_json: The full draft as a JSON string (same shape as check_brand_rules).
        """
        try:
            draft = parse_draft(draft_json)
        except ValueError as e:
            return {"error": str(e)}
        rep = HallucinationDetectionEngine().detect(
            draft, ctx.product, validation_report=ctx.validation_report
        )
        section_c = rep.get("section_c") or {}
        unsupported = [
            c.get("claim", "") if isinstance(c, dict) else str(c)
            for c in section_c.get("unsupported_claims", [])
        ] or list(rep.get("unsupported_claims", []))
        wrong_fields = [v.get("issue", "") for v in rep.get("hallucinated_fields", [])]
        return {
            "passed": not unsupported and not wrong_fields,
            "score": rep.get("score"),
            "unsupported_claims": unsupported,
            "contradicting_fields": wrong_fields,
            "fix": "Delete each unsupported claim; state only values from get_verified_facts." if (unsupported or wrong_fields) else "",
        }

    return [get_verified_facts, get_category_guidance, check_brand_rules, check_seo, check_hallucination]


def summarize_result(result: Any, limit: int = 300) -> str:
    """Compact one-line preview of a tool result for the UI trace."""
    try:
        text = json.dumps(result, default=str)
    except Exception:
        text = str(result)
    return text if len(text) <= limit else text[:limit] + "…"
