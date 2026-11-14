import json
from typing import List, Optional

from app.models.product import Product
from app.core.logger import logger


class DynamicPromptBuilder:
    """
    Orchestrates prompt construction by assembling distinct logical components.

    Enhancements:
    - Only includes semantically validated (accepted) attributes in PRODUCT DATA.
    - When semantic validation was DEGRADED, includes raw features labelled as UNVALIDATED.
    - Injects a FORBIDDEN CLAIMS section for all rejected attributes.
    - Instructs the model never to infer or assume missing specifications.
    """

    def get_instruction(self) -> str:
        return "Task: Generate professional product merchandising content based on provided validated attributes."

    def get_business_rules(self) -> str:
        return (
            "Business Rules & Constraints:\n"
            "- Highlight real, validated attributes and factual benefit statements.\n"
            "- Do NOT invent specifications, numbers, features, or metrics not explicitly listed in Product Data.\n"
            "- Never infer or assume missing values. If a detail (e.g. battery capacity, chip model, screen size) "
            "is not provided in the Product Data, do NOT include it.\n"
            "- Only describe what is confirmed in the validated attribute list.\n"
            "- Ensure description lengths match desired targets.\n"
            "- Avoid restricted words such as 'best', 'cheapest', 'guaranteed', 'perfect'.\n"
            "- Rely only on provided and validated facts to prevent hallucinated claims."
        )

    def get_category_context(self, category_guidance: str) -> str:
        return f"Category Guidelines:\n{category_guidance}"

    def get_brand_voice_guidance(self, brand_voice: str, tone: str) -> str:
        return f"Tone & Brand Voice:\n- Brand Voice: {brand_voice}\n- Tone: {tone}"

    def get_seo_guidance(self, keywords: List[str]) -> str:
        return f"SEO Guidelines:\n- Target keywords to integrate naturally: {', '.join(keywords)}"

    def get_output_schema(self) -> str:
        schema = {
            "title": "Descriptive, SEO-optimized title.",
            "short_description": "Short summary of the product (~50 words).",
            "long_description": "Detailed description (~150-250 words depending on target).",
            "bullet_points": ["List of key product features (bullet points)."],
            "meta_description": "SEO meta description under 160 characters.",
            "seo_keywords_used": ["List of target keywords that were actually used in the text."],
            "specification_summary": {
                "additional_spec_key": "spec_value"
            }
        }
        return (
            "Output Format:\n"
            "Return ONLY a valid JSON object matching this structure "
            "(specification_summary MUST be a flat key-value dictionary of specs, "
            "NOT a markdown string or table):\n"
            f"{json.dumps(schema, indent=2)}"
        )

    def get_product_data(
        self,
        product: Product,
        enrichment_tags: List[str],
        accepted_features: Optional[List[str]] = None,
        semantic_status: str = "PASS",
    ) -> str:
        # Use semantically validated features if available; otherwise raw features
        if accepted_features is not None and semantic_status == "PASS":
            features_to_use = accepted_features
            features_note   = "(Semantically validated)"
        elif semantic_status == "DEGRADED":
            features_to_use = product.features
            features_note   = "(UNVALIDATED — semantic validation unavailable; use with caution)"
        else:
            features_to_use = product.features
            features_note   = ""

        data = {
            "product_name": product.product_name,
            "brand": product.brand,
            "category": product.category,
            "subcategory": product.subcategory,
            "material": product.material,
            "color": product.color,
            "weight": product.weight_raw or (f"{product.weight_grams} g" if product.weight_grams else None),
            "dimensions": product.dimensions,
            f"features {features_note}".strip(): features_to_use,
            "specifications": product.specifications,
            "target_audience": product.target_audience,
            "warranty": product.warranty,
            "package_contents": product.package_contents,
            "enrichment_tags": enrichment_tags,
            "description_length_target": f"approx {product.description_length} words",
        }
        # Filter out None values and empty structures
        data_clean = {k: v for k, v in data.items() if v is not None and v != {} and v != []}
        return f"Product Source Attributes:\n{json.dumps(data_clean, indent=2)}"

    def get_forbidden_claims(self, rejected_attributes: List[str]) -> str:
        if not rejected_attributes:
            return ""
        lines = "\n".join([f"  • {a}" for a in rejected_attributes])
        return (
            "⛔ FORBIDDEN CLAIMS — DO NOT MENTION THESE:\n"
            "The following attributes were REJECTED during semantic validation "
            "because they do not belong to this product category.\n"
            "DO NOT mention, imply, reference, or infer them in ANY part of your output:\n"
            f"{lines}"
        )

    def build(
        self,
        product: Product,
        category_guidance: str,
        enrichment_tags: List[str],
        accepted_features: Optional[List[str]] = None,
        rejected_attributes: Optional[List[str]] = None,
        semantic_status: str = "PASS",
    ) -> str:
        logger.info(f"[PromptBuilder] Building prompt for '{product.product_name}' | "
                    f"accepted={len(accepted_features or [])} | rejected={len(rejected_attributes or [])}")

        rejected_attributes = rejected_attributes or []
        forbidden_section = self.get_forbidden_claims(rejected_attributes)

        parts = [
            "=== INSTRUCTION ===",
            self.get_instruction(),
            "",
            "=== PRODUCT DATA ===",
            self.get_product_data(product, enrichment_tags, accepted_features, semantic_status),
            "",
            "=== CATEGORY CONTEXT ===",
            self.get_category_context(category_guidance),
            "",
            "=== BRAND VOICE ===",
            self.get_brand_voice_guidance(product.brand_voice, product.tone),
            "",
            "=== SEO ===",
            self.get_seo_guidance(product.seo_keywords),
            "",
            "=== BUSINESS RULES ===",
            self.get_business_rules(),
        ]

        if forbidden_section:
            parts += [
                "",
                "=== FORBIDDEN CLAIMS ===",
                forbidden_section,
            ]

        parts += [
            "",
            "=== OUTPUT JSON SCHEMA ===",
            self.get_output_schema(),
        ]

        return "\n".join(parts)
