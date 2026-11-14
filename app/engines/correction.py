import json
from typing import List, Dict, Any

from app.core.logger import logger


class CorrectionPromptBuilder:
    """
    Constructs a self-correction prompt that feeds specific error details back
    to the LLM.

    Enhancements over the original:
    - Lists rejected user attributes by name so the LLM never re-introduces them.
    - Provides the validated (accepted) attribute list as the authoritative source.
    - Instructs the model never to infer or assume missing specifications.
    """

    def build(
        self,
        original_prompt: str,
        previous_output: Dict[str, Any],
        errors: List[str],
        rejected_attributes: List[str] | None = None,
        accepted_attributes: List[str] | None = None,
    ) -> str:
        logger.info(f"[Correction] Building correction prompt with {len(errors)} errors.")

        formatted_errors = "\n".join([f"  - {e}" for e in errors])

        # Build FORBIDDEN section if rejected attrs are provided
        forbidden_section = ""
        if rejected_attributes:
            forbidden_list = "\n".join([f"  • {a}" for a in rejected_attributes])
            forbidden_section = f"""
=========================================
⛔ FORBIDDEN CLAIMS — DO NOT MENTION THESE:
=========================================
The following attributes were REJECTED during semantic validation because they
do not belong to this product category.
DO NOT mention, imply, reference, or infer them in any part of your response:
{forbidden_list}
"""

        # Build AUTHORITATIVE ATTRIBUTES section
        auth_section = ""
        if accepted_attributes:
            auth_list = "\n".join([f"  ✓ {a}" for a in accepted_attributes])
            auth_section = f"""
=========================================
✅ AUTHORITATIVE VALIDATED ATTRIBUTES:
=========================================
Only the following attributes have been semantically validated and approved
for use. Base your content exclusively on these:
{auth_list}

IMPORTANT: Never infer, assume, or invent specifications that are not in the
list above. If a detail (e.g. battery capacity, processor model, weight) is
not in the validated attributes, do NOT include it.
"""

        correction_prompt = f"""{original_prompt}
{forbidden_section}
{auth_section}
=========================================
⚠️ CRITICAL ERROR CORRECTION REQUIRED:
=========================================
Your previous output failed our quality gate checks.
Review the errors below and correct them in your new response.

PRESERVE all correct content from your previous response.
Modify ONLY the fields listed in the DETECTED ERRORS section.

DETECTED ERRORS:
{formatted_errors}

YOUR PREVIOUS GENERATED OUTPUT:
{json.dumps(previous_output, indent=2)}

CORRECTION INSTRUCTIONS:
1. Modify ONLY the fields listed in DETECTED ERRORS above.
2. For all other fields (title, descriptions, etc.) that did not fail, preserve
   them EXACTLY as they were — do not change a single word.
3. If specifications, weights, or dimensions failed, correct those specific keys
   in 'specification_summary' to match the validated input facts.
4. If SEO keywords are missing, integrate them naturally into the long description
   or short description only — do not alter other fields.
5. If restricted brand terms were used, replace them with neutral alternatives.
6. Never infer or assume missing values. Only describe what is in the validated
   attribute list.
7. Return strictly the updated JSON matching the output schema.
"""
        return correction_prompt
