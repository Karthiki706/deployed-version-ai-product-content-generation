from typing import Dict, Any, List, Optional
from app.models.reports import HallucinationReport, SEOReport, BrandReport

class ReflectionEngine:
    """Analyzes validation reports to compile a structured list of weaknesses."""
    
    @staticmethod
    def reflect(
        h_model: Optional[HallucinationReport],
        s_model: Optional[SEOReport],
        b_model: Optional[BrandReport],
        json_error: Optional[str] = None
    ) -> Dict[str, Any]:
        weaknesses: List[str] = []
        
        if json_error:
            weaknesses.append(f"JSON Validation: Output syntax is not valid JSON: {json_error}")
            
        if h_model:
            h_rep = h_model.model_dump()
            for v in h_rep.get("hallucinated_fields", []):
                weaknesses.append(f"Hallucination: {v['issue']}")
            section_c = h_rep.get("section_c") or {}
            for claim in section_c.get("unsupported_claims", []):
                if isinstance(claim, dict):
                    weaknesses.append(
                        f"Unsupported AI claim [{claim.get('severity','?')} severity]: {claim.get('claim','')}"
                    )
            for claim in h_rep.get("unsupported_claims", []):
                if isinstance(claim, str):
                    weaknesses.append(f"Unsupported claim: {claim}")
                    
        if s_model:
            s_rep = s_model.model_dump()
            if s_rep.get("missing_keywords"):
                weaknesses.append(
                    f"SEO Compliance: Target keywords missing from copywriting: {s_rep['missing_keywords']}"
                )
                
        if b_model:
            b_rep = b_model.model_dump()
            if b_rep.get("violations"):
                weaknesses.append(
                    f"Brand Compliance: Prohibited restricted keywords found: {b_rep['violations']}"
                )
                
        if weaknesses:
            summary = "Compliance validation failed. Directives to correct:\n" + \
                      "\n".join([f"  • {w}" for w in weaknesses])
        else:
            summary = "Auditing passed. Zero violations found."
            
        return {
            "reflection_summary": summary,
            "weaknesses": weaknesses,
        }
