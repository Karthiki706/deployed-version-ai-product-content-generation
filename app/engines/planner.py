from typing import Dict, Any
from app.models.product import Product

class PlannerEngine:
    """Deterministic copywriting planner analyzing inputs and structuring the plan specifications."""
    
    @staticmethod
    def plan(product: Product) -> Dict[str, Any]:
        category = product.category or "Default"
        
        if category.lower() in ["electronics", "appliances"]:
            writing_strategy = "Specs-first technical breakdown focusing on validated parameters."
            bullet_style = "Technical metrics and spec-first benefits."
            content_structure = "Title -> Tech Intro -> Technical Bullet Points -> Spec Summary"
        elif category.lower() in ["footwear", "fashion", "apparel"]:
            writing_strategy = "Style and materials focus emphasizing fit, feel, and design details."
            bullet_style = "Stylistic feature and benefit points."
            content_structure = "Title -> Style Narrative -> Fit/Comfort highlights -> Specifications"
        else:
            writing_strategy = "General benefit statements focusing on key validated attributes."
            bullet_style = "General benefits and uses list."
            content_structure = "Title -> Short Intro -> Benefit Bullets -> Summary"
            
        brand_voice = product.brand_voice or "Professional"
        tone = product.tone or "Expert"
        selected_brand_style = f"Brand Voice: {brand_voice}"
        
        return {
            "writing_strategy": writing_strategy,
            "description_length_target": product.description_length,
            "selected_tone": tone,
            "selected_brand_style": selected_brand_style,
            "seo_keyword_strategy": "Distribute target keywords naturally across title, short description, and specifications summary.",
            "bullet_style": bullet_style,
            "content_structure": content_structure,
        }
