from typing import Dict, Any, Optional
from app.core.constants import CATEGORY_MAP
from app.core.logger import logger

class CategoryClassificationEngine:
    """Deterministic classification of products into high-level categories and subcategories based on naming conventions."""
    
    def classify(self, product_name: str, user_category: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"Classifying product: {product_name} (User category input: {user_category})")
        
        # If user explicitly provided a category, check if it fits our maps
        if user_category:
            user_cat_clean = user_category.strip().title()
            if user_cat_clean in CATEGORY_MAP:
                # Still check name keywords under this category to find a subcategory
                name_lower = product_name.lower()
                for kw in CATEGORY_MAP[user_cat_clean]:
                    if kw in name_lower:
                        logger.info(f"Using valid user category: {user_cat_clean} with detected subcategory: {kw.title()}")
                        return {"category": user_cat_clean, "subcategory": kw.title(), "confidence": 1.0}
                
                logger.info(f"Using valid user-provided category: {user_cat_clean}")
                return {"category": user_cat_clean, "subcategory": None, "confidence": 1.0}
            else:
                # Try to classify based on combined text
                name_lower = f"{product_name} {user_category}".lower()
        else:
            name_lower = product_name.lower()
            
        for category, keywords in CATEGORY_MAP.items():
            for kw in keywords:
                if kw in name_lower:
                    logger.info(f"Classified category: {category}, subcategory: {kw.title()} (confidence: 1.0) based on name keywords")
                    return {"category": category, "subcategory": kw.title(), "confidence": 1.0}
                
        # If not classified but user gave category, return it with 0.8 confidence
        if user_category:
            logger.info(f"Falling back to user-provided category: {user_category.strip().title()}")
            return {"category": user_category.strip().title(), "subcategory": None, "confidence": 0.8}
            
        logger.info("Category classification fallback to 'General' (confidence: 0.5)")
        return {"category": "General", "subcategory": None, "confidence": 0.5}