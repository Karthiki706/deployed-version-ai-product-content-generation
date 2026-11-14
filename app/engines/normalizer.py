import re
from typing import List, Dict, Any, Union
from app.models.product import Product
from app.core.logger import logger

class NormalizationEngine:
    """Normalizes raw input product data and generated AI content for consistency."""
    
    @staticmethod
    def normalize_value(val: Any) -> Any:
        """
        Normalizes a single value (string, numeric, list, dict) for deterministic comparison.
        
        - Lowercases all strings.
        - Trims and collapses whitespaces.
        - Normalizes punctuation (treats '-' and ' ' as equivalent).
        - Normalizes dimension separators (treats '×', 'x', 'X' as equivalent, removes surrounding spaces).
        - Resolves color abbreviations (e.g. blk -> black).
        - Translates sizes/inches (e.g. 55-inch, 55in, 55\" -> 55 inch).
        - Converts all weights to standard grams 'g' (e.g. 16.5kg -> 16500 g).
        - Resolves boolean-equivalent strings (e.g. YES/Yes/yes/true -> yes).
        """
        if val is None:
            return ""
            
        if isinstance(val, bool):
            return "yes" if val else "no"
            
        if isinstance(val, (int, float)):
            # Represent floats with clean strings or floats
            return float(val)
            
        if isinstance(val, list):
            normalized_list = []
            for item in val:
                norm_item = NormalizationEngine.normalize_value(item)
                if norm_item != "":
                    normalized_list.append(norm_item)
            # Return sorted to ensure order-independent comparisons
            return sorted(list(set(normalized_list)), key=lambda x: str(x))
            
        if isinstance(val, dict):
            normalized_dict = {}
            for k, v in val.items():
                norm_k = str(k).strip().lower()
                norm_k = re.sub(r"[\s\-_]+", " ", norm_k).strip()
                normalized_dict[norm_k] = NormalizationEngine.normalize_value(v)
            return normalized_dict
            
        # Treat as string
        text = str(val).strip().lower()
        
        # Replace multiple spaces/tabs/newlines with a single space
        text = re.sub(r"\s+", " ", text)
        
        # Treat hyphens, underscores and spaces as equivalent
        text = re.sub(r"[\-_]", " ", text)
        text = re.sub(r"\s+", " ", text) # collapse extra space if hyphens had spaces around them
        
        # Normalize dimension multiplication signs (e.g. 3840 x 2160 -> 3840x2160)
        text = re.sub(r"\s*[xX×]\s*", "x", text)
        
        # Color abbreviation mappings
        color_map = {
            "blk": "black", "wht": "white", "gry": "grey", "blu": "blue",
            "red": "red", "grn": "green", "ylw": "yellow", "org": "orange",
            "ppl": "purple", "pnk": "pink", "brwn": "brown"
        }
        for abbrev, full in color_map.items():
            text = re.sub(rf"\b{abbrev}\b", full, text)
            
        # Standarize weight units and convert all matching weight instances to grams (g)
        # e.g., 16.5kg or 16.5 kg -> 16500 g
        weight_kg_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|kilograms|kilogram|kgs)\b", text)
        if weight_kg_match:
            val_g = float(weight_kg_match.group(1)) * 1000.0
            val_g_str = f"{int(val_g)}" if val_g.is_integer() else f"{val_g}"
            text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:kg|kilograms|kilogram|kgs)\b", f"{val_g_str} g", text)
            
        weight_g_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|gm|grams|gram|gms)\b", text)
        if weight_g_match:
            val_g = float(weight_g_match.group(1))
            val_g_str = f"{int(val_g)}" if val_g.is_integer() else f"{val_g}"
            text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:g|gm|grams|gram|gms)\b", f"{val_g_str} g", text)
            
        # Standardize size/inch units (55-inch, 55in, 55 inch -> 55 inch)
        text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:inch|inches|in)\b", r"\1 inch", text)
        # Handle quotes as inches: e.g. 55" -> 55 inch
        text = re.sub(r"(\d+(?:\.\d+)?)\s*\"", r"\1 inch", text)
        
        # Normalize boolean text
        if text in ["yes", "true"]:
            return "yes"
        if text in ["no", "false"]:
            return "no"
            
        # Compare numeric values where applicable
        try:
            val_float = float(text.replace(" ", ""))
            return val_float
        except ValueError:
            pass
            
        return text.strip()

    @staticmethod
    def normalize(product: Product) -> Product:
        """Normalizes an entire Product Pydantic model in place."""
        logger.info(f"Normalizing product data: {product.product_name}")
        
        product.product_name = str(product.product_name).strip()
        product.brand = str(product.brand).strip()
        product.material = str(product.material).strip()
        product.target_audience = str(product.target_audience).strip()
        product.brand_voice = str(product.brand_voice).strip()
        product.tone = str(product.tone).strip()
        
        if product.category:
            product.category = str(product.category).strip()
        if product.subcategory:
            product.subcategory = str(product.subcategory).strip()
        if product.warranty:
            product.warranty = str(product.warranty).strip()
            
        if product.color:
            color_norm = NormalizationEngine.normalize_value(product.color)
            product.color = str(color_norm).capitalize()
            
        # Normalize weight parsing
        if product.weight_raw:
            weight_norm = str(NormalizationEngine.normalize_value(product.weight_raw))
            if " g" in weight_norm:
                try:
                    product.weight_grams = float(weight_norm.replace(" g", ""))
                except Exception:
                    pass
            else:
                # Try to parse it as a pure float (e.g. plain number "80")
                try:
                    cleaned_num = re.sub(r"[^\d.]", "", weight_norm)
                    if cleaned_num:
                        product.weight_grams = float(cleaned_num)
                except Exception:
                    pass
        
        if product.weight_grams is not None:
            product.weight_raw = f"{product.weight_grams:.1f} g"
            
        # Normalize lists (with case-insensitive deduplication)
        def clean_list(lst):
            seen = set()
            cleaned = []
            for item in lst:
                item_clean = str(item).strip()
                if item_clean and item_clean.lower() not in seen:
                    seen.add(item_clean.lower())
                    cleaned.append(item_clean)
            return cleaned

        product.features = clean_list(product.features)
        product.seo_keywords = clean_list(product.seo_keywords)
        product.package_contents = clean_list(product.package_contents)
        
        # Normalize dicts
        normalized_specs = {}
        for k, v in product.specifications.items():
            normalized_specs[str(k).strip()] = str(v).strip()
        product.specifications = normalized_specs
        
        return product
