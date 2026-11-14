# Category map matching lowercase words to standard categories
CATEGORY_MAP = {
    "Footwear": ["shoe", "boot", "sneaker", "runner", "sandal", "heel", "clog"],
    "Electronics": ["laptop", "phone", "camera", "tablet", "stove", "speaker", "headphone", "smartwatch", "tv"],
    "Beauty": ["face wash", "serum", "cream", "lotion", "cleanser", "shampoo", "makeup", "skincare"],
    "Fashion": ["shirt", "jacket", "pants", "dress", "clothing", "t-shirt", "coat", "jeans"],
    "Furniture": ["chair", "table", "sofa", "desk", "bed", "dresser", "cabinet", "shelf"],
    "Grocery": ["coffee", "tea", "snack", "oil", "sauce", "rice", "pasta", "food", "beverage"]
}

# Derived tags based on classified category
ENRICHMENT_RULES = {
    "Footwear": ["Sports", "Fitness", "Outdoor", "Athletic", "Apparel"],
    "Electronics": ["Tech", "Gadget", "Digital", "Hardware", "Smart"],
    "Beauty": ["Self-Care", "Wellness", "Grooming", "Cosmetics", "Organic"],
    "Fashion": ["Style", "Fit", "Occasion", "Comfort", "Apparel"],
    "Furniture": ["Interior", "Home Decor", "Durable", "Living", "Modern"],
    "Grocery": ["Pantry", "Food & Drink", "Natural", "Edible", "Fresh"]
}

# Brand compliance parameters
RESTRICTED_WORDS = ["best", "cheapest", "guaranteed", "perfect", "revolutionary", "risk-free", "ultimate", "no risk"]

# Standard units mapping for normalizer
UNIT_NORMALIZATION_RULES = {
    r"(\d+)\s*(?:g|gm|gms|grams|gram)\b": r"\1 g",
    r"(\d+)\s*(?:kg|kgs|kilograms|kilogram)\b": r"\1 kg",
    r"(\d+)\s*(?:ml|mls|milliliters|milliliter)\b": r"\1 ml",
    r"(\d+)\s*(?:l|liters|liter)\b": r"\1 l",
    r"(\d+)\s*(?:cm|cms|centimeters|centimeter)\b": r"\1 cm",
    r"(\d+)\s*(?:mm|mms|millimeters|millimeter)\b": r"\1 mm",
    r"(\d+)\s*(?:in|inches|inch|[\"])\b": r"\1 inch"
}

# Text normalization mappings (abbreviations to standards)
TEXT_NORMALIZATION_MAP = {
    "blk": "Black",
    "wht": "White",
    "gry": "Grey",
    "blu": "Blue",
    "red": "Red",
    "grn": "Green",
    "ylw": "Yellow",
    "org": "Orange",
    "ppl": "Purple",
    "pnk": "Pink",
    "brwn": "Brown"
}
