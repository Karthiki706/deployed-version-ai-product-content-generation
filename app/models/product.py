from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional

class Product(BaseModel):
    product_id: Optional[str] = None
    product_name: str = Field(..., min_length=2)
    brand: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    material: str
    color: str
    weight_grams: Optional[float] = None
    weight_raw: Optional[str] = None  # To capture raw user inputs like "320gm" for normalization
    dimensions: Dict[str, float] = Field(default_factory=dict)  # e.g., {"height": 8, "width": 5, "depth": 2}
    features: List[str] = Field(default_factory=list)
    specifications: Dict[str, Any] = Field(default_factory=dict)
    target_audience: str
    seo_keywords: List[str] = Field(default_factory=list)
    brand_voice: str = "Professional"
    tone: str = "Expert"
    description_length: int = 200
    warranty: Optional[str] = None
    package_contents: List[str] = Field(default_factory=list)

    @field_validator('weight_grams')
    @classmethod
    def validate_weight(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Weight must be positive")
        return v
