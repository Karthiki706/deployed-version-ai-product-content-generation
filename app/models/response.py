from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class GeneratedContent(BaseModel):
    title: str = Field(..., description="An attention-grabbing, SEO-optimized title for the product.")
    short_description: str = Field(..., description="A brief summary of the product (approx. 50 words).")
    long_description: str = Field(..., description="A detailed product description matching the desired word count.")
    bullet_points: List[str] = Field(..., description="Key features and benefits as bullet points.")
    meta_description: str = Field(..., description="SEO meta description under 160 characters.")
    seo_keywords_used: List[str] = Field(..., description="List of target keywords actually integrated into the content.")
    specification_summary: Dict[str, Any] = Field(..., description="A structured dictionary mapping specification keys to values.")

class PipelineResponse(BaseModel):
    request_id: str
    content: GeneratedContent
    scores: Dict[str, float] = Field(
        ...,
        description="Dictionary containing 'overall', 'hallucination', 'seo', and 'brand' scores (0-100)."
    )
    reports: Dict[str, Any] = Field(
        ...,
        description="The detailed validation reports for hallucination, seo, brand, and quality."
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Execution details such as execution time, retry count, model used, and prompt version."
    )
