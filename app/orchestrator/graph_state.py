from typing import TypedDict, List, Dict, Any, Optional
from app.models.product import Product
from app.models.response import GeneratedContent
from app.models.reports import HallucinationReport, SEOReport, BrandReport, QualityReport

class InputState(TypedDict, total=False):
    product: Product
    normalized_product: Product
    category: str
    subcategory: str
    enrichment_tags: List[str]

class ValidationState(TypedDict, total=False):
    validation_report: Dict[str, Any]
    accepted_attributes: List[str]
    warning_attributes: List[str]
    rejected_attributes: List[str]
    semantic_status: str
    validation_confidence: float

class PlannerState(TypedDict, total=False):
    writing_strategy: str
    description_length_target: int
    selected_tone: str
    selected_brand_style: str
    seo_keyword_strategy: str
    bullet_style: str
    content_structure: str
    planner_risks: List[str]
    planner_source: str          # "llm" or "deterministic"

class GenerationState(TypedDict, total=False):
    generation_prompt: str
    prompt: str
    generated_content_raw: str
    generated_content: Optional[GeneratedContent]
    json_error: Optional[str]

class ReflectionState(TypedDict, total=False):
    reflection_summary: str
    weaknesses: List[str]

class AuditState(TypedDict, total=False):
    hallucination_report: Optional[HallucinationReport]
    seo_report: Optional[SEOReport]
    brand_report: Optional[BrandReport]
    comparison_report: List[Dict[str, Any]]
    quality_report: Optional[QualityReport]
    overall_quality: float
    safety_score: float
    risk_level: str
    generation_confidence: float

class RetryState(TypedDict, total=False):
    retry_count: int
    best_attempt: Optional[Dict[str, Any]]
    best_quality_score: float
    retry_reason: str
    decision_log: List[str]
    attempts_history: List[Dict[str, Any]]
    supervisor_focus: str        # LLM supervisor's strategic directive for the next attempt
    supervisor_action: str       # "regenerate" | "escalate"

class ObservabilityState(TypedDict, total=False):
    request_id: str
    request_logger: Optional[Any]
    timeline: List[Dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    agent_trace: List[Dict[str, Any]]   # tool calls made by the agentic writer
    manual_review_required: bool
    next_step: str

# Composite GraphState using TypedDict composition
class GraphState(
    InputState,
    ValidationState,
    PlannerState,
    GenerationState,
    ReflectionState,
    AuditState,
    RetryState,
    ObservabilityState
):
    pass
