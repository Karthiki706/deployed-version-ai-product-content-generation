import time
import json
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableConfig

from app.core.config import settings
from app.core.logger import logger
from app.core.request_logger import RequestLogger

from app.models.product import Product
from app.models.response import GeneratedContent
from app.models.reports import HallucinationReport, SEOReport, BrandReport, QualityReport

from app.engines.normalizer import NormalizationEngine
from app.engines.validator import AttributeValidationEngine
from app.engines.classifier import CategoryClassificationEngine
from app.engines.enrichment import ProductEnrichmentEngine
from app.engines.prompt_registry import PromptRegistry
from app.engines.prompt_builder import DynamicPromptBuilder
from app.engines.generator import GeminiContentGenerator
from app.engines.json_validator import JSONValidationEngine
from app.engines.hallucination import HallucinationDetectionEngine
from app.engines.seo import SEOValidationEngine
from app.engines.brand import BrandComplianceEngine
from app.engines.scoring import QualityScoringEngine
from app.engines.correction import CorrectionPromptBuilder

from app.orchestrator.graph_state import GraphState
from app.orchestrator.registry import agent_registry
from app.agents.base import BaseAgent

# Import core reasoning agents from app/agents/
from app.agents.planner import PlannerAgent
from app.agents.reflection import ReflectionAgent
from app.agents.supervisor import SupervisorAgent
from app.agents.writer import AgenticGenerationAgent


class NormalizationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Normalization Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        product = state["product"]
        normalized_product = NormalizationEngine.normalize(product)
        return {"normalized_product": normalized_product}


class ClassificationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Category Classification Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        normalized_product = state["normalized_product"]
        classifier = CategoryClassificationEngine()
        res = classifier.classify(normalized_product.product_name, normalized_product.category)
        
        normalized_product.category = res["category"]
        if res.get("subcategory") and not normalized_product.subcategory:
            normalized_product.subcategory = res["subcategory"]
            
        return {
            "normalized_product": normalized_product,
            "category": res["category"],
            "subcategory": res.get("subcategory", ""),
        }


class ValidationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Semantic Validation Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        normalized_product = state["normalized_product"]
        detected_category = state["category"]
        detected_subcategory = state.get("subcategory")
        validator_category = detected_subcategory or detected_category
        
        api_key_override = config.get("configurable", {}).get("api_key_override")
        
        validation_rep = AttributeValidationEngine.validate(
            normalized_product,
            category=validator_category,
            api_key=api_key_override,
        )
        
        validation_rep["_detected_category"] = detected_category
        
        if validation_rep["status"] == "FAIL":
            error_msg = f"Attribute validation failed: {', '.join(validation_rep['errors'])}"
            raise ValueError(error_msg)
            
        semantic_status = validation_rep.get("semantic_status", "PASS")
        accepted_features = validation_rep.get("accepted", [])
        rejected_attrs_raw = validation_rep.get("rejected", [])
        warned_attrs_raw = validation_rep.get("warnings", [])
        rejected_names = [r.get("attribute", "") for r in rejected_attrs_raw]
        warned_names = [w.get("attribute", w) for w in warned_attrs_raw]
        
        req_logger = state.get("request_logger")
        if req_logger:
            req_logger.record_validation_decision(
                detected_category=detected_category,
                accepted=accepted_features,
                rejected=rejected_names,
                warned=warned_names,
                latency=float(validation_rep.get("validation_latency_seconds", 0.0)),
            )
            if semantic_status == "DEGRADED":
                req_logger.add_warning("Semantic validation is DEGRADED — all attributes are unvalidated.")
            elif rejected_names:
                req_logger.add_warning(f"Semantic validation rejected {len(rejected_names)} attribute(s).")
                
        return {
            "validation_report": validation_rep,
            "accepted_attributes": accepted_features,
            "rejected_attributes": rejected_names,
            "warning_attributes": warned_names,
            "semantic_status": semantic_status,
            "validation_confidence": validation_rep.get("validation_confidence", 0.0),
        }


class EnrichmentAgent(BaseAgent):
    def __init__(self):
        super().__init__("Product Enrichment Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        category = state["category"]
        enricher = ProductEnrichmentEngine()
        enrichment_tags = enricher.enrich(category)
        return {"enrichment_tags": enrichment_tags}


class PromptBuilderAgent(BaseAgent):
    def __init__(self):
        super().__init__("Prompt Builder Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        category = state["category"]
        category_guidance, prompt_version = PromptRegistry.get_template(category, "v1")
        
        req_logger = state.get("request_logger")
        if req_logger:
            req_logger.prompt_version = prompt_version
            
        builder = DynamicPromptBuilder()
        
        # Integrate Planner outcomes into writing instruction directives
        writing_strategy = state.get("writing_strategy", "")
        selected_brand_style = state.get("selected_brand_style", "")
        seo_keyword_strategy = state.get("seo_keyword_strategy", "")
        bullet_style = state.get("bullet_style", "")
        content_structure = state.get("content_structure", "")
        
        # Append Planner directives to category guidance
        planning_directives = (
            f"\n\nPlanner Directives:\n"
            f"- Writing Strategy: {writing_strategy}\n"
            f"- Brand Style: {selected_brand_style}\n"
            f"- SEO Keyword Strategy: {seo_keyword_strategy}\n"
            f"- Bullet Points Layout: {bullet_style}\n"
            f"- Content Structure: {content_structure}"
        )
        planner_risks = state.get("planner_risks") or []
        if planner_risks:
            planning_directives += "\n- Risks To Avoid: " + "; ".join(planner_risks)
        if state.get("supervisor_focus"):
            planning_directives += f"\n- Supervisor Focus (highest priority): {state['supervisor_focus']}"
        category_guidance_planned = category_guidance + planning_directives
        
        master_prompt = builder.build(
            product=state["normalized_product"],
            category_guidance=category_guidance_planned,
            enrichment_tags=state.get("enrichment_tags", []),
            accepted_features=state["accepted_attributes"] if state["semantic_status"] == "PASS" else None,
            rejected_attributes=state["rejected_attributes"],
            semantic_status=state["semantic_status"],
        )
        
        return {
            "generation_prompt": master_prompt,
            "prompt": state.get("prompt") or master_prompt,
            "prompt_version": prompt_version,
        }


class GenerationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Generation Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        active_prompt = state["prompt"]
        api_key_override = config.get("configurable", {}).get("api_key_override")
        
        generator = GeminiContentGenerator(api_key=api_key_override)
        req_logger = state.get("request_logger")
        
        raw_output = generator.generate(active_prompt, req_logger=req_logger)
        
        prompt_tokens = 0
        completion_tokens = 0
        cost = 0.0
        if req_logger:
            prompt_tokens = req_logger.prompt_tokens
            completion_tokens = req_logger.completion_tokens
            cost = req_logger.estimated_cost_usd
            
        return {
            "generated_content_raw": raw_output,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_cost": cost,
        }


class JSONValidationAgent(BaseAgent):
    def __init__(self):
        super().__init__("JSON Validation Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        raw_output = state["generated_content_raw"]
        
        try:
            parsed_content = JSONValidationEngine.validate(raw_output)
            return {
                "generated_content": parsed_content,
                "json_error": None,
            }
        except Exception as e:
            return {
                "generated_content": None,
                "json_error": str(e),
            }


class HallucinationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Hallucination Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        content_dict = state["generated_content"].model_dump() if state["generated_content"] else {}
        normalized_product = state["normalized_product"]
        validation_report = state["validation_report"]
        
        h_detector = HallucinationDetectionEngine()
        h_rep = h_detector.detect(
            content_dict,
            normalized_product,
            validation_report=validation_report,
        )
        
        h_model = HallucinationReport(**h_rep)
        return {
            "hallucination_report": h_model,
            "comparison_report": h_rep.get("comparison_report", []),
        }


class SEOAgent(BaseAgent):
    def __init__(self):
        super().__init__("SEO Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        content_dict = state["generated_content"].model_dump() if state["generated_content"] else {}
        normalized_product = state["normalized_product"]
        
        seo_engine = SEOValidationEngine()
        s_rep = seo_engine.validate(
            content_dict,
            normalized_product.seo_keywords,
            normalized_product.description_length,
        )
        
        s_model = SEOReport(**s_rep)
        return {"seo_report": s_model}


class BrandComplianceAgent(BaseAgent):
    def __init__(self):
        super().__init__("Brand Compliance Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        content_dict = state["generated_content"].model_dump() if state["generated_content"] else {}
        normalized_product = state["normalized_product"]
        
        brand_engine = BrandComplianceEngine()
        b_rep = brand_engine.check(
            content_dict,
            normalized_product.brand_voice,
            normalized_product.tone,
        )
        
        b_model = BrandReport(**b_rep)
        return {"brand_report": b_model}


class QualityScoringAgent(BaseAgent):
    def __init__(self):
        super().__init__("Quality Scoring Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        h_rep = state["hallucination_report"].model_dump() if state["hallucination_report"] else {}
        s_rep = state["seo_report"].model_dump() if state["seo_report"] else {}
        b_rep = state["brand_report"].model_dump() if state["brand_report"] else {}
        validation_report = state["validation_report"]
        
        scoring_engine = QualityScoringEngine()
        q_rep = scoring_engine.calculate(
            h_rep, s_rep, b_rep,
            validation_report=validation_report,
        )
        
        q_model = QualityReport(**q_rep)
        return {
            "quality_report": q_model,
            "overall_quality": q_rep.get("overall_quality", 0.0),
            "safety_score": q_rep.get("safety_score", 100.0),
            "risk_level": q_rep.get("risk_level", "Low"),
            "generation_confidence": q_rep.get("generation_confidence", 1.0),
        }


class CorrectionAgent(BaseAgent):
    def __init__(self):
        super().__init__("Correction Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        master_prompt = state["generation_prompt"]
        content_dict = state["generated_content"].model_dump() if state["generated_content"] else {}
        errors_to_correct = list(state.get("weaknesses", []))
        if state.get("supervisor_focus"):
            errors_to_correct.append(f"Supervisor directive: {state['supervisor_focus']}")
        rejected_names = state["rejected_attributes"]
        accepted_features = state["accepted_attributes"]
        
        correction_builder = CorrectionPromptBuilder()
        new_prompt = correction_builder.build(
            master_prompt,
            content_dict,
            errors_to_correct,
            rejected_attributes=rejected_names,
            accepted_attributes=accepted_features,
        )
        
        return {"prompt": new_prompt}


# Register all class instances dynamically in the pluggable registry with explicit node name keys
agent_registry.register("normalization_node", NormalizationAgent(), "sequential")
agent_registry.register("classification_node", ClassificationAgent(), "sequential")
agent_registry.register("validation_node", ValidationAgent(), "sequential")
agent_registry.register("planner_agent_node", PlannerAgent(), "sequential")
agent_registry.register("enrichment_node", EnrichmentAgent(), "sequential")
agent_registry.register("prompt_builder_node", PromptBuilderAgent(), "sequential")
agent_registry.register("generation_node", AgenticGenerationAgent(), "sequential")  # tool-using; falls back to GenerationAgent logic
agent_registry.register("json_validation_node", JSONValidationAgent(), "sequential")

# Parallel agents fanning out from JSON Validation
agent_registry.register("hallucination_node", HallucinationAgent(), "parallel", depends_on="json_validation_node")
agent_registry.register("seo_node", SEOAgent(), "parallel", depends_on="json_validation_node")
agent_registry.register("brand_node", BrandComplianceAgent(), "parallel", depends_on="json_validation_node")

# Merge nodes running sequentially
agent_registry.register("quality_scoring_node", QualityScoringAgent(), "sequential")
agent_registry.register("reflection_agent_node", ReflectionAgent(), "sequential")
agent_registry.register("supervisor_agent_node", SupervisorAgent(), "sequential")
agent_registry.register("correction_node", CorrectionAgent(), "sequential")

