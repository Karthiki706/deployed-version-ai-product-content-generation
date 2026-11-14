"""Offline tests for the agentic layer. A scripted fake Gemini client stands in for the API."""
import json
from types import SimpleNamespace

import pytest
from google.genai import types

from app.agents.tool_loop import ToolLoopError, run_tool_loop
from app.agents.tools import ToolContext, build_tools, parse_draft
from app.agents.supervisor import SupervisorAgent
from app.core.config import settings
from app.models.product import Product

GOOD = {
    "title": "Alpine Carbon Fiber Pole: Lightweight Hiking Pole",
    "short_description": "A lightweight hiking pole for trail walkers.",
    "long_description": "The Carbon Fiber Pole is a hiking pole with an ultralight design, made of carbon fiber, that weighs 150 grams.",
    "bullet_points": ["Ultralight design", "Carbon fiber build", "Weighs 150 grams"],
    "meta_description": "Lightweight carbon fiber hiking pole.",
    "seo_keywords_used": ["hiking pole"],
    "specification_summary": {"material": "Carbon Fiber", "weight": "150 g"},
}
BAD = dict(GOOD, title="The Ultimate Carbon Fiber Pole",
           long_description="The ultimate pole, guaranteed perfect, with a 10-year battery.")


@pytest.fixture
def product():
    return Product(product_name="Carbon Fiber Pole", brand="Alpine", material="Carbon Fiber",
                   color="Grey", weight_grams=150.0, features=["Ultralight design"],
                   target_audience="Hikers", seo_keywords=["hiking pole"])


@pytest.fixture
def ctx(product):
    return ToolContext(product=product, category="Sports", accepted_attributes=["Ultralight design"])


def call_turn(name, args=None):
    content = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args or {}))])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text=None, usage_metadata=None)


def text_turn(text):
    content = types.Content(role="model", parts=[types.Part(text=text)])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text=text, usage_metadata=None)


class FakeClient:
    """Plays back scripted responses and records every request."""
    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.requests.append((model, len(contents), config))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ---------------------------------------------------------------- tools
def test_parse_draft_handles_fences():
    assert parse_draft("```json\n" + json.dumps(GOOD) + "\n```")["title"] == GOOD["title"]
    with pytest.raises(ValueError):
        parse_draft("not json")


def test_brand_tool_catches_banned_words(ctx):
    check = {t.__name__: t for t in build_tools(ctx)}["check_brand_rules"]
    bad, good = check(json.dumps(BAD)), check(json.dumps(GOOD))
    assert not bad["passed"] and "ultimate" in bad["banned_words_found"]
    assert good["passed"]


def test_seo_tool_reports_missing_keywords(ctx):
    check = {t.__name__: t for t in build_tools(ctx)}["check_seo"]
    assert check(json.dumps(GOOD))["passed"]
    assert check(json.dumps(dict(GOOD, title="Pole", short_description="x", long_description="y")))["missing_keywords"] == ["hiking pole"]


def test_facts_tool_lists_forbidden_attributes(product):
    c = ToolContext(product=product, category="Sports", rejected_attributes=["heart rate monitor"])
    facts = {t.__name__: t for t in build_tools(c)}["get_verified_facts"]()
    assert facts["FORBIDDEN_attributes_never_mention"] == ["heart rate monitor"]
    assert facts["weight_grams"] == 150.0


def test_tool_error_is_returned_not_raised(ctx):
    assert "error" in {t.__name__: t for t in build_tools(ctx)}["check_seo"]("garbage")


# ---------------------------------------------------------------- tool loop
def test_loop_runs_tools_then_returns_final_json(ctx):
    client = FakeClient([
        call_turn("get_verified_facts"),
        call_turn("check_brand_rules", {"draft_json": json.dumps(BAD)}),   # agent self-checks, finds problem
        call_turn("check_brand_rules", {"draft_json": json.dumps(GOOD)}),  # fixes, re-checks
        text_turn(json.dumps(GOOD)),
    ])
    final, trace = run_tool_loop(client, "write copy", build_tools(ctx))
    assert json.loads(final)["title"] == GOOD["title"]
    assert [t["tool"] for t in trace] == ["get_verified_facts", "check_brand_rules", "check_brand_rules", None]
    assert "false" in trace[1]["result_preview"].lower()      # first check failed
    assert "true" in trace[2]["result_preview"].lower()       # second passed
    assert client.requests[0][2].automatic_function_calling.disable is True


def test_loop_survives_unknown_tool_and_bad_args(ctx):
    client = FakeClient([call_turn("does_not_exist"), call_turn("check_seo", {"wrong": 1}), text_turn("{}")])
    _, trace = run_tool_loop(client, "p", build_tools(ctx))
    assert "Unknown tool" in trace[0]["result_preview"]
    assert "error" in trace[1]["result_preview"]


def test_loop_forces_final_answer_when_budget_exhausted(ctx):
    client = FakeClient([call_turn("get_verified_facts")] * 2 + [text_turn(json.dumps(GOOD))])
    final, trace = run_tool_loop(client, "p", build_tools(ctx), max_steps=3)
    assert trace[-1]["type"] == "final" and len(trace) == 3


def test_loop_falls_through_model_chain_then_raises(ctx):
    n = len(__import__("app.agents.llm_utils", fromlist=["x"]).model_chain())
    client = FakeClient([RuntimeError("503")] * n)
    with pytest.raises(ToolLoopError):
        run_tool_loop(client, "p", build_tools(ctx))


# ---------------------------------------------------------------- supervisor advisor guardrails
def _advise(monkeypatch, llm_json, retry_count):
    monkeypatch.setattr("app.agents.supervisor.generate_json", lambda *a, **k: llm_json)
    monkeypatch.setattr("app.core.gemini_service.GeminiService.get_client", classmethod(lambda cls: object()))
    return SupervisorAgent()._advise({}, ["Brand: ultimate"], [], [], retry_count)


def test_advisor_cannot_escalate_before_a_correction_attempt(monkeypatch):
    action, _, reason = _advise(monkeypatch, {"action": "escalate", "focus": "x", "reason": "r"}, retry_count=0)
    assert action == "regenerate" and "Guardrail" in reason


def test_advisor_can_escalate_after_retry(monkeypatch):
    assert _advise(monkeypatch, {"action": "escalate", "focus": "f", "reason": "r"}, retry_count=1)[0] == "escalate"


def test_advisor_rejects_unknown_actions(monkeypatch):
    assert _advise(monkeypatch, {"action": "accept_draft", "focus": "f", "reason": "r"}, retry_count=1)[0] == "regenerate"


def test_advisor_failure_degrades_to_regenerate(monkeypatch):
    def boom(*a, **k): raise RuntimeError("api down")
    monkeypatch.setattr("app.agents.supervisor.generate_json", boom)
    monkeypatch.setattr("app.core.gemini_service.GeminiService.get_client", classmethod(lambda cls: object()))
    assert SupervisorAgent()._advise({}, [], [], [], 1)[0] == "regenerate"


def test_advisor_disabled_by_flag(monkeypatch):
    monkeypatch.setattr(settings, "llm_supervisor_enabled", False)
    assert SupervisorAgent()._advise({}, [], [], [], 1)[0] == "regenerate"


# ---------------------------------------------------------------- end-to-end through the real LangGraph
class ScriptedGemini:
    """Routes each request by type: planner, supervisor advisor, validator, agent loop."""
    def __init__(self):
        self.models = SimpleNamespace(generate_content=self._gen)
        self.agent_runs = 0
        self._queue = []

    @staticmethod
    def _json(obj):
        return SimpleNamespace(text=json.dumps(obj), usage_metadata=None, candidates=[])

    def _gen(self, model, contents, config):
        if getattr(config, "tools", None):                       # agent tool loop
            if not self._queue:
                self.agent_runs += 1
                if self.agent_runs == 1:                          # attempt 1: submits a draft with banned words
                    self._queue = [call_turn("get_verified_facts"), text_turn(json.dumps(BAD))]
                else:                                             # attempt 2: self-checks, then submits clean draft
                    self._queue = [call_turn("get_verified_facts"),
                                   call_turn("check_brand_rules", {"draft_json": json.dumps(GOOD)}),
                                   text_turn(json.dumps(GOOD))]
            return self._queue.pop(0)
        prompt = contents if isinstance(contents, str) else ""
        if "planning agent" in prompt:
            return self._json({"writing_strategy": "Lead with weight and material.", "selected_brand_style": "Calm expert",
                               "seo_keyword_strategy": "Keyword in title.", "bullet_style": "Short facts",
                               "content_structure": "Title -> Intro -> Bullets",
                               "planner_risks": ["Do not invent battery or warranty claims"]})
        if "You supervise" in prompt:
            return self._json({"action": "regenerate", "focus": "Drop superlatives like 'ultimate'; state plain facts.",
                               "reason": "Brand violations on first draft."})
        if "e-commerce product content validator" in prompt:
            return self._json({"accepted": ["Ultralight design"], "warnings": [], "rejected": [],
                               "confidence": 0.95, "confidence_reason": "ok"})
        return SimpleNamespace(text=json.dumps(GOOD), usage_metadata=None, candidates=[])


def test_pipeline_end_to_end_agentic(monkeypatch, tmp_path, product):
    from app.core.gemini_service import GeminiService
    from app.orchestrator.orchestrator import AIOrchestrator

    fake = ScriptedGemini()
    monkeypatch.setattr(GeminiService, "get_client", classmethod(lambda cls: fake))
    monkeypatch.setattr(settings, "outputs_dir", str(tmp_path))

    result = AIOrchestrator().run_pipeline(product)
    meta = result.metadata

    assert result.content.title == GOOD["title"]                       # clean draft won
    assert meta["agentic_mode"] is True and meta["planner_source"] == "llm"
    assert fake.agent_runs == 2                                        # first draft failed audit -> retried
    assert any("Supervisor (LLM)" in d for d in meta["supervisor_decisions"])
    tools_used = [t["tool"] for t in meta["agent_trace"] if t["tool"]]
    assert "get_verified_facts" in tools_used and "check_brand_rules" in tools_used
    assert {t["attempt"] for t in meta["agent_trace"]} == {1, 2}
    assert meta["manual_review_required"] is False


def test_pipeline_falls_back_when_agent_loop_breaks(monkeypatch, tmp_path, product):
    """If tool calling fails, the pipeline must still complete via classic generation."""
    from app.core.gemini_service import GeminiService
    from app.orchestrator.orchestrator import AIOrchestrator

    class NoToolsGemini(ScriptedGemini):
        def _gen(self, model, contents, config):
            if getattr(config, "tools", None):
                raise RuntimeError("function calling unavailable")
            return super()._gen(model, contents, config)

    monkeypatch.setattr(GeminiService, "get_client", classmethod(lambda cls: NoToolsGemini()))
    monkeypatch.setattr(settings, "outputs_dir", str(tmp_path))
    result = AIOrchestrator().run_pipeline(product)
    assert result.content.title == GOOD["title"]
    assert any(t["type"] == "classic_generation" for t in result.metadata["agent_trace"])
