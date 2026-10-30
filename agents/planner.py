# agents/planner.py
import ast
import json
import re
from typing import Any, Dict

from .base import Agent

PLANNER_SYSTEM = """
You are the Planner for an Agentic AI Analyst that must work on ANY tabular CSV.

You MUST return ONLY valid JSON (one object). No prose, no code, no markdown.
Use DOUBLE QUOTES for all keys/strings.

Hard limits:
- EDA steps: 2–4
- Preprocessing steps: 0–2
- Modeling steps: 0–1

Schema:
{
  "eda_steps": [{"id": str, "description": str}],
  "preprocessing_steps": [{"id": str, "description": str}],
  "need_modeling": boolean,
  "modeling_request": {
    "target": str | null,
    "problem_type": "classification" | "regression" | null,
    "priority_metric": "f1_weighted" | "accuracy" | "roc_auc_ovr" | "r2" | "rmse" | null,
    "notes": str
  }
}

Rules:
- If target_recommendation.confidence < 0.6 OR target is null: need_modeling=false.
- If classification and imbalance likely: prefer priority_metric="f1_weighted".
- If multiclass: prefer priority_metric="f1_weighted" (roc_auc_ovr only if predict_proba is planned).
"""


def _extract_first_object(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)


def _parse_llm_json(text: str) -> Dict[str, Any]:
    raw = text.strip()
    if not raw:
        raise ValueError("PlannerAgent: empty response from LLM.")

    # strict
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # extract
    candidate = _extract_first_object(raw)
    if not candidate:
        raise ValueError(f"PlannerAgent: no JSON object found:\n{raw}")

    # strict candidate
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # repair commas
    repaired = _remove_trailing_commas(candidate)
    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # literal_eval fallback (last resort)
    py_like = repaired
    py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)
    py_like = re.sub(r"\btrue\b", "True", py_like, flags=re.IGNORECASE)
    py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
    obj = ast.literal_eval(py_like)
    return json.loads(json.dumps(obj))


class PlannerAgent(Agent):
    def run(self):
        profile = self.bus.get_state("data_profile") or {}

        text, duration = self.llm.chat(
            PLANNER_SYSTEM,
            json.dumps(profile, indent=2),
            max_tokens=512,
        )
        self.bus.publish(self.name, {"duration": duration, "raw_llm_response": text})

        plan = _parse_llm_json(text)

        plan.setdefault("eda_steps", [])
        plan.setdefault("preprocessing_steps", [])
        plan.setdefault("need_modeling", False)
        plan.setdefault("modeling_request", {"target": None, "problem_type": None, "priority_metric": None, "notes": ""})

        # enforce rule from ingestion confidence
        tr = profile.get("target_recommendation") or {}
        target = tr.get("target")
        conf = float(tr.get("confidence", 0.0) or 0.0)

        if (not target) or (conf < 0.6):
            plan["need_modeling"] = False
            plan["modeling_request"] = {
                "target": None,
                "problem_type": None,
                "priority_metric": None,
                "notes": "Target confidence too low; modeling disabled.",
            }
        else:
            # Ensure modeling_request is populated
            mr = plan.get("modeling_request") or {}
            mr["target"] = mr.get("target") or target
            mr["problem_type"] = mr.get("problem_type") or tr.get("problem_type")
            if not mr.get("priority_metric"):
                mr["priority_metric"] = "f1_weighted" if mr["problem_type"] == "classification" else "r2"
            mr["notes"] = mr.get("notes") or "Train a fast, appropriate baseline using mixed feature types."
            plan["modeling_request"] = mr
            plan["need_modeling"] = True

        # Hard clamp step counts
        plan["eda_steps"] = (plan.get("eda_steps") or [])[:4]
        plan["preprocessing_steps"] = (plan.get("preprocessing_steps") or [])[:2]

        self.bus.set_state("analysis_plan", plan)
        self.bus.publish(self.name, {"plan": plan})
        return plan
