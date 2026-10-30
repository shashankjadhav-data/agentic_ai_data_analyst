# agents/critic.py
import json
from typing import Any, Dict, Optional

from .base import Agent

CRITIC_SYSTEM = """
Return STRICT JSON ONLY:

{
  "status": "ok" | "retry",
  "reasons": [str, ...],
  "revised_modeling_steps": [ ... ]
}

If there are no modeling_steps, always status="ok".
"""


def _extract_first_balanced_json_object(text: str) -> Optional[str]:
    s = text.strip()
    if not s:
        return None
    start = s.find("{")
    if start == -1:
        return None

    in_string = False
    escape = False
    depth = 0

    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


class CritiqueAgent(Agent):
    def run(self):
        results = self.bus.get_state("analysis_results") or {}
        plan = self.bus.get_state("analysis_plan") or {}

        payload = {
            "metrics": results.get("metrics", []),
            "errors": results.get("errors", []),
            "modeling_steps": plan.get("modeling_steps", []),
        }

        # If no modeling, do not spend an LLM call
        if not payload["modeling_steps"]:
            critique = {"status": "ok", "reasons": ["No modeling steps."], "revised_modeling_steps": []}
            self.bus.set_state("critique", critique)
            self.bus.publish(self.name, {"critique": critique})
            return critique

        text, duration = self.llm.chat(CRITIC_SYSTEM, json.dumps(payload, indent=2), max_tokens=256)
        self.bus.publish(self.name, {"duration": duration, "raw_llm_response": text})

        try:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError("Critic returned non-dict JSON")
        except Exception:
            cand = _extract_first_balanced_json_object(text)
            if cand:
                obj = json.loads(cand)
            else:
                obj = {"status": "ok", "reasons": ["Critic output unusable; fallback ok."], "revised_modeling_steps": []}

        obj.setdefault("status", "ok")
        obj.setdefault("reasons", [])
        obj.setdefault("revised_modeling_steps", [])

        self.bus.set_state("critique", obj)
        self.bus.publish(self.name, {"critique": obj})
        return obj
