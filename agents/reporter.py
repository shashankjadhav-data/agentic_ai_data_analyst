# agents/reporter.py

from __future__ import annotations

from pathlib import Path
import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
import markdown

from .base import Agent


REPORTER_SYSTEM = """Write a concise MARKDOWN report with sections:
1. Dataset overview (rows/cols, schema)
2. Data quality issues
3. EDA highlights (refer to plots conceptually)
4. Modeling results (if any), mention confusion matrix
5. Limitations & recommendations

No HTML tags. Keep it short.
"""


DEFAULT_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Agentic AI Data Analysis Report</title>
  <style>
    body { font-family: system-ui, Segoe UI, sans-serif; margin: 2rem; line-height: 1.5; }
    h1, h2 { color: #003366; }
    h1 { border-bottom: 2px solid #003366; padding-bottom: 0.5rem; }
    .small { font-size: 0.85rem; color: #666; }
    .section { margin: 2rem 0; }
    .grid { display: flex; flex-wrap: wrap; gap: 1rem; }
    .card { border: 1px solid #ccc; padding: 0.5rem; max-width: 360px; }
    .card img { max-width: 100%; height: auto; display: block; }
    table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
    th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: left; }
    th { background: #f0f0f0; }
    pre { background: #f7f7f7; padding: 0.75rem; overflow-x: auto; }
    .error { background: #ffecec; border: 1px solid #ffb3b3; padding: 0.5rem; }
    .badge { display: inline-block; padding: 0.1rem 0.4rem; border: 1px solid #ccc; border-radius: 0.3rem; font-size: 0.8rem; color: #444; }
  </style>
</head>
<body>
  <h1>Agentic AI Data Analysis Report</h1>
  <p class="small">
    Generated at {{ generated_at }} | Critique status: {{ critique.get("status", "n/a") }} | Metrics recorded: {{ metrics|length }}
  </p>

  <div class="section">
    {{ markdown_body | safe }}
  </div>

  {% if figures.histograms %}
  <div class="section">
    <h2>Exploratory Histograms</h2>
    <div class="grid">
      {% for f in figures.histograms %}
      <div class="card">
        <img src="figures/{{ f }}" alt="{{ f }}" />
        <div class="small">{{ f }}</div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  {% if figures.boxplots %}
  <div class="section">
    <h2>Box Plots</h2>
    <div class="grid">
      {% for f in figures.boxplots %}
      <div class="card">
        <img src="figures/{{ f }}" alt="{{ f }}" />
        <div class="small">{{ f }}</div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  {% if figures.barplots %}
  <div class="section">
    <h2>Bar Plots</h2>
    <div class="grid">
      {% for f in figures.barplots %}
      <div class="card">
        <img src="figures/{{ f }}" alt="{{ f }}" />
        <div class="small">{{ f }}</div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  {% if figures.corrheatmaps or figures.othereda %}
  <div class="section">
    <h2>Correlation / Other EDA</h2>
    <div class="grid">
      {% for f in figures.corrheatmaps %}
      <div class="card">
        <img src="figures/{{ f }}" alt="{{ f }}" />
        <div class="small">{{ f }}</div>
      </div>
      {% endfor %}
      {% for f in figures.othereda %}
      <div class="card">
        <img src="figures/{{ f }}" alt="{{ f }}" />
        <div class="small">{{ f }}</div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  {% if figures.confusionmatrices %}
  <div class="section">
    <h2>Confusion Matrices</h2>
    <div class="grid">
      {% for f in figures.confusionmatrices %}
      <div class="card">
        <img src="figures/{{ f }}" alt="{{ f }}" />
        <div class="small">{{ f }}</div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  {% if figures.otherml %}
  <div class="section">
    <h2>Additional ML Plots</h2>
    <div class="grid">
      {% for f in figures.otherml %}
      <div class="card">
        <img src="figures/{{ f }}" alt="{{ f }}" />
        <div class="small">{{ f }}</div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  {% if figures.uncategorised %}
  <div class="section">
    <h2>Other Figures</h2>
    <div class="grid">
      {% for f in figures.uncategorised %}
      <div class="card">
        <img src="figures/{{ f }}" alt="{{ f }}" />
        <div class="small">{{ f }}</div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  <div class="section">
    <h2>ML Metrics</h2>

    {% if not metrics %}
      <p>No ML metrics were recorded (possibly no clear target was found, or modeling was skipped).</p>
    {% else %}
      {# Split metrics by problem type so we can show the right columns #}
      {% set clf = [] %}
      {% set reg = [] %}
      {% for m in metrics %}
        {% set pt = (m.get("problemtype") or m.get("problem_type") or "") %}
        {% if pt == "classification" %}
          {% set _ = clf.append(m) %}
        {% elif pt == "regression" %}
          {% set _ = reg.append(m) %}
        {% else %}
          {# If unknown, keep it visible; default to regression-style columns #}
          {% set _ = reg.append(m) %}
        {% endif %}
      {% endfor %}

      {% if clf %}
      <div class="small"><span class="badge">classification</span></div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Step</th>
            <th>Model</th>
            <th>Type</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
          </tr>
        </thead>
        <tbody>
          {% for m in clf %}
          {% set mm = m.get("metrics", {}) %}
          <tr>
            <td>{{ loop.index }}</td>
            <td>{{ m.get("stepid", m.get("step_id","")) }}</td>
            <td>{{ m.get("model","") }}</td>
            <td>{{ m.get("problemtype", m.get("problem_type","")) }}</td>
            <td>{{ mm.get("accuracy") if mm.get("accuracy") is not none else "N/A" }}</td>
            <td>{{ mm.get("precision") if mm.get("precision") is not none else "N/A" }}</td>
            <td>{{ mm.get("recall") if mm.get("recall") is not none else "N/A" }}</td>
            <td>{{ mm.get("f1") if mm.get("f1") is not none else "N/A" }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% endif %}

      {% if reg %}
      <div style="margin-top: 1rem;" class="small"><span class="badge">regression</span></div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Step</th>
            <th>Model</th>
            <th>Type</th>
            <th>R2</th>
            <th>RMSE</th>
            <th>MAE</th>
          </tr>
        </thead>
        <tbody>
          {% for m in reg %}
          {% set mm = m.get("metrics", {}) %}
          <tr>
            <td>{{ loop.index }}</td>
            <td>{{ m.get("stepid", m.get("step_id","")) }}</td>
            <td>{{ m.get("model","") }}</td>
            <td>{{ m.get("problemtype", m.get("problem_type","")) }}</td>
            <td>{{ mm.get("r2") if mm.get("r2") is not none else "N/A" }}</td>
            <td>{{ mm.get("rmse") if mm.get("rmse") is not none else "N/A" }}</td>
            <td>{{ mm.get("mae") if mm.get("mae") is not none else "N/A" }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% endif %}
    {% endif %}
  </div>

  {% if errors %}
  <div class="section">
    <h2>Execution Errors</h2>
    {% for e in errors %}
    <div class="error"><pre>{{ e | tojson(indent=2) }}</pre></div>
    {% endfor %}
  </div>
  {% endif %}
</body>
</html>
"""


def _json_md_block(title: str, obj: Any, max_chars: int = 20000) -> str:
    try:
        s = json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        s = json.dumps({"error": "Could not serialize object to JSON."}, indent=2)

    if s is None:
        s = "null"

    if len(s) > max_chars:
        s = s[:max_chars] + "\n... truncated ..."

    return f"\n\n## {title}\n\n```json\n{s}\n```\n"


class ReportingAgent(Agent):
    def __init__(self, name, llm_client, bus, templates_dir: Path):
        super().__init__(name, llm_client, bus)

        self.templates_dir = Path(templates_dir).resolve()
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        self.template_path = self.templates_dir / "report.html.j2"
        self._ensure_default_template_up_to_date()

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def _ensure_default_template_up_to_date(self) -> None:
        """
        Create report.html.j2 if missing.
        If it exists but looks like the *old default* ML table (Accuracy..RMSE always),
        auto-upgrade it to the new problem-type aware default.
        """
        if not self.template_path.exists():
            self.template_path.write_text(DEFAULT_TEMPLATE, encoding="utf-8")
            return

        try:
            existing = self.template_path.read_text(encoding="utf-8")
        except Exception:
            # If unreadable, replace with a safe known-good template.
            self.template_path.write_text(DEFAULT_TEMPLATE, encoding="utf-8")
            return

        # Heuristic: old default had a single table header including Accuracy..F1 and R2/RMSE together.
        looks_like_old_default = (
            ("<h2>ML Metric" in existing or "<h2>ML Metrics" in existing)
            and ("<th>Accuracy</th>" in existing)
            and ("<th>Precision</th>" in existing)
            and ("<th>Recall</th>" in existing)
            and ("<th>F1</th>" in existing)
            and ("<th>R2</th>" in existing)
            and ("<th>RMSE</th>" in existing)
            and ("<th>MAE</th>" not in existing)
        )

        if looks_like_old_default:
            self.template_path.write_text(DEFAULT_TEMPLATE, encoding="utf-8")

    def categorise_figures(self, figures_root: Path) -> Dict[str, List[str]]:
        categories: Dict[str, List[str]] = defaultdict(list)

        if not figures_root.exists():
            return categories

        for path in sorted(figures_root.glob("*.png")):
            name = path.name

            if name.startswith("edahist"):
                categories["histograms"].append(name)
            elif name.startswith("edabox"):
                categories["boxplots"].append(name)
            elif name.startswith("edabar"):
                categories["barplots"].append(name)
            elif name.startswith("edacorrheatmap") or name.startswith("edaheatmap"):
                categories["corrheatmaps"].append(name)
            elif name.startswith("edaother"):
                categories["othereda"].append(name)
            elif name.startswith("mlconfusionmatrix"):
                categories["confusionmatrices"].append(name)
            elif name.startswith("mlplot"):
                categories["otherml"].append(name)
            else:
                categories["uncategorised"].append(name)

        return categories

    def run(self, report_dir: Path) -> Path:
        profile = self.bus.get_state("data_profile") or {}
        plan = self.bus.get_state("analysis_plan") or {}
        results = self.bus.get_state("analysis_results") or {}
        critique = self.bus.get_state("critique") or {}
        modelspec = self.bus.get_state("modelspec")
        best_model_summary = self.bus.get_state("bestmodelsummary")

        metrics = results.get("metrics", []) or []
        errors = results.get("errors", []) or []

        payload = {
            "profile": profile,
            "plan": plan,
            "metrics": metrics,
            "errors": errors,
            "critique": critique,
            "modelspec": modelspec,
            "bestmodelsummary": best_model_summary,
        }

        md_text, duration = self.llm.chat(
            REPORTER_SYSTEM,
            json.dumps(payload, indent=2),
            max_tokens=1000,
        )

        # Always append machine-readable blocks to the markdown.
        md_text = (md_text or "").strip()
        md_text += _json_md_block("Final ModelSpec (LLM-selected)", modelspec)
        md_text += _json_md_block("Best model summary", best_model_summary)

        html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

        figures_root = Path(report_dir) / "figures"
        figure_categories = self.categorise_figures(figures_root)

        tmpl = self.env.get_template("report.html.j2")

        html = tmpl.render(
            markdown_body=html_body,
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            metrics=metrics,
            errors=errors,
            critique=critique,
            figures=figure_categories,
        )

        out_path = Path(report_dir) / "report.html"
        out_path.write_text(html, encoding="utf-8")

        self.bus.publish(
            self.name,
            {
                "duration": duration,
                "reportpath": str(out_path),
                "templatesdir": str(self.templates_dir),
                "nfigurestotal": sum(len(v) for v in figure_categories.values()),
            },
        )

        return out_path
