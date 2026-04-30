"""
mlens/report/html_generator.py
================================
Renders an interactive single-file HTML audit report
using Jinja2 templating and embedded Plotly charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np


class ReportGenerator:
    """
    Generates a self-contained HTML audit report.

    Parameters
    ----------
    report : AuditReport
        Populated audit report from ModelAuditor.run().
    """

    TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.html.j2"

    def __init__(self, report) -> None:
        self.report = report

    def render(self, output_path: Path) -> None:
        """Render the HTML report to output_path."""
        try:
            from jinja2 import Environment, FileSystemLoader
            env = Environment(
                loader=FileSystemLoader(str(self.TEMPLATE_PATH.parent))
            )
            template = env.get_template(self.TEMPLATE_PATH.name)
            html = template.render(**self._build_context())
        except Exception:
            # Fallback to inline template if Jinja2 not available
            html = self._inline_render()

        output_path.write_text(html, encoding="utf-8")

    # ---------------------------------------------------------------- private

    def _build_context(self) -> Dict[str, Any]:
        r = self.report
        ctx: Dict[str, Any] = {
            "model_name":       r.model_name,
            "timestamp":        r.audit_timestamp,
            "runtime":          f"{r.runtime_seconds:.2f}s",
            "summary_lines":    r.summary_lines,
            "metadata":         r.metadata,
            "shap":             None,
            "fairness":         None,
            "drift":            None,
            "shap_chart_json":  "null",
            "drift_chart_json": "null",
        }

        if r.shap_result:
            ctx["shap"]            = r.shap_result.top_features(n=15)
            ctx["shap_chart_json"] = self._shap_chart_json(r.shap_result)

        if r.fairness_result:
            ctx["fairness"] = r.fairness_result.to_dict()

        if r.drift_result:
            ctx["drift"]            = r.drift_result.to_dict()
            ctx["drift_chart_json"] = self._drift_chart_json(r.drift_result)

        return ctx

    def _shap_chart_json(self, shap_result) -> str:
        top     = shap_result.top_features(n=15)
        names   = [f["name"] for f in top][::-1]
        values  = [f["mean_abs_shap"] for f in top][::-1]
        max_val = max(values) if values else 1
        colors  = [
            "#534AB7" if v >= max_val * 0.7 else
            "#7F77DD" if v >= max_val * 0.4 else "#AFA9EC"
            for v in values
        ]
        chart = {
            "data": [{"type": "bar", "orientation": "h",
                      "x": values, "y": names,
                      "marker": {"color": colors},
                      "hovertemplate": "%{y}: %{x:.4f}<extra></extra>"}],
            "layout": {
                "title": "Global Feature Importance — Mean |SHAP|",
                "paper_bgcolor": "#1a1a2e", "plot_bgcolor": "#16213e",
                "font": {"color": "#e8e8e8"},
                "xaxis": {"title": "mean |SHAP value|", "gridcolor": "#333355"},
                "yaxis": {"gridcolor": "#333355"},
                "margin": {"l": 20, "r": 20, "t": 50, "b": 20},
                "height": 450,
            },
        }
        return json.dumps(chart)

    def _drift_chart_json(self, drift_result) -> str:
        color_map = {"significant": "#E24B4A", "moderate": "#EF9F27", "stable": "#639922"}
        features  = [f["feature"]    for f in drift_result.feature_results]
        psi_vals  = [f["psi"]        for f in drift_result.feature_results]
        statuses  = [f["psi_status"] for f in drift_result.feature_results]
        colors    = [color_map.get(s, "#888780") for s in statuses]
        chart = {
            "data": [{"type": "bar", "x": features, "y": psi_vals,
                      "marker": {"color": colors},
                      "hovertemplate": "%{x}<br>PSI: %{y:.4f}<extra></extra>"}],
            "layout": {
                "title": "PSI per Feature",
                "paper_bgcolor": "#1a1a2e", "plot_bgcolor": "#16213e",
                "font": {"color": "#e8e8e8"},
                "xaxis": {"gridcolor": "#333355"},
                "yaxis": {"title": "PSI", "gridcolor": "#333355"},
                "shapes": [
                    {"type": "line", "x0": -0.5, "x1": len(features) - 0.5,
                     "y0": 0.10, "y1": 0.10,
                     "line": {"color": "#EF9F27", "dash": "dash", "width": 1.5}},
                    {"type": "line", "x0": -0.5, "x1": len(features) - 0.5,
                     "y0": 0.25, "y1": 0.25,
                     "line": {"color": "#E24B4A", "dash": "dash", "width": 1.5}},
                ],
                "margin": {"l": 20, "r": 20, "t": 50, "b": 80},
                "height": 380,
            },
        }
        return json.dumps(chart)

    def _inline_render(self) -> str:
        """Minimal fallback HTML when Jinja2 is unavailable."""
        r   = self.report
        d   = r.to_dict()
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>MLens Audit — {r.model_name}</title>
<style>
  body{{background:#1a1a2e;color:#e8e8e8;font-family:monospace;padding:2rem;}}
  h1{{color:#7F77DD;}} pre{{background:#16213e;padding:1rem;border-radius:8px;}}
</style></head><body>
<h1>🔬 MLens Audit Report</h1>
<p>Model: <b>{r.model_name}</b> | {r.audit_timestamp} | {r.runtime_seconds:.2f}s</p>
<pre>{json.dumps(d, indent=2)}</pre>
</body></html>"""
