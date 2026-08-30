"""
mlens/automl/auto_report.py
=============================
Combines all advisors into one unified action plan report.
Runs ModelRecommender, FairnessAdvisor, DriftAdvisor, and ShapAdvisor,
then writes a single prioritised HTML action plan.

Usage
-----
>>> from mlens.automl.auto_report import AutoReport
>>> auto = AutoReport(report, X_train, y_train)
>>> auto.generate("action_plan.html")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class AutoReport:
    """
    All-in-one intelligent audit action plan.

    Orchestrates all four advisors and produces a single
    prioritised HTML improvement plan.

    Parameters
    ----------
    report : AuditReport
        Populated report from ModelAuditor.run().
    X_train : array-like
    y_train : array-like
    """

    def __init__(self, report: Any, X_train: Any, y_train: Any) -> None:
        self.report  = report
        self.X_train = X_train
        self.y_train = y_train

    # ---------------------------------------------------------------- public

    def generate(self, path: str = "mlens_action_plan.html") -> str:
        """
        Run all advisors and save an HTML action plan.

        Returns
        -------
        str : output path
        """
        print("\n[MLens AutoReport] Generating action plan …")

        # ── Run all advisors ───────────────────────────────────────────────
        from mlens.automl.model_recommender         import ModelRecommender
        from mlens.recommendations.fairness_advisor import FairnessAdvisor
        from mlens.recommendations.drift_advisor    import DriftAdvisor
        from mlens.recommendations.shap_advisor     import ShapAdvisor

        rec_result  = ModelRecommender(self.report, self.X_train, self.y_train).recommend()
        fair_plan   = FairnessAdvisor(self.report).advise()
        drift_plan  = DriftAdvisor(self.report).advise()
        shap_plan   = ShapAdvisor(self.report).advise()

        html = self._render(rec_result, fair_plan, drift_plan, shap_plan)
        Path(path).write_text(html, encoding="utf-8")
        print(f"[MLens AutoReport] Action plan saved → {Path(path).resolve()}")
        return str(Path(path).resolve())

    # --------------------------------------------------------------- private

    def _render(self, rec, fair, drift, shap) -> str:
        # ── Build action sections ──────────────────────────────────────────
        def action_card(priority, title, explanation, code=""):
            colors = {
                "critical": "#E24B4A", "high": "#EF9F27",
                "medium": "#7F77DD",   "low": "#1D9E75",
            }
            c = colors.get(priority, "#aaaaaa")
            code_html = (
                f'<pre style="background:#14141f;border-radius:6px;padding:12px;'
                f'font-size:11px;overflow-x:auto;margin-top:8px;color:#2dd4a8">'
                f'{code.replace("<", "&lt;").replace(">", "&gt;")}</pre>'
            ) if code else ""
            return f"""
            <div style="border-left:3px solid {c};background:rgba(255,255,255,0.03);
                        border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:10px;">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <span style="background:{c}22;color:{c};padding:2px 10px;border-radius:12px;
                             font-size:11px;font-weight:600;font-family:monospace">{priority}</span>
                <span style="font-size:14px;font-weight:500">{title}</span>
              </div>
              <p style="font-size:12px;color:#aaaaaa;margin:0">{explanation}</p>
              {code_html}
            </div>"""

        # ── Model recommendations ──────────────────────────────────────────
        model_cards = ""
        for s in rec.top(n=4):
            model_cards += f"""
            <div style="background:#16213e;border:1px solid #333355;border-radius:8px;
                        padding:14px;margin-bottom:10px">
              <div style="font-size:13px;font-weight:500;color:#8b80f0;margin-bottom:4px">
                {s.model_name}
              </div>
              <div style="font-size:11px;color:#aaaaaa;margin-bottom:6px">{s.reason}</div>
              <code style="font-size:11px;color:#2dd4a8;background:rgba(45,212,168,0.08);
                           padding:2px 8px;border-radius:4px">{s.model_class}</code>
              <span style="font-size:11px;color:#aaaaaa;margin-left:8px">
                → {s.expected_improvement}</span>
            </div>"""

        # ── Reasoning summary ──────────────────────────────────────────────
        reasoning_html = "".join(
            f'<div style="padding:8px 12px;margin-bottom:6px;background:rgba(139,128,240,0.07);'
            f'border-radius:6px;font-size:12px;color:#ece9f7">{r}</div>'
            for r in rec.reasoning_summary
        )

        # ── Fairness actions ───────────────────────────────────────────────
        fair_html = ""
        for a in fair.actions[:4]:
            fair_html += action_card(a.priority, a.title, a.explanation, a.code_snippet)

        # ── Drift actions ──────────────────────────────────────────────────
        drift_html = ""
        for a in drift.actions[:3]:
            drift_html += action_card(a.priority, a.title, a.explanation, a.code_snippet)

        # ── SHAP insights ──────────────────────────────────────────────────
        shap_html = ""
        for i in shap.insights[:3]:
            shap_html += action_card(i.priority, i.title, i.explanation, i.code_snippet)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MLens Action Plan — {self.report.model_name}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#1a1a2e;color:#e8e8e8;font-family:'Segoe UI',system-ui,sans-serif;
       padding:2rem;max-width:1000px;margin:0 auto;font-size:14px}}
  header{{border-bottom:2px solid #534AB7;padding-bottom:1rem;margin-bottom:2rem}}
  header h1{{font-size:22px;color:#7F77DD;margin-bottom:6px}}
  header p{{font-size:12px;color:#aaaaaa}}
  .meta{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:2rem}}
  .mc{{background:#16213e;border:1px solid #333355;border-radius:10px;padding:12px}}
  .mc-l{{font-size:11px;color:#aaaaaa;margin-bottom:3px}}
  .mc-v{{font-size:16px;font-weight:600}}
  .sec{{background:#16213e;border:1px solid #333355;border-radius:12px;
        padding:1.5rem;margin-bottom:1.5rem}}
  .sec h2{{color:#7F77DD;font-size:14px;margin-bottom:1rem;
           padding-bottom:8px;border-bottom:1px solid #333355}}
  footer{{text-align:center;color:#aaaaaa;font-size:11px;margin-top:2rem;
          padding-top:1rem;border-top:1px solid #333355}}
</style>
</head>
<body>

<header>
  <h1>🤖 MLens Action Plan — {self.report.model_name}</h1>
  <p>Generated: {self.report.audit_timestamp} &nbsp;|&nbsp;
     Retraining urgency: <strong style="color:{'#E24B4A' if drift.retraining_urgency == 'immediate' else '#EF9F27' if drift.retraining_urgency == 'scheduled' else '#639922'}">{drift.retraining_urgency.upper()}</strong> &nbsp;|&nbsp;
     github.com/saiganesh47/mlens</p>
</header>

<div class="meta">
  <div class="mc"><div class="mc-l">Model</div>
    <div class="mc-v" style="font-size:12px">{self.report.model_name}</div></div>
  <div class="mc"><div class="mc-l">Fairness</div>
    <div class="mc-v" style="color:{'#639922' if self.report.fairness_result and self.report.fairness_result.is_fair else '#E24B4A'}">
      {'✓ Fair' if self.report.fairness_result and self.report.fairness_result.is_fair else '⚠ Flagged'}</div></div>
  <div class="mc"><div class="mc-l">Drift</div>
    <div class="mc-v" style="color:{'#639922' if drift.overall_status == 'stable' else '#EF9F27' if drift.overall_status == 'moderate' else '#E24B4A'}">{drift.overall_status.upper()}</div></div>
  <div class="mc"><div class="mc-l">Model suggestions</div>
    <div class="mc-v">{len(rec.suggestions)}</div></div>
</div>

<div class="sec">
  <h2>🏆 Better Model Recommendations</h2>
  <div style="margin-bottom:12px">{reasoning_html}</div>
  {model_cards}
</div>

<div class="sec">
  <h2>⚖️ Fairness Fixes — {fair.summary}</h2>
  {fair_html if fair_html else '<p style="color:#aaaaaa;font-size:12px">No fairness issues detected.</p>'}
</div>

<div class="sec">
  <h2>📊 Drift Remediation — {drift.retraining_schedule}</h2>
  {drift_html}
</div>

<div class="sec">
  <h2>🧠 Feature Engineering Insights — {shap.summary}</h2>
  {shap_html if shap_html else '<p style="color:#aaaaaa;font-size:12px">No SHAP insights available.</p>'}
</div>

<footer>
  Generated by <strong>MLens v0.7.0 AutoReport</strong> &nbsp;|&nbsp;
  github.com/saiganesh47/mlens
</footer>

</body>
</html>"""
