"""
mlens/batch/batch_report.py
==============================
Generates a master HTML summary report for a BatchAuditSummary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BatchReportGenerator:
    """
    Renders a BatchAuditSummary to a single HTML summary page.

    Parameters
    ----------
    summary : BatchAuditSummary
        Populated result from BatchAuditor.run().
    """

    def __init__(self, summary: Any) -> None:
        self.summary = summary

    def render(self, path: str = "batch_summary.html") -> str:
        """Render and save the HTML report."""
        html = self._build_html()
        Path(path).write_text(html, encoding="utf-8")
        print(f"[MLens Batch] Summary report saved → {Path(path).resolve()}")
        return str(Path(path).resolve())

    # ---------------------------------------------------------------- private

    def _build_html(self) -> str:
        s     = self.summary
        table = s.summary_table

        # ── Table rows ────────────────────────────────────────────────────
        rows_html = ""
        for name, row in table.iterrows():
            is_fair  = row.get("is_fair", None)
            drift    = row.get("drift_status", "—")
            error    = row.get("error", None)
            report_p = row.get("report_path", None)

            fair_badge = (
                '<span style="color:#639922">✓ Fair</span>' if is_fair is True
                else '<span style="color:#E24B4A">⚠ Flagged</span>' if is_fair is False
                else '—'
            )
            drift_color = {
                "stable": "#639922", "moderate": "#EF9F27",
                "significant": "#E24B4A"
            }.get(str(drift), "#aaaaaa")

            report_link = (
                f'<a href="{report_p}" style="color:#7F77DD">View</a>'
                if report_p else "—"
            )
            error_cell = (
                f'<span style="color:#E24B4A">{str(error)[:40]}</span>'
                if error else '<span style="color:#639922">✓</span>'
            )

            rows_html += f"""
            <tr>
              <td style="color:#e8e8e8;font-family:monospace">{name}</td>
              <td>{row.get("top_feature", "—")}</td>
              <td>{row.get("dp_gap", "—")}</td>
              <td>{row.get("eo_gap", "—")}</td>
              <td>{fair_badge}</td>
              <td>{row.get("n_drifted", "—")}</td>
              <td>{row.get("max_psi", "—")}</td>
              <td style="color:{drift_color};font-weight:600">{drift}</td>
              <td>{row.get("runtime_s", "—")}s</td>
              <td>{error_cell}</td>
              <td>{report_link}</td>
            </tr>"""

        # ── PSI chart data ─────────────────────────────────────────────────
        names_list = list(table.index)
        psi_list   = [
            float(table.loc[n, "max_psi"])
            if "max_psi" in table.columns and n in table.index else 0.0
            for n in names_list
        ]
        dp_list = [
            float(table.loc[n, "dp_gap"])
            if "dp_gap" in table.columns and n in table.index else 0.0
            for n in names_list
        ]

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MLens — Batch Audit Summary</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#1a1a2e;color:#e8e8e8;font-family:'Segoe UI',system-ui,sans-serif;padding:2rem;max-width:1200px;margin:0 auto;font-size:14px}}
  header{{border-bottom:2px solid #534AB7;padding-bottom:1rem;margin-bottom:2rem}}
  header h1{{font-size:22px;color:#7F77DD;margin-bottom:6px}}
  header p{{font-size:12px;color:#aaaaaa}}
  .meta-row{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:2rem}}
  .mc{{background:#16213e;border:1px solid #333355;border-radius:10px;padding:12px}}
  .mc-l{{font-size:11px;color:#aaaaaa;margin-bottom:3px}}
  .mc-v{{font-size:20px;font-weight:600}}
  .charts{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.5rem}}
  .chart-card{{background:#16213e;border:1px solid #333355;border-radius:10px;padding:1rem}}
  .chart-card h3{{font-size:12px;color:#aaaaaa;margin-bottom:8px;font-family:monospace}}
  .sec{{background:#16213e;border:1px solid #333355;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem}}
  .sec h2{{color:#7F77DD;font-size:14px;margin-bottom:1rem;padding-bottom:8px;border-bottom:1px solid #333355}}
  .fail-box{{background:rgba(226,75,74,0.07);border-left:3px solid #E24B4A;border-radius:0 6px 6px 0;padding:8px 12px;margin-top:8px;font-size:12px;color:#ef9a9a;font-family:monospace}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{background:#534AB7;color:#fff;padding:8px 10px;text-align:left;font-size:11px}}
  td{{padding:7px 10px;border-bottom:1px solid #333355;color:#aaaaaa}}
  tr:nth-child(even) td{{background:rgba(255,255,255,0.02)}}
  footer{{text-align:center;color:#aaaaaa;font-size:11px;margin-top:2rem;padding-top:1rem;border-top:1px solid #333355}}
</style>
</head>
<body>

<header>
  <h1>🔬 MLens Batch Audit Summary</h1>
  <p>Runtime: {s.runtime_seconds:.2f}s &nbsp;|&nbsp; Output: {s.output_dir} &nbsp;|&nbsp; github.com/saiganesh47/mlens</p>
</header>

<div class="meta-row">
  <div class="mc"><div class="mc-l">Models audited</div><div class="mc-v">{s.n_models}</div></div>
  <div class="mc"><div class="mc-l">Succeeded</div><div class="mc-v" style="color:#639922">{s.n_success}</div></div>
  <div class="mc"><div class="mc-l">Failed</div><div class="mc-v" style="color:{'#E24B4A' if s.n_failed else '#639922'}">{s.n_failed}</div></div>
  <div class="mc"><div class="mc-l">Success rate</div><div class="mc-v">{100*s.n_success//s.n_models if s.n_models else 0}%</div></div>
  <div class="mc"><div class="mc-l">Total runtime</div><div class="mc-v">{s.runtime_seconds:.2f}s</div></div>
</div>

<div class="charts">
  <div class="chart-card">
    <h3>Max PSI per model ↓ lower is better</h3>
    <canvas id="psiChart" height="200"></canvas>
  </div>
  <div class="chart-card">
    <h3>Demographic parity gap ↓ lower is better</h3>
    <canvas id="dpChart" height="200"></canvas>
  </div>
</div>

<div class="sec">
  <h2>📋 Model Summary Table</h2>
  <table>
    <tr>
      <th>Model</th><th>Top Feature</th><th>DP Gap</th><th>EO Gap</th>
      <th>Fair?</th><th>Drifted</th><th>Max PSI</th>
      <th>Drift Status</th><th>Runtime</th><th>Status</th><th>Report</th>
    </tr>
    {rows_html}
  </table>
</div>

{'<div class="sec"><h2>❌ Failed Models</h2>' + "".join(f'<div class="fail-box">{n}: {e}</div>' for n,e in s.failed_models.items()) + "</div>" if s.failed_models else ""}

<footer>Generated by <strong>MLens v0.6.0</strong> &nbsp;|&nbsp; github.com/saiganesh47/mlens</footer>

<script>
const names  = {json.dumps(names_list)};
const psiVals= {json.dumps(psi_list)};
const dpVals = {json.dumps(dp_list)};
const COLORS = ['#534AB7','#1D9E75','#E24B4A','#EF9F27','#AFA9EC','#2dd4a8','#f0625f','#f0a847'];
const base = {{
  responsive:true, maintainAspectRatio:false,
  plugins:{{legend:{{display:false}}}},
  scales:{{
    x:{{grid:{{display:false}},ticks:{{color:'#aaa',font:{{size:10}}}}}},
    y:{{grid:{{color:'rgba(255,255,255,0.07)'}},ticks:{{color:'#aaa',font:{{size:10}}}}}}
  }}
}};
new Chart(document.getElementById('psiChart'),{{
  type:'bar',data:{{labels:names,datasets:[{{data:psiVals,backgroundColor:COLORS,borderRadius:4,borderSkipped:false}}]}},
  options:{{...base}}
}});
new Chart(document.getElementById('dpChart'),{{
  type:'bar',data:{{labels:names,datasets:[{{data:dpVals,backgroundColor:COLORS,borderRadius:4,borderSkipped:false}}]}},
  options:{{...base}}
}});
</script>
</body>
</html>"""
