"""
mlens/comparison/comparison_report.py
=======================================
Generates a side-by-side interactive HTML comparison report
for N models audited by ModelComparator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ComparisonReportGenerator:
    """
    Renders a ComparisonResult to a self-contained HTML file.

    Parameters
    ----------
    result : ComparisonResult
        Populated comparison from ModelComparator.compare().
    """

    def __init__(self, result: Any) -> None:
        self.result = result

    def render(self, path: str = "comparison_report.html") -> str:
        """Render the HTML report and return the output path."""
        html = self._build_html()
        Path(path).write_text(html, encoding="utf-8")
        print(f"[MLens] Comparison report saved → {Path(path).resolve()}")
        return str(Path(path).resolve())

    # ---------------------------------------------------------------- private

    def _build_html(self) -> str:
        r     = self.result
        table = r.comparison_table
        names = r.model_names

        # ── Chart data ────────────────────────────────────────────────────
        dp_gaps = [
            float(table.loc[n, "dp_gap"]) if "dp_gap" in table.columns
            and n in table.index else 0.0
            for n in names
        ]
        eo_gaps = [
            float(table.loc[n, "eo_gap"]) if "eo_gap" in table.columns
            and n in table.index else 0.0
            for n in names
        ]
        max_psi = [
            float(table.loc[n, "max_psi"]) if "max_psi" in table.columns
            and n in table.index else 0.0
            for n in names
        ]

        # ── Table rows ────────────────────────────────────────────────────
        table_rows = ""
        for name in names:
            row = table.loc[name] if name in table.index else {}
            is_fair      = row.get("is_fair", None)
            drift_status = row.get("drift_status", "—")
            fair_badge   = (
                '<span style="color:#639922">✓ Fair</span>' if is_fair is True
                else '<span style="color:#E24B4A">⚠ Flagged</span>' if is_fair is False
                else '—'
            )
            drift_color  = {
                "stable": "#639922", "moderate": "#EF9F27", "significant": "#E24B4A"
            }.get(str(drift_status), "#aaaaaa")

            table_rows += f"""
            <tr>
              <td style="color:#e8e8e8;font-family:monospace">{name}</td>
              <td>{row.get("top_feature", "—")}</td>
              <td>{row.get("top_feature_shap", "—")}</td>
              <td>{row.get("dp_gap", "—")}</td>
              <td>{row.get("eo_gap", "—")}</td>
              <td>{row.get("disparate_impact", "—")}</td>
              <td>{fair_badge}</td>
              <td>{row.get("n_drifted", "—")}</td>
              <td>{row.get("max_psi", "—")}</td>
              <td style="color:{drift_color};font-weight:600">{drift_status}</td>
              <td>{row.get("runtime_s", "—")}s</td>
            </tr>"""

        # ── Summary cards ─────────────────────────────────────────────────
        try:
            best_fair_name = table["dp_gap"].idxmin() if "dp_gap" in table.columns else "N/A"
            best_fair_val  = f"{table['dp_gap'].min():.4f}" if "dp_gap" in table.columns else "—"
        except Exception:
            best_fair_name, best_fair_val = "N/A", "—"

        try:
            best_drift_name = table["max_psi"].idxmin() if "max_psi" in table.columns else "N/A"
            best_drift_val  = f"{table['max_psi'].min():.4f}" if "max_psi" in table.columns else "—"
        except Exception:
            best_drift_name, best_drift_val = "N/A", "—"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MLens — Model Comparison Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#1a1a2e;color:#e8e8e8;font-family:'Segoe UI',system-ui,sans-serif;padding:2rem;max-width:1200px;margin:0 auto}}
  header{{border-bottom:2px solid #534AB7;padding-bottom:1rem;margin-bottom:2rem}}
  header h1{{font-size:22px;color:#7F77DD;margin-bottom:6px}}
  header p{{font-size:12px;color:#aaaaaa}}
  .meta-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:2rem}}
  .mc{{background:#16213e;border:1px solid #333355;border-radius:10px;padding:12px}}
  .mc-l{{font-size:11px;color:#aaaaaa;margin-bottom:3px}}
  .mc-v{{font-size:18px;font-weight:600}}
  .sec{{background:#16213e;border:1px solid #333355;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem}}
  .sec h2{{color:#7F77DD;font-size:14px;margin-bottom:1rem;padding-bottom:8px;border-bottom:1px solid #333355}}
  .charts{{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem}}
  .chart-card{{background:#16213e;border:1px solid #333355;border-radius:10px;padding:1rem}}
  .chart-card h3{{font-size:12px;color:#aaaaaa;margin-bottom:8px;font-family:monospace}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{background:#534AB7;color:#fff;padding:8px 10px;text-align:left;font-size:11px}}
  td{{padding:7px 10px;border-bottom:1px solid #333355;color:#aaaaaa}}
  tr:nth-child(even) td{{background:rgba(255,255,255,0.02)}}
  footer{{text-align:center;color:#aaaaaa;font-size:11px;margin-top:2rem;padding-top:1rem;border-top:1px solid #333355}}
</style>
</head>
<body>

<header>
  <h1>🔬 MLens Model Comparison Report</h1>
  <p>Compared {len(names)} models &nbsp;|&nbsp; {r.timestamp} &nbsp;|&nbsp; Runtime: {r.runtime_seconds:.2f}s &nbsp;|&nbsp; github.com/saiganesh47/mlens</p>
</header>

<div class="meta-row">
  <div class="mc"><div class="mc-l">Models compared</div><div class="mc-v">{len(names)}</div></div>
  <div class="mc"><div class="mc-l">Fairest model</div><div class="mc-v" style="font-size:13px">{best_fair_name} <small style="color:#aaa">({best_fair_val})</small></div></div>
  <div class="mc"><div class="mc-l">Least drifted</div><div class="mc-v" style="font-size:13px">{best_drift_name} <small style="color:#aaa">({best_drift_val})</small></div></div>
  <div class="mc"><div class="mc-l">Total runtime</div><div class="mc-v">{r.runtime_seconds:.2f}s</div></div>
</div>

<div class="charts">
  <div class="chart-card">
    <h3>Demographic Parity Gap ↓ lower is better</h3>
    <canvas id="dpChart" height="180"></canvas>
  </div>
  <div class="chart-card">
    <h3>Equalized Odds Gap ↓ lower is better</h3>
    <canvas id="eoChart" height="180"></canvas>
  </div>
  <div class="chart-card">
    <h3>Max PSI (Drift) ↓ lower is better</h3>
    <canvas id="psiChart" height="180"></canvas>
  </div>
</div>

<div class="sec">
  <h2>📊 Full Comparison Table</h2>
  <table>
    <tr>
      <th>Model</th><th>Top Feature</th><th>SHAP</th>
      <th>DP Gap</th><th>EO Gap</th><th>Disp. Impact</th>
      <th>Fair?</th><th>Drifted</th><th>Max PSI</th>
      <th>Drift Status</th><th>Runtime</th>
    </tr>
    {table_rows}
  </table>
</div>

<footer>Generated by <strong>MLens v0.6.0</strong> &nbsp;|&nbsp; github.com/saiganesh47/mlens</footer>

<script>
const names  = {json.dumps(names)};
const dpGaps = {json.dumps(dp_gaps)};
const eoGaps = {json.dumps(eo_gaps)};
const maxPsi = {json.dumps(max_psi)};
const COLORS = ['#534AB7','#1D9E75','#E24B4A','#EF9F27','#AFA9EC','#2dd4a8'];
const base = {{
  responsive:true, maintainAspectRatio:false,
  plugins:{{legend:{{display:false}}}},
  scales:{{
    x:{{grid:{{display:false}},ticks:{{color:'#aaaaaa',font:{{size:10}}}}}},
    y:{{grid:{{color:'rgba(255,255,255,0.07)'}},ticks:{{color:'#aaaaaa',font:{{size:10}}}}}}
  }}
}};
new Chart(document.getElementById('dpChart'),{{
  type:'bar',
  data:{{labels:names,datasets:[{{data:dpGaps,backgroundColor:COLORS,borderRadius:4,borderSkipped:false}}]}},
  options:{{...base}}
}});
new Chart(document.getElementById('eoChart'),{{
  type:'bar',
  data:{{labels:names,datasets:[{{data:eoGaps,backgroundColor:COLORS,borderRadius:4,borderSkipped:false}}]}},
  options:{{...base}}
}});
new Chart(document.getElementById('psiChart'),{{
  type:'bar',
  data:{{labels:names,datasets:[{{data:maxPsi,backgroundColor:COLORS,borderRadius:4,borderSkipped:false}}]}},
  options:{{...base}}
}});
</script>
</body>
</html>"""
