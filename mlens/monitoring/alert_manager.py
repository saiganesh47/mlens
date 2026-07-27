"""
mlens/monitoring/alert_manager.py
====================================
Send email or Slack alerts when drift or fairness
thresholds are breached during a scheduled audit.

Usage
-----
>>> from mlens.monitoring.alert_manager import AlertManager
>>> alert = AlertManager(
...     slack_webhook="https://hooks.slack.com/services/...",
...     email_to="team@company.com",
... )
>>> alert.check_and_notify(report)
"""

from __future__ import annotations

import json
import smtplib
import urllib.request
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional


# ── Alert config ───────────────────────────────────────────────────────────

@dataclass
class AlertConfig:
    """
    Thresholds that trigger alerts.

    Attributes
    ----------
    dp_gap_threshold : float
        Demographic parity gap above which to alert (default: 0.10).
    psi_threshold : float
        Per-feature PSI above which to alert (default: 0.25).
    n_drifted_threshold : int
        Number of drifted features above which to alert (default: 2).
    """
    dp_gap_threshold    : float = 0.10
    psi_threshold       : float = 0.25
    n_drifted_threshold : int   = 2


# ── Alert Manager ──────────────────────────────────────────────────────────

class AlertManager:
    """
    Monitor AuditReport results and send alerts via Slack or email.

    Parameters
    ----------
    slack_webhook : str, optional
        Slack incoming webhook URL.
    email_to : str or list, optional
        Recipient email address(es).
    email_from : str, optional
        Sender email address.
    smtp_host : str, optional
        SMTP server host (default: 'smtp.gmail.com').
    smtp_port : int, optional
        SMTP port (default: 587).
    smtp_password : str, optional
        SMTP/app password.
    config : AlertConfig, optional
        Custom alert thresholds.
    """

    def __init__(
        self,
        slack_webhook : Optional[str]        = None,
        email_to      : Optional[Any]        = None,
        email_from    : Optional[str]        = None,
        smtp_host     : str                  = "smtp.gmail.com",
        smtp_port     : int                  = 587,
        smtp_password : Optional[str]        = None,
        config        : Optional[AlertConfig]= None,
    ) -> None:
        self.slack_webhook = slack_webhook
        self.email_to      = [email_to] if isinstance(email_to, str) else email_to or []
        self.email_from    = email_from
        self.smtp_host     = smtp_host
        self.smtp_port     = smtp_port
        self.smtp_password = smtp_password
        self.config        = config or AlertConfig()

    # ---------------------------------------------------------------- public

    def check_and_notify(self, report: Any) -> List[str]:
        """
        Check an AuditReport against thresholds and send alerts if needed.

        Parameters
        ----------
        report : AuditReport
            Populated report from ModelAuditor.run().

        Returns
        -------
        list of str : Alert messages that were sent (empty if no alerts).
        """
        alerts = self._generate_alerts(report)

        if not alerts:
            print(f"[MLens Alert] ✅ No alerts for '{report.model_name}'.")
            return []

        message = self._format_message(report, alerts)

        if self.slack_webhook:
            self._send_slack(message)

        if self.email_to and self.email_from:
            self._send_email(report.model_name, message)

        print(f"[MLens Alert] ⚠️  {len(alerts)} alert(s) sent for '{report.model_name}'.")
        return alerts

    # --------------------------------------------------------------- private

    def _generate_alerts(self, report: Any) -> List[str]:
        """Evaluate thresholds and return a list of triggered alert messages."""
        alerts: List[str] = []
        cfg    = self.config

        # Fairness alerts
        if report.fairness_result:
            fr = report.fairness_result
            if fr.demographic_parity_gap > cfg.dp_gap_threshold:
                alerts.append(
                    f"⚠️ *Fairness Alert* — Demographic parity gap "
                    f"`{fr.demographic_parity_gap:.4f}` exceeds "
                    f"threshold `{cfg.dp_gap_threshold}`."
                )
            for flag in fr.flags:
                alerts.append(f"⚠️ *Fairness Flag* — {flag}")

        # Drift alerts
        if report.drift_result:
            dr = report.drift_result
            if dr.n_drifted >= cfg.n_drifted_threshold:
                drifted = ", ".join(dr.drifted_features()[:5])
                alerts.append(
                    f"📊 *Drift Alert* — {dr.n_drifted} feature(s) drifted "
                    f"(≥ {cfg.n_drifted_threshold} threshold): `{drifted}`."
                )
            for feat in dr.feature_results:
                if feat["psi"] > cfg.psi_threshold:
                    alerts.append(
                        f"📊 *High PSI* — `{feat['feature']}` PSI "
                        f"`{feat['psi']:.4f}` exceeds `{cfg.psi_threshold}`."
                    )

        return alerts

    def _format_message(self, report: Any, alerts: List[str]) -> str:
        """Format a plain-text/Markdown alert message."""
        lines = [
            f"🔬 *MLens Alert — {report.model_name}*",
            f"Audited at: {report.audit_timestamp}",
            f"Runtime: {report.runtime_seconds:.2f}s",
            "",
            f"{len(alerts)} issue(s) detected:",
            "",
        ]
        lines += [f"  {a}" for a in alerts]
        lines += [
            "",
            "View the full audit report for details.",
            "https://github.com/saiganesh47/mlens",
        ]
        return "\n".join(lines)

    def _send_slack(self, message: str) -> None:
        """Post a message to a Slack webhook."""
        payload = json.dumps({"text": message}).encode("utf-8")
        req     = urllib.request.Request(
            self.slack_webhook,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    print("[MLens Alert] ✅ Slack message sent.")
                else:
                    print(f"[MLens Alert] Slack returned status {resp.status}.")
        except Exception as exc:
            print(f"[MLens Alert] Slack send failed: {exc}")

    def _send_email(self, subject_model: str, body: str) -> None:
        """Send an alert email via SMTP."""
        if not self.smtp_password:
            print("[MLens Alert] Email skipped — no SMTP password provided.")
            return

        msg                     = MIMEMultipart("alternative")
        msg["Subject"]          = f"[MLens Alert] {subject_model} — Issues Detected"
        msg["From"]             = self.email_from
        msg["To"]               = ", ".join(self.email_to)
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_from, self.smtp_password)
                server.sendmail(self.email_from, self.email_to, msg.as_string())
            print(f"[MLens Alert] ✅ Email sent to {self.email_to}.")
        except Exception as exc:
            print(f"[MLens Alert] Email send failed: {exc}")
