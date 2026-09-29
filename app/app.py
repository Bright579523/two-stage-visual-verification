"""
app.py — Two-Stage Visual Verification Dashboard (Phase 2B-2)
Multi-Domain Decision Support for False Alarm Reduction:
  - Application 1: Workplace Safety Monitoring
  - Application 2: Quality Assembly Inspection

Backed by a single shared TwoStageVerificationEngine.
Frozen Operating Point: V1 >= 0.05 AND V2 >= 0.20.
"""

import os
import sys
from datetime import datetime
from PIL import Image
import gradio as gr

# Ensure root directory is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.engine import (
    TwoStageVerificationEngine,
    V1_THRESHOLD,
    V2_THRESHOLD,
    DECISION_LOGIC
)

# ---------------------------------------------------------------------------
# Path configuration for demonstration sample images
# ---------------------------------------------------------------------------
SAMPLES_DIR = os.path.join(CURRENT_DIR, "samples")

SAFETY_SAMPLE_ALERT = os.path.join(SAMPLES_DIR, "safety_violation_no_helmet.jpg")
SAFETY_SAMPLE_COMPLIANT = os.path.join(SAMPLES_DIR, "safety_compliant_with_helmet.jpg")

QUALITY_SAMPLE_ALERT = os.path.join(SAMPLES_DIR, "quality_defect_missing_bolt.jpg")
QUALITY_SAMPLE_COMPLIANT = os.path.join(SAMPLES_DIR, "quality_compliant_assembly.jpg")

# ---------------------------------------------------------------------------
# Custom CSS for clean, professional executive styling (Dark & Light Mode)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    max-width: 1240px !important;
    margin: auto !important;
}

/* Light Mode Defaults */
:root {
    --card-bg: #ffffff;
    --card-border: #e2e8f0;
    --card-title: #475569;
    --card-desc: #64748b;
    --card-val: #0f172a;
    --fn-bg: #f8fafc;
    --fn-border: #e2e8f0;
    --fn-accent: #0284c7;
    --fn-text: #334155;
    --fn-strong: #0f172a;
    --sc-bg: #ffffff;
    --sc-border: #e2e8f0;
    --sc-header: #0f172a;
    --sc-label: #1e293b;
    --sc-sub: #64748b;
    --sc-val: #0f172a;
    --sc-meta: #475569;
    --sc-divider: #e2e8f0;
    --dec-alert-bg: #fef2f2;
    --dec-alert-border: #ef4444;
    --dec-alert-title: #b91c1c;
    --dec-alert-desc: #334155;
    --dec-filt-bg: #f0fdf4;
    --dec-filt-border: #22c55e;
    --dec-filt-title: #15803d;
    --dec-filt-desc: #334155;
}

/* Dark Mode Overrides via OS preference */
@media (prefers-color-scheme: dark) {
    :root {
        --card-bg: #1e293b;
        --card-border: #334155;
        --card-title: #94a3b8;
        --card-desc: #94a3b8;
        --card-val: #f8fafc;
        --fn-bg: #0f172a;
        --fn-border: #1e293b;
        --fn-accent: #38bdf8;
        --fn-text: #cbd5e1;
        --fn-strong: #f8fafc;
        --sc-bg: #1e293b;
        --sc-border: #334155;
        --sc-header: #f8fafc;
        --sc-label: #e2e8f0;
        --sc-sub: #94a3b8;
        --sc-val: #f8fafc;
        --sc-meta: #94a3b8;
        --sc-divider: #334155;
        --dec-alert-bg: rgba(239, 68, 68, 0.18);
        --dec-alert-border: #ef4444;
        --dec-alert-title: #f87171;
        --dec-alert-desc: #fca5a5;
        --dec-filt-bg: rgba(34, 197, 94, 0.18);
        --dec-filt-border: #22c55e;
        --dec-filt-title: #4ade80;
        --dec-filt-desc: #86efac;
    }
}

/* Dark Mode Overrides via Gradio theme class */
.dark, body.dark, .gradio-container.dark {
    --card-bg: #1e293b;
    --card-border: #334155;
    --card-title: #94a3b8;
    --card-desc: #94a3b8;
    --card-val: #f8fafc;
    --fn-bg: #0f172a;
    --fn-border: #1e293b;
    --fn-accent: #38bdf8;
    --fn-text: #cbd5e1;
    --fn-strong: #f8fafc;
    --sc-bg: #1e293b;
    --sc-border: #334155;
    --sc-header: #f8fafc;
    --sc-label: #e2e8f0;
    --sc-sub: #94a3b8;
    --sc-val: #f8fafc;
    --sc-meta: #94a3b8;
    --sc-divider: #334155;
    --dec-alert-bg: rgba(239, 68, 68, 0.18);
    --dec-alert-border: #ef4444;
    --dec-alert-title: #f87171;
    --dec-alert-desc: #fca5a5;
    --dec-filt-bg: rgba(34, 197, 94, 0.18);
    --dec-filt-border: #22c55e;
    --dec-filt-title: #4ade80;
    --dec-filt-desc: #86efac;
}

/* Layout & Elements */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 8px;
}

.kpi-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: left;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

.kpi-title {
    font-size: 0.80rem;
    font-weight: 700;
    color: var(--card-title) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

.kpi-value {
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.2;
}

.kpi-desc {
    font-size: 0.75rem;
    color: var(--card-desc) !important;
    margin-top: 4px;
    line-height: 1.3;
}

.benchmark-footnote {
    font-size: 0.78rem;
    color: var(--fn-text) !important;
    background: var(--fn-bg) !important;
    border: 1px solid var(--fn-border) !important;
    border-left: 4px solid var(--fn-accent) !important;
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 20px;
    line-height: 1.45;
}

.benchmark-footnote strong {
    color: var(--fn-strong) !important;
}

.score-card {
    background: var(--sc-bg) !important;
    border: 1px solid var(--sc-border) !important;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

.score-header {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--sc-header) !important;
    margin-bottom: 10px;
    border-bottom: 1px solid var(--sc-divider);
    padding-bottom: 6px;
}

.score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.score-label {
    font-weight: 600;
    font-size: 0.82rem;
    color: var(--sc-label) !important;
}

.score-sub {
    font-size: 0.74rem;
    color: var(--sc-sub) !important;
}

.score-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--sc-val) !important;
}

.score-footer {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px dashed var(--sc-divider);
    font-size: 0.76rem;
    color: var(--sc-meta) !important;
    display: flex;
    justify-content: space-between;
}

.score-meta {
    font-size: 0.74rem;
    color: var(--sc-sub) !important;
    margin-top: 4px;
    display: flex;
    justify-content: space-between;
}

.gate-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.gate-pass {
    background: #dcfce7 !important;
    color: #15803d !important;
    border: 1px solid #bbf7d0 !important;
}

.dark .gate-pass, body.dark .gate-pass, .gradio-container.dark .gate-pass {
    background: rgba(34, 197, 94, 0.2) !important;
    color: #4ade80 !important;
    border: 1px solid #22c55e !important;
}

.gate-fail {
    background: #fee2e2 !important;
    color: #b91c1c !important;
    border: 1px solid #fecaca !important;
}

.dark .gate-fail, body.dark .gate-fail, .gradio-container.dark .gate-fail {
    background: rgba(239, 68, 68, 0.2) !important;
    color: #f87171 !important;
    border: 1px solid #ef4444 !important;
}

.decision-alert {
    background: var(--dec-alert-bg) !important;
    border: 1.5px solid var(--dec-alert-border) !important;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    margin-bottom: 14px;
}

.decision-alert .decision-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--dec-alert-title) !important;
    margin: 4px 0;
}

.decision-alert .decision-desc {
    font-size: 0.82rem;
    color: var(--dec-alert-desc) !important;
}

.decision-filtered {
    background: var(--dec-filt-bg) !important;
    border: 1.5px solid var(--dec-filt-border) !important;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    margin-bottom: 14px;
}

.decision-filtered .decision-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--dec-filt-title) !important;
    margin: 4px 0;
}

.decision-filtered .decision-desc {
    font-size: 0.82rem;
    color: var(--dec-filt-desc) !important;
}

.human-action-box {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 8px;
    padding: 14px;
    margin-top: 14px;
}
"""


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------
def render_kpi_header():
    """Renders the frozen benchmark performance summary bar."""
    return f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-title">Recall Retention</div>
            <div class="kpi-value" style="color: #0284c7;">90.75%*</div>
            <div class="kpi-desc">726 / 800 positive benchmark cases detected</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">False Alarm Reduction</div>
            <div class="kpi-value" style="color: #16a34a;">76.75%*</div>
            <div class="kpi-desc">Reduced from 1,200 down to 279 baseline false alarms</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Review Workload Avoided</div>
            <div class="kpi-value" style="color: #7c3aed;">49.8%*</div>
            <div class="kpi-desc">995 / 2,000 cases filtered before human review</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Benchmark Precision</div>
            <div class="kpi-value" style="color: #ea580c;">72.24%*</div>
            <div class="kpi-desc">Proportion of confirmed alerts that are genuine</div>
        </div>
    </div>
    <div class="benchmark-footnote">
        <strong>* Benchmark Validation Findings:</strong> All metrics reflect empirical results on a 2,000-sample frozen COCO-based benchmark dataset under the frozen decision rule (V1 &ge; {V1_THRESHOLD:.2f} AND V2 &ge; {V2_THRESHOLD:.2f}).
        Workplace Safety Monitoring and Quality Inspection tabs are demonstration applications showing architectural domain generalization. Operational site performance will vary based on camera setup, lighting, and resolution.
    </div>
    """


def render_decision_card(res):
    """Renders the AI decision card and evidence breakdown."""
    if res is None:
        return "<div style='color: #94a3b8; text-align: center; padding: 24px;'>Click 'Run Verification' to inspect image.</div>"

    is_confirmed = res["is_confirmed"]
    decision = res["decision"]
    v1_score = res["v1_score"]
    v1_pass = res["v1_pass"]
    v2_score = res["v2_score"]
    v2_pass = res["v2_pass"]
    dim = res["box_dimensions"]
    lat_v1 = res["latency_v1_ms"]
    lat_v2 = res["latency_v2_ms"]
    res_warn = res["resolution_warning"]

    if is_confirmed:
        card_class = "decision-alert"
        badge_icon = "⚠️"
        badge_title = "ALERT CONFIRMED"
        badge_desc = "Both verification conditions are satisfied. Escalated to primary operator triage queue."
    else:
        card_class = "decision-filtered"
        badge_icon = "✅"
        badge_title = "NOT CONFIRMED (FILTERED)"
        badge_desc = "The alert does not pass the verification gate and is filtered from the primary human triage queue."

    pill_v1 = f'<span class="gate-pill {"gate-pass" if v1_pass else "gate-fail"}">{"PASS" if v1_pass else "FAIL"}</span>'
    pill_v2 = f'<span class="gate-pill {"gate-pass" if v2_pass else "gate-fail"}">{"PASS" if v2_pass else "FAIL"}</span>'

    warning_html = ""
    if res_warn:
        warning_html = f"""
        <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-left: 3px solid #f59e0b; padding: 8px 10px; border-radius: 4px; font-size: 0.76rem; color: #d97706; margin-top: 10px;">
            <strong>Notice:</strong> Crop size is small ({dim}). 10% context padding was applied to protect CLIP feature resolution.
        </div>
        """

    return f"""
    <div class="{card_class}">
        <div style="font-size: 1.8rem; line-height: 1;">{badge_icon}</div>
        <div class="decision-title">{badge_title}</div>
        <div class="decision-desc">{badge_desc}</div>
    </div>

    <div class="score-card">
        <div class="score-header">
            VERIFICATION EVIDENCE &amp; GATE STATUS
        </div>
        
        <div class="score-row">
            <div>
                <span class="score-label">Stage 1: OWL-ViT Confidence (V1)</span>
                <div class="score-sub">Cutoff: &ge; {V1_THRESHOLD:.2f}</div>
            </div>
            <div style="text-align: right;">
                <span class="score-val">{v1_score:.4f}</span>
                <span style="margin-left: 6px;">{pill_v1}</span>
            </div>
        </div>

        <div class="score-row">
            <div>
                <span class="score-label">Stage 2: CLIP Semantic Similarity (V2)</span>
                <div class="score-sub">Cutoff: &ge; {V2_THRESHOLD:.2f}</div>
            </div>
            <div style="text-align: right;">
                <span class="score-val">{v2_score:.4f}</span>
                <span style="margin-left: 6px;">{pill_v2}</span>
            </div>
        </div>

        <div class="score-footer">
            <span><strong>Rule:</strong> AND Logic (V1 &ge; {V1_THRESHOLD:.2f} &amp; V2 &ge; {V2_THRESHOLD:.2f})</span>
            <span><strong>ROI:</strong> {dim}</span>
        </div>
        <div class="score-meta">
            <span>V1 Latency: {lat_v1:.1f}ms</span>
            <span>V2 Latency: {lat_v2:.1f}ms</span>
        </div>
        {warning_html}
    </div>
    """


# ---------------------------------------------------------------------------
# Backend execution handlers
# ---------------------------------------------------------------------------
def run_verification(image, prompt):
    """Executes verification and returns visual and textual evidence."""
    if image is None:
        raise gr.Error("Please upload or select an inspection image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a target query prompt.")

    engine = TwoStageVerificationEngine.get_instance()
    res = engine.verify(image, prompt)

    html_evidence = render_decision_card(res)
    return res["annotated_image"], res["crop_image"], html_evidence


def handle_human_disposition(action_type):
    """Logs simulated human operator action."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if action_type == "ACKNOWLEDGE":
        msg = f"✅ [{now_str}] OPERATOR ACTION ACKNOWLEDGED: Alert confirmed by operator. Acknowledgement recorded."
        color = "#15803d"
    elif action_type == "DISMISS":
        msg = f"⚪ [{now_str}] OPERATOR ACTION DISMISSED: Operator confirmed as False Alarm. Logged to false-alarm audit trail."
        color = "#64748b"
    elif action_type == "REVIEW":
        msg = f"🚩 [{now_str}] ESCALATED FOR SUPERVISOR REVIEW: Flagged for secondary inspection by shift manager."
        color = "#b45309"
    else:
        msg = "No action registered."
        color = "#64748b"

    return f"<div style='color: {color}; font-weight: 600; font-size: 0.82rem;'>{msg}</div>"


# ---------------------------------------------------------------------------
# Application Tab Builder
# ---------------------------------------------------------------------------
def create_app_tab(
    tab_title: str,
    domain_description: str,
    sample_alert_path: str,
    sample_compliant_path: str,
    default_prompt_alert: str,
    default_prompt_compliant: str,
    alert_preset_label: str,
    compliant_preset_label: str
):
    """Builds a standardized application tab backed by the shared engine."""
    with gr.Tab(tab_title):
        gr.Markdown(f"**Domain Context:** {domain_description}")

        with gr.Row():
            # -----------------------------------------------------
            # Left Column: Inputs & Visual Inspection Workspace
            # -----------------------------------------------------
            with gr.Column(scale=5):
                gr.Markdown("#### 1. Input & Target Definition")
                
                preset_selector = gr.Radio(
                    choices=[
                        alert_preset_label,
                        compliant_preset_label,
                        "Custom Upload"
                    ],
                    value=alert_preset_label,
                    label="Inspection Scenarios (Presets or Custom)",
                    info="Choose a pre-configured scenario to demonstrate alert confirmation vs filtering."
                )

                input_image = gr.Image(
                    value=sample_alert_path if os.path.exists(sample_alert_path) else None,
                    type="pil",
                    label="Inspection Image (CCTV / Automated Inspection Camera)",
                    height=320
                )

                prompt_input = gr.Textbox(
                    value=default_prompt_alert,
                    label="Target Query Prompt",
                    placeholder="e.g. a photo of a worker",
                    info="Natural language target query evaluated across both stages."
                )

                run_btn = gr.Button("⚡ Run Two-Stage Verification", variant="primary", size="lg")

                gr.Markdown("#### 2. Visual Evidence")
                with gr.Row():
                    annotated_out = gr.Image(
                        label="Stage 1: Candidate Proposal Bounding Box",
                        interactive=False,
                        height=260
                    )
                    crop_out = gr.Image(
                        label="Stage 2: Verification Crop (ROI Evaluated by CLIP)",
                        interactive=False,
                        height=260
                    )

            # -----------------------------------------------------
            # Right Column: Decision, Scores & Human Triage
            # -----------------------------------------------------
            with gr.Column(scale=4):
                gr.Markdown("#### 3. AI Verification Decision")
                decision_card_out = gr.HTML(value=render_decision_card(None))

                gr.Markdown("#### 4. Simulated Human-in-the-Loop Triage")
                gr.Markdown(
                    "<span style='font-size: 0.80rem; color: #64748b;'>"
                    "The AI system provides binary triage (Confirmed vs Filtered). "
                    "Human operators make the final operational disposition below."
                    "</span>"
                )

                with gr.Row():
                    btn_ack = gr.Button("Acknowledge Alert", variant="secondary", size="sm")
                    btn_dismiss = gr.Button("Dismiss as FA", variant="secondary", size="sm")
                    btn_review = gr.Button("Flag for Review", variant="secondary", size="sm")

                human_status_out = gr.HTML(
                    value="<div style='color: #94a3b8; font-size: 0.80rem;'>Awaiting operator disposition...</div>"
                )

        # Preset selection logic
        def on_preset_change(choice):
            if choice == alert_preset_label and os.path.exists(sample_alert_path):
                img = Image.open(sample_alert_path)
                return img, default_prompt_alert
            elif choice == compliant_preset_label and os.path.exists(sample_compliant_path):
                img = Image.open(sample_compliant_path)
                return img, default_prompt_compliant
            return None, ""

        preset_selector.change(
            fn=on_preset_change,
            inputs=[preset_selector],
            outputs=[input_image, prompt_input]
        )

        # Verification button click
        run_btn.click(
            fn=run_verification,
            inputs=[input_image, prompt_input],
            outputs=[annotated_out, crop_out, decision_card_out]
        )

        # Human actions
        btn_ack.click(
            fn=lambda: handle_human_disposition("ACKNOWLEDGE"),
            outputs=[human_status_out]
        )
        btn_dismiss.click(
            fn=lambda: handle_human_disposition("DISMISS"),
            outputs=[human_status_out]
        )
        btn_review.click(
            fn=lambda: handle_human_disposition("REVIEW"),
            outputs=[human_status_out]
        )


# ---------------------------------------------------------------------------
# Main Gradio Application Dashboard
# ---------------------------------------------------------------------------
def build_app():
    """Assembles the executive Two-Stage Verification Gradio dashboard."""
    with gr.Blocks(title="Zero-Shot VLM Hallucination Diagnostics") as demo:
        gr.Markdown(
            """
            # 🔬 Zero-Shot VLM Hallucination & Verification Diagnostics
            ### Research Investigation: Reliability Limits of OWL-ViT + CLIP in Industrial Visual Monitoring

            > ⚠️ **Research Demonstration — Not Production Ready:**
            > This dashboard is a **diagnostic research tool** evaluating the reliability limits of zero-shot vision-language models. In real-world industrial tests, zero-shot models **are not accurate enough for autonomous safety monitoring**: CLIP struggles with prompt negation (often flagging compliant workers as violations), and OWL-ViT misses small or distant targets. Use this interface to analyze where and why detections fail.
            """
        )

        # Render Benchmark KPI Header
        gr.HTML(render_kpi_header())

        # Tab 1: Safety Monitoring
        create_app_tab(
            tab_title="🛡️ Workplace Safety Monitoring",
            domain_description="Active hazard & compliance monitoring (PPE compliance, forklift-pedestrian boundaries, walkway obstructions).",
            sample_alert_path=SAFETY_SAMPLE_ALERT,
            sample_compliant_path=SAFETY_SAMPLE_COMPLIANT,
            default_prompt_alert="a photo of a worker without a hardhat",
            default_prompt_compliant="a photo of a worker without a hardhat",
            alert_preset_label="Scenario 1: PPE Safety Breach (Worker without Hardhat)",
            compliant_preset_label="Scenario 2: Compliant PPE (Worker wearing Hardhat)"
        )

        # Tab 2: Quality Inspection
        create_app_tab(
            tab_title="🔍 Quality & Assembly Inspection",
            domain_description="Component presence & assembly integrity verification (missing fasteners, unseated connectors, cover anomalies).",
            sample_alert_path=QUALITY_SAMPLE_ALERT,
            sample_compliant_path=QUALITY_SAMPLE_COMPLIANT,
            default_prompt_alert="a photo of an empty bolt hole",
            default_prompt_compliant="a photo of an empty bolt hole",
            alert_preset_label="Scenario 1: Fastener Defect (Empty Bolt Hole)",
            compliant_preset_label="Scenario 2: Fully Assembled Unit (All Bolts Fastened)"
        )

        gr.Markdown(
            """
            ---
            <div style="font-size: 0.78rem; color: #94a3b8; text-align: center;">
                Zero-Shot VLM Diagnostic Engine | OWL-ViT Candidate Detection + CLIP Visual-Semantic Verification | Research Operating Point: AND Gate (V1 &ge; 0.05, V2 &ge; 0.20)
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    print("[App] Pre-loading TwoStageVerificationEngine...")
    _ = TwoStageVerificationEngine.get_instance()

    server_port = int(os.environ.get("PORT", 7860))
    server_name = os.environ.get("SERVER_NAME", "0.0.0.0")
    print(f"[App] Launching Gradio Dashboard on {server_name}:{server_port}...")
    app = build_app()
    app.launch(server_name=server_name, server_port=server_port, show_error=True, css=CUSTOM_CSS)


