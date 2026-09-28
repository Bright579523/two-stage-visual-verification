# From AI Hallucination to Factory Floor
### Two-Stage Visual Verification for Open-Vocabulary Industrial Computer Vision

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20Demo-blue)](https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/UI-Gradio%206-orange.svg)](https://gradio.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

> **Try it now:** [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification) · [Direct Web App](https://bright87-two-stage-visual-verification.hf.space)

![Dashboard Preview](assets/dashboard_preview.png)

---

## What This Does

Open-vocabulary detectors like OWL-ViT can find arbitrary objects from natural language queries — but when used for automated monitoring, they generate **60% false positives at sensitive thresholds**. Human operators experience alert fatigue, review queues back up, and teams either raise the threshold (missing real hazards) or stop trusting the system altogether.

This pipeline adds a **second verification step** before escalating alerts to human reviewers:

1. **Stage 1 (OWL-ViT)** proposes candidate regions at high sensitivity
2. **Stage 2 (CLIP)** verifies whether the cropped region actually matches the query
3. **AND Gate** — alerts reach human review only when both stages agree

### Results on 2,000-Sample Frozen Test Set

| | Raw Baseline (Stage 1 Alone) | **Two-Stage Pipeline** |
|:---|:---:|:---:|
| **Recall** | 100.0% | **90.75%** |
| **Precision** | 40.0% | **72.24%** |
| **False Alarm Reduction** | — | **76.75%** |
| **Cases Filtered Before Human Review** | 0 / 2,000 | **995 / 2,000 (49.8%)** |

> Thresholds ($V_1 \ge 0.05$, $V_2 \ge 0.20$) were selected on a separate 3,000-sample Dev set under a strict constraint: **Recall must stay ≥ 90%**. The test set was never used for tuning.

---

## How It Works

```
Input Image + Query Prompt
         │
         ▼
[Stage 1: OWL-ViT Candidate Localization] ──► Bounding Box + V1 Score
         │
         ├─── V1 < 0.05 ───► Filtered (no human review needed)
         │
         ▼ (V1 ≥ 0.05)
[Crop with 10% Context Margin]
         │
         ▼
[Stage 2: CLIP Semantic Verification]     ──► V2 Score on Crop
         │
         ├─── V2 < 0.20 ───► Filtered (no human review needed)
         │
         ▼ (V2 ≥ 0.20)
[Alert Confirmed — Escalated to Human Triage Queue]
```

**Why the 10% context margin?** When cropping around the detected object, adding a 10% border keeps surrounding spatial context visible to CLIP — preventing edge clipping from distorting the verification result.

---

## Where It Breaks — Failure Modes

The system was stress-tested across 10 photorealistic industrial scenarios (PPE monitoring, forklift operations, crane surveillance, machined parts, scrap detection). Three operational limitations were identified:

### 1. Small & Distant Objects
A distant factory worker occupying only 22×43 pixels scored $V_1 = 0.031$, falling below the Stage 1 threshold. Very small or distant targets reduce detection confidence to the point of missed alerts.

### 2. Sequential Gate Bottleneck
When Stage 1 misses a candidate ($V_1 < 0.05$), Stage 2 never gets a chance to verify it — even if the semantic match would have passed. This is a structural property of the sequential architecture: Stage 2 cannot rescue what Stage 1 drops.

### 3. Visual Clutter & Specular Highlights
Dense machining trays with polished steel cylinders create complex reflection patterns that suppress open-vocabulary embedding confidence below operating thresholds.

### 4. Prompt Sensitivity
Canonical phrasing (`"a photo of a worker without a hardhat"`) scored $V_1 = 0.117$ (PASS), while colloquial phrasing (`"person not wearing helmet"`) dropped to $V_1 = 0.019$ (FAIL). Production deployments should enforce consistent prompt templates.

<details>
<summary><strong>📊 Full Stress Test Results (10 Scenarios)</strong></summary>

| # | Scenario | Query Prompt | V1 | V2 | ROI Size | Decision | Finding |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | Forklift in Warehouse | `a photo of a forklift` | 0.5799 ✓ | 0.3003 ✓ | 489×340 | ⚠️ Confirmed | High-confidence vehicle detection |
| 2 | Forklift Close-up (Step) | `a photo of safety boots` | 0.2511 ✓ | 0.3255 ✓ | 220×145 | ⚠️ Confirmed | Footwear verified during dismount |
| 3 | Overhead Crane (Lifting) | `a photo of an overhead crane` | 0.2753 ✓ | 0.3344 ✓ | 339×415 | ⚠️ Confirmed | Suspended load & hoist identified |
| 4 | Crane + Worker (Distant) | `a photo of a factory worker` | 0.0310 ✗ | 0.2453 ✓ | **22×43** | 🛡️ Filtered | **Failure: Small/distant target** |
| 5 | Safety Boots (Low-angle) | `a photo of safety boots` | 0.2890 ✓ | 0.3141 ✓ | 410×309 | ⚠️ Confirmed | Low-angle footwear verification |
| 6 | PPE Breach (No Hardhat) | `a photo of a worker without a hardhat` | 0.0647 ✓ | 0.3264 ✓ | 230×667 | ⚠️ Confirmed | Violation confirmed by Stage 2 |
| 7 | Single Abnormal Part | `a photo of scrap metal debris` | 0.1026 ✓ | 0.2111 ✓ | 1020×717 | ⚠️ Confirmed | Metallic scrap in groove detected |
| 8 | Batch Inspection (1 in 12) | `a photo of scrap metal debris` | 0.1153 ✓ | 0.2216 ✓ | 1330×761 | ⚠️ Confirmed | Single defective part in batch |
| 9 | Macro Burr on Thread | `a photo of a metal burr defect` | 0.1389 ✓ | 0.2840 ✓ | 622×516 | ⚠️ Confirmed | High-contrast burr confirmed |
| 10 | Production Tray Clutter | `a photo of scrap metal debris` | 0.0399 ✗ | 0.2491 ✓ | 463×219 | 🛡️ Filtered | **Failure: Stage 1 miss under clutter** |

</details>

---

## Interactive Dashboard

The Gradio dashboard demonstrates verification logic across two industrial contexts with a shared engine:

1. **PPE Monitoring** — hardhats, safety footwear, personnel near crane loads
2. **Assembly Inspection** — missing fasteners, metal shavings, surface burrs

*(These are demonstration workflows, not validated for production safety or quality-control decisions.)*

### Dashboard Features
* **Benchmark KPI Header** — validated metrics displayed at a glance
* **Side-by-Side Visualizer** — full image with bounding box alongside the 10% context-padded crop
* **Verification Evidence Card** — V1 and V2 scores with PASS/FAIL indicators
* **Simulated Human Triage** — interactive disposition buttons (Acknowledge / Dismiss / Flag)

<details>
<summary><strong>📊 Full Benchmark Comparison (All Strategies)</strong></summary>

| Configuration | TP | FP | Recall | Precision | FAR Reduction | Reviews Avoided |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Stage 1 (Baseline)** | 800 | 1,200 | 100.0% | 40.00% | 0.0% | 0 / 2,000 (0.0%) |
| **Stage 1 + V1 ($t_1=0.09$)** | 670 | 162 | 83.75% | 80.53% | 86.50% | 1,168 / 2,000 (58.4%) |
| **Stage 1 + V2 ($t_2=0.25$)** | 624 | 231 | 78.00% | 72.98% | 80.75% | 1,145 / 2,000 (57.3%) |
| **Strict AND ($t_1=0.09, t_2=0.25$)** | 563 | 73 | 70.37% | 88.52% | 93.92% | 1,364 / 2,000 (68.2%) |
| **OR Gate ($t_1=0.09, t_2=0.25$)** | 731 | 320 | 91.37% | 69.55% | 73.33% | 949 / 2,000 (47.5%) |
| **Selected: AND ($t_1=0.05, t_2=0.20$) ★** | **726** | **279** | **90.75%** | **72.24%** | **76.75%** | **995 / 2,000 (49.8%)** |

> ★ The strict AND rule ($t_1=0.09, t_2=0.25$) achieved 93.92% FAR reduction but recall dropped to 70.37%. By relaxing thresholds on the Dev set, the selected operating point retained ≥ 90% recall while still cutting false alarms by 76.75%.

![Operating Point Trade-off](assets/operating_point_tradeoff.png)
![Operating Point Confusion Matrix](assets/operating_point_cm.png)

</details>

<details>
<summary><strong>⏱️ Runtime Performance & Repeatability</strong></summary>

| Metric | Value | Notes |
| :--- | :---: | :--- |
| **Cold Model Loading** | 7.83 s | OWL-ViT + CLIP initialization on CPU |
| **First Inference (Cold)** | 1,009.5 ms | Initial tensor allocation and warmup |
| **Average Warm Inference (CPU)** | 752.7 ms | Stage 1: ~580 ms, Stage 2: ~170 ms |
| **Incremental Memory Delta** | 25.4 MB | Working set ~550 MB, stable across runs |
| **Repeatability** | Repeatable locally | Identical scores across repeated local runs. Public GPU inference produced numerically close scores with identical decisions |

</details>

---

## Quickstart

```bash
# 1. Clone repository
git clone https://github.com/Bright579523/two-stage-visual-verification.git
cd two-stage-visual-verification

# 2. Set up virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r hf_space/requirements.txt

# 4. Run the dashboard locally
python app/app.py
```
Access the application locally at: `http://127.0.0.1:7860`

### Run Stress Test Suite
```bash
python data/run_10_tests.py
```

---

## Project Structure

```text
two-stage-visual-verification/
├── README.md                      # This file
├── REPORT.md                      # Full technical report
├── BUSINESS_APPLICATION_SPEC.md   # Functional spec and triage workflows
├── assets/                        # Benchmark plots and dashboard screenshots
├── app/                           # Gradio application and verification engine
│   ├── engine.py                  # TwoStageVerificationEngine
│   ├── app.py                     # Multi-domain Gradio dashboard
│   └── samples/                   # Demonstration images
├── evaluation/                    # Benchmark scripts and test data
│   ├── evaluate_two_stage.py      # 5,000-sample pipeline evaluation
│   ├── optimize_operating_point.py# Threshold optimization under Recall >= 90%
│   └── final_comparison_table.csv # Comparison table data
├── data/                          # Dataset and test runners
│   ├── stress_test_samples/       # 10 photorealistic industrial test images
│   └── run_10_tests.py            # Automated 10-scenario validation runner
└── hf_space/                      # Hugging Face Spaces deployment package
    ├── app.py                     # ZeroGPU-aware entrypoint
    ├── engine.py                  # Dynamic device management engine
    ├── requirements.txt           # Runtime dependencies
    └── samples/                   # Test samples (Git LFS)
```

---

## License

Distributed under the Apache 2.0 License. See `LICENSE` for details.
