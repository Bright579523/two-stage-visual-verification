# Two-Stage Visual Verification Platform
### Reducing False Alarms in Open-Vocabulary Industrial Computer Vision

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20Demo-blue)](https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/UI-Gradio%206-orange.svg)](https://gradio.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

This project explores whether a second visual verification step can reduce false alarms from open-vocabulary object detection while keeping recall above a predefined target. The pipeline couples candidate localization (OWL-ViT) with context-padded semantic verification (CLIP) and was evaluated on a 2,000-sample test set before being deployed to Hugging Face Spaces.

**Live Demo:** [https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification](https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification)  
**Direct Web App:** [https://bright87-two-stage-visual-verification.hf.space](https://bright87-two-stage-visual-verification.hf.space)

---

## 1. Problem & Motivation

Open-vocabulary object detectors (like OWL-ViT) can detect arbitrary categories from natural language queries, making them flexible for automated inspection. However, using them in visual monitoring presents a practical dilemma:

```
[Open-Vocabulary Detector (OWL-ViT)]
               │
               ▼
     High-Sensitivity Trigger
               │
  ┌────────────┴────────────┐
  ▼                         ▼
Genuine Hazards     High False-Alarm Volume
  │                         │
  ▼                         ▼
Safety Action       Alert Fatigue & Review Queue Overload
```

* **The Problem:** Setting a sensitive detection threshold flags genuine hazards, but also generates a high volume of false alarms (60% false positive rate at raw baseline). In a real triage queue, this creates severe alert fatigue for human reviewers. Conversely, raising the detection threshold to cut false alarms causes the model to miss genuine defects or safety breaches.
* **The Approach:** Instead of relying on a single confidence threshold, this pipeline decouples localization from verification:
  1. **Stage 1 (OWL-ViT):** Proposes candidate bounding boxes at high sensitivity ($V_1 \ge 0.05$).
  2. **Stage 2 (CLIP):** Checks whether the cropped region actually matches the target query ($V_2 \ge 0.20$) using a $10\%$ context-padded crop.
  3. **Decision Gate:** Passes the alert to the primary human triage queue only when both conditions are met (`AND` logic).

---

## 2. Pipeline Architecture

The system uses a single shared verification engine (`TwoStageVerificationEngine`) across inspection domains.

```
Input Image + Query Prompt
         │
         ▼
[Stage 1: OWL-ViT Candidate Localization] ──► Produces Bounding Box + V1 Score
         │
         ├─── If V1 < 0.05 ───► [Filtered: Not Confirmed]
         │                      *(Filtered from primary human triage queue)*
         ▼ (If V1 >= 0.05)
[10% Context-Padded Crop Extraction]
         │
         ▼
[Stage 2: CLIP Semantic Verification]    ──► Produces V2 Score on Crop
         │
         ├─── If V2 < 0.20 ───► [Filtered: Not Confirmed]
         │                      *(Filtered from primary human triage queue)*
         ▼ (If V2 >= 0.20)
[Alert Confirmed (AND Gate Passed)]      ──► Escalated to Human Triage Queue
```

### Context-Padded Crop Extraction
After Stage 1 finds a candidate, the system crops the detected region with 10% extra context before sending it to CLIP. This extra margin keeps boundary details from getting clipped and preserves surrounding spatial context:

$$\text{pad}_w = 0.10 \times \text{width}, \quad \text{pad}_h = 0.10 \times \text{height}$$

$$x_{\min}' = \max(0, x_{\min} - \text{pad}_w), \quad y_{\min}' = \max(0, y_{\min} - \text{pad}_h)$$
$$x_{\max}' = \min(W, x_{\max} + \text{pad}_w), \quad y_{\max}' = \min(H, y_{\max} + \text{pad}_h)$$

---

## 3. Benchmark Evaluation & Threshold Selection

### Dataset & Evaluation Protocol
* **Dataset:** 5,000 COCO-based benchmark samples consisting of 2,000 positive cases, 2,000 hard negatives, and 1,000 easy negatives. The benchmark evaluates prompt-image mismatch and false-positive filtering behavior; it is not a production industrial dataset.
* **Evaluation Split:** The dataset was split into a 60% Development Set (3,000 samples) for parameter sweeps and a 40% Frozen Test Set (2,000 samples: 800 positives, 1,200 negatives) held out for final evaluation.
* **Operating Constraint:** To ensure safety monitoring remains viable, thresholds were selected under a strict constraint: **Recall must remain $\ge 90.0\%$**.

### Frozen Test Set Results (2,000 Samples)

| Configuration | True Alerts (TP) | False Alarms (FP) | Recall | Precision | FAR Reduction | Reviews Avoided |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Stage 1 (Baseline)** | 800 | 1,200 | 100.0% | **40.00%** | 0.0% | 0 / 2,000 (0.0%) |
| **Stage 1 + V1 ($t_1=0.09$)** | 670 | 162 | 83.75% | 80.53% | 86.50% | 1,168 / 2,000 (58.4%) |
| **Stage 1 + V2 ($t_2=0.25$)** | 624 | 231 | 78.00% | 72.98% | 80.75% | 1,145 / 2,000 (57.3%) |
| **Strict AND ($t_1=0.09, t_2=0.25$)** | 563 | 73 | 70.37% | 88.52% | 93.92% | 1,364 / 2,000 (68.2%) |
| **OR Gate ($t_1=0.09, t_2=0.25$)** | 731 | 320 | 91.37% | 69.55% | 73.33% | 949 / 2,000 (47.5%) |
| **Selected Operating Point: AND ($t_1=0.05, t_2=0.20$) ★** | **726** | **279** | **90.75%** | **72.24%** | **76.75%** | **995 / 2,000 (49.8%)** |

> ★ **Selected Operating Point:** The strict AND rule ($t_1=0.09, t_2=0.25$) reduced false alarms by 93.92%, but recall dropped to 70.37%, missing nearly 30% of genuine alerts. By tuning thresholds on the Dev set to $V_1 \ge 0.05$ and $V_2 \ge 0.20$, the pipeline satisfied the $\ge 90\%$ recall constraint—retaining **90.75% recall**, reducing false alarms by **76.75%**, and filtering **995 of 2,000 benchmark cases (49.8%)** before the primary human-triage stage.

### Trade-off Curve & Confusion Matrix
![Operating Point Trade-off](assets/operating_point_tradeoff.png)
![Operating Point Confusion Matrix](assets/operating_point_cm.png)

---

## 4. Interactive Dashboard

A Gradio dashboard was built to demonstrate how this verification logic behaves in two industrial contexts:
1. **PPE Monitoring:** Checking whether workers in hazardous zones have required safety gear (e.g. hardhats, safety footwear).
2. **Assembly Inspection:** Checking for missing fasteners, metal shavings, and surface burrs on machined components.

*(Note: Safety and quality workflows are demonstration applications and are not validated for production safety or quality-control decisions.)*

![Dashboard Preview](assets/dashboard_preview.png)

### Key UI Features
* **Benchmark KPI Header:** Displays the validated benchmark metrics ($V_1 \ge 0.05, V_2 \ge 0.20$, Recall: 90.75%, FAR Reduction: 76.75%, Precision: 72.24%).
* **Side-by-Side Visualizer:** Shows the full image with the detected bounding box alongside the 10% context-padded crop.
* **Verification Evidence Card:** Displays $V_1$ and $V_2$ scores with clear PASS/FAIL indicators.
* **Simulated Human Triage:** Interactive buttons for operator disposition (`Acknowledge Alert`, `Dismiss as False Alarm`, `Flag for Review`).

---

## 5. Deployment & Runtime Performance

The application is hosted on Hugging Face Spaces with dynamic execution on ZeroGPU and CPU Basic.

### Measured Latency & Resource Benchmarks
| Metric | Measured Value | Notes |
| :--- | :---: | :--- |
| **Cold Model Loading** | **7.83 s** | Initializing OWL-ViT and CLIP on CPU |
| **First Inference (Cold)** | **1,009.5 ms** | Initial tensor allocation and warmup |
| **Average Warm Inference (CPU)** | **752.7 ms** | Stage 1: ~580 ms, Stage 2: ~170 ms |
| **Incremental Memory Delta** | **25.4 MB** | Working set ~550 MB, stable across runs |
| **Repeatability** | **Repeatable locally** | Local inference produced identical scores across repeated runs. Public GPU inference produced numerically close scores with identical final decisions |

---

## 6. Industrial Scenario Stress Test

To probe robustness beyond the benchmark, 10 photorealistic industrial scenarios were generated and evaluated across safety-monitoring and manufacturing-quality use cases:

| # | Domain / Scenario | Query Prompt | Stage 1 ($V_1$) | Stage 2 ($V_2$) | ROI Dimensions | Decision | Finding |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Forklift in Warehouse** | `a photo of a forklift` | **0.5799** (PASS) | **0.3003** (PASS) | $489 \times 340 \text{ px}$ | ⚠️ Confirmed | High-confidence vehicle detection |
| **2** | **Forklift Close-up (Step)** | `a photo of safety boots` | **0.2511** (PASS) | **0.3255** (PASS) | $220 \times 145 \text{ px}$ | ⚠️ Confirmed | Footwear verified during dismount |
| **3** | **Overhead Crane (Lifting)** | `a photo of an overhead crane` | **0.2753** (PASS) | **0.3344** (PASS) | $339 \times 415 \text{ px}$ | ⚠️ Confirmed | Suspended load & hoist identified |
| **4** | **Crane + Worker (Distant)** | `a photo of a factory worker` | **0.0310** (FAIL) | **0.2453** (PASS) | **$22 \times 43 \text{ px}$** | 🛡️ Filtered | **Failure Mode: Small/distant target** |
| **5** | **Safety Boots (Low-angle)** | `a photo of safety boots` | **0.2890** (PASS) | **0.3141** (PASS) | $410 \times 309 \text{ px}$ | ⚠️ Confirmed | Low-angle footwear verification |
| **6** | **PPE Breach (No Hardhat)** | `a photo of a worker without a hardhat` | **0.0647** (PASS) | **0.3264** (PASS) | $230 \times 667 \text{ px}$ | ⚠️ Confirmed | Violation confirmed by Stage 2 gate |
| **7** | **Single Abnormal Part** | `a photo of scrap metal debris` | **0.1026** (PASS) | **0.2111** (PASS) | $1020 \times 717 \text{ px}$ | ⚠️ Confirmed | Metallic scrap in groove detected |
| **8** | **Batch Inspection (1 in 12)** | `a photo of scrap metal debris` | **0.1153** (PASS) | **0.2216** (PASS) | $1330 \times 761 \text{ px}$ | ⚠️ Confirmed | Single defective part located in batch |
| **9** | **Macro Burr on Thread** | `a photo of a metal burr defect` | **0.1389** (PASS) | **0.2840** (PASS) | $622 \times 516 \text{ px}$ | ⚠️ Confirmed | High-contrast burr confirmed |
| **10** | **Production Tray Clutter** | `a photo of scrap metal debris` | **0.0399** (FAIL) | **0.2491** (PASS) | $463 \times 219 \text{ px}$ | 🛡️ Filtered | **Failure Mode: Stage 1 miss under visual clutter** |

---

## 7. Limitations & Failure Modes

The stress tests also showed where the current approach starts to fail:

### 1. Small and Distant Objects
* **Observation:** In Scenario #4 (crane bay surveillance), a distant worker occupying only $22 \times 43$ pixels scored $V_1 = 0.0310 < 0.05$, filtering the alert.
* **Limitation:** *Small or distant objects can reduce Stage 1 detection confidence, increasing the risk of missed alerts.* Very small or distant targets may fall below the Stage 1 detection threshold.
* **Potential Mitigation:** Higher-resolution cameras, tighter optical framing, or tiled inference could be evaluated in future work.

### 2. Sequential Gate Bottleneck (Unassisted Stage 1 Drops)
* **Observation:** In Scenario #10 (production tray with reflective cylinders), scrap debris amidst clutter scored $V_1 = 0.0399 < 0.05$, while Stage 2 scored $V_2 = 0.2491 > 0.20$.
* **Limitation:** *Stage 2 cannot rescue an alert when Stage 1 fails to propose a candidate passing the initial threshold.* Because the architecture is strictly sequential, any defect that fails the localization gate is pruned regardless of Stage 2 semantic alignment.

### 3. Visual Clutter & Specular Highlights
* **Observation:** Dense machining trays with multiple polished steel cylinders create complex reflection patterns, reducing contrast between defect shavings and base part geometries.
* **Limitation:** Strong specular highlights in open-vocabulary embeddings can depress detector confidence scores below default operating cutoffs.

### 4. Prompt Engineering & Canonical Phrasing Sensitivity
* **Observation:** Stage 1 confidence is sensitive to query structure:
  * Canonical phrasing (`"a photo of a worker without a hardhat"`): $V_1 = 0.1173$ (PASS)
  * Colloquial phrasing (`"person not wearing helmet"`): $V_1 = 0.0188$ (FAIL)
* **Note:** The reported operating point was selected using the canonical prompts used during evaluation; thresholds were not retuned for alternative phrasings. In deployment, inputs should use consistent canonical prompt templates rather than uncurated free text.

---

## 8. Quickstart & Installation

```bash
# 1. Clone repository
git clone https://github.com/Bright579523/hallucination-detection.git
cd hallucination-detection

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

## 9. Project Structure

```text
Advance AI Hallucination detect/
├── README.md                      # Project documentation and evaluation summary
├── REPORT.md                      # Full technical benchmark and progress report
├── BUSINESS_APPLICATION_SPEC.md   # Functional specification and triage workflows
├── assets/                        # Benchmark plots and dashboard preview images
│   ├── dashboard_preview.png
│   ├── operating_point_cm.png
│   ├── operating_point_tradeoff.png
│   ├── quality_alert_confirmed.png
│   └── safety_alert_confirmed.png
├── app/                           # Gradio application and verification engine
│   ├── engine.py                  # TwoStageVerificationEngine Singleton
│   ├── app.py                     # Multi-domain Gradio dashboard
│   └── samples/                   # Demonstration images
├── evaluation/                    # Benchmark scripts and test data
│   ├── evaluate_two_stage.py      # 5,000-sample pipeline evaluation
│   ├── optimize_operating_point.py# Threshold optimization under Recall >= 90%
│   ├── final_comparison_table.csv # Comparison table data
│   └── stress_test_results.json   # Initial stress test metrics
├── data/                          # Dataset and test runners
│   ├── stress_test_samples/       # 10 photorealistic industrial test images
│   └── run_10_tests.py            # Automated 10-scenario validation runner
└── hf_space/                      # Hugging Face Spaces deployment package
    ├── README.md                  # HF Spaces configuration & documentation
    ├── app.py                     # Deployment entrypoint with ZeroGPU support
    ├── engine.py                  # Dynamic device management engine
    ├── requirements.txt           # Minimal runtime dependencies
    └── samples/                   # Test samples tracked via Git LFS
```

---

## 10. License & Citation

Distributed under the Apache 2.0 License. See `LICENSE` for details.

```bibtex
@software{twostage_visual_verification_2026,
  author = {Watcharapon Anaketanaset},
  title = {Two-Stage Visual Verification Platform: False Alarm Reduction for Industrial Computer Vision},
  year = {2026},
  url = {https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification}
}
```
