# Business Application Specification: Two-Stage Visual Verification Engine
**Document Version:** 1.0  
**Phase:** 2B-1 (System Architecture & Operational Workflow Design)  
**Target Applications:** Workplace Safety Monitoring & Quality Assembly Inspection  

---

## 1. Executive Summary & Objective

This specification establishes the operational decision workflow, terminology, and UI structure for deploying the **Two-Stage Visual Verification Engine** into operational decision-support applications. 

The primary business purpose of this system is to act as an intelligent filter: **suppress false alarms while reliably preserving genuine hazard and defect alerts**, thereby cutting down unnecessary human review workload.

### System Boundaries & Anti-Scope Constraints
* **Core Engine:** Pre-trained Open-Vocabulary Detector (`OWL-ViT`) + Visual-Semantic Verifier (`CLIP`).
* **Frozen Operating Point:** Optimization completed in Phase 2A. Rules are frozen:
  $$\text{Decision Rule} = (Score_{V1} \ge 0.05) \land (Score_{V2} \ge 0.20)$$
* **Strict Constraints:**
  - ❌ No new model training or architecture changes.
  - ❌ No runtime threshold re-optimization or dynamic tuning.
  - ❌ No Top-K heuristic candidate picking.
  - ❌ No LLM or generative explanations.
  - ❌ No real-time video stream ingestion or RTSP integration (static sample verification only).
  - ❌ No claims of real-world labor savings; all metrics must be explicitly cited as benchmark validation findings.

---

## 2. Reusable Core Workflow (Shared Engine)

Both the **Safety Monitoring** and **Quality Inspection** applications share an identical verification engine and human-in-the-loop workflow.

```text
[ Inspection Image Input ]
           ↓
[ Stage 1: Broad Candidate Detection (OWL-ViT) ]
  - Scans full image with wide recall bias
  - Generates candidate bounding box & detection score (V1)
           ↓
[ Targeted Crop Extraction ]
  - Extracts bounding box region of interest for verification
           ↓
[ Stage 2: Visual Semantic Verification (CLIP) ]
  - Computes semantic similarity between crop and target prompt (V2)
           ↓
[ AI Verification Gate (Frozen AND Logic) ]
  - Condition: (V1 >= 0.05) AND (V2 >= 0.20)
           ↓
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
[ Alert Confirmed ]                                 [ Not Confirmed (Filtered) ]
(Both verification conditions satisfied)            (Filtered from triage queue)
  - Enters Operator High-Priority Queue               - Auto-cleared / Logged for Audit
           ↓                                                        ↓
[ Human Review & Disposition ]                      [ Optional Periodic Audit ]
  - Operator verifies physical scene                  - Quality audit on filtered cases
  - Dispositions:                                     - Dispositions:
    • Acknowledge / Dispatch Action                     • Confirmed Correct Rejection
    • Dismiss as False Alarm                            • Flag Missed Detection
    • Flag for Supervisor Review
```

---

## 3. Standardized Terminology & Role Separation

To avoid confusion between machine predictions and human operational responsibilities, terminology is strictly partitioned:

| Term | Category | Definition & Scope |
| :--- | :--- | :--- |
| **Stage 1 Score ($V1$)** | Model Metric | Raw detection confidence score from OWL-ViT candidate detection. |
| **Stage 2 Score ($V2$)** | Model Metric | Normalized cosine similarity score from CLIP semantic verification. |
| **Alert Confirmed** | **AI Decision** | Binary system output (`1`). Both verification conditions are satisfied ($V1 \ge 0.05$ and $V2 \ge 0.20$). Escalated to operator queue. |
| **Not Confirmed (Filtered)** | **AI Decision** | Binary system output (`0`). The alert does not pass the verification gate and is filtered from the primary human triage queue. |
| **Human Review** | **Human Workflow** | **Operational queue state**, NOT an AI prediction class. Represents the manual inspection task performed by human operators on escalated alerts. |
| **Human Disposition** | **Human Workflow** | Final action taken by an operator (`Acknowledgement Recorded`, `Dismissed`, `Escalated`). |

---

## 4. Multi-Domain Application Tabs

The user interface provides two dedicated domain tabs backed by the exact same engine:

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ Two-Stage Visual Verification Platform                                      │
├──────────────────────────────────────┬─────────────────────────────────────┤
│ 🛡️ Application 1: Safety Monitoring   │ 🔍 Application 2: Quality Inspection │
└──────────────────────────────────────┴─────────────────────────────────────┘
```

### Application 1: Workplace & Logistics Safety Monitoring
* **Core Operational Question:** *"Is there an active safety hazard or compliance violation requiring immediate intervention?"*
* **Example Scenarios:**
  - PPE non-compliance (missing helmet, missing high-visibility vest).
  - Hazard zone encroachment (pedestrian detected inside forklift operational envelope).
  - Logistics obstructions (pallets blocking fire exits or designated emergency walkways).
* **Sample Prompts:** `"person without safety helmet"`, `"forklift near pedestrian"`, `"aisle obstruction pallet"`.

### Application 2: Quality & Assembly Inspection
* **Core Operational Question:** *"Is there a component defect, missing part, or assembly anomaly on the unit?"*
* **Example Scenarios:**
  - Missing assembly fasteners (missing screw, missing bolt).
  - Incorrect component alignment or missing enclosure covers.
  - Connector anomalies (unlatched harness, loose terminal).
* **Sample Prompts:** `"missing bolt fastener"`, `"unlatched wire connector"`, `"missing enclosure cover"`.

---

## 5. UI Layout & Component Wireframe

The interface is structured into three clean functional zones:

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ [TOP BAR] Benchmark Performance & Verification Health                                         │
│ • False Alarm Reduction: 76.75%*  • Recall Retention: 90.75%*  • Cases Avoided: 49.8% (995)* │
│   *Benchmark validation results on 2,000-sample frozen test set.                             │
├──────────────────────────────────────────────────────────────┬───────────────────────────────┤
│ [LEFT COLUMN] Input & Inspection Workspace                   │ [RIGHT COLUMN] Verification   │
│                                                              │ Evidence & AI Decision        │
│ 1. Preset / Image Selection                                  │                               │
│    [ Sample 1 (True Alert) ] [ Sample 2 (False Alarm) ]      │ 3. Score Breakdown            │
│    [ Upload Custom Inspection Image ]                        │    • Stage 1 (OWL-ViT):  0.14 │
│                                                              │      [ Threshold: >= 0.05 ]   │
│ 2. Visual Inspection Workspace                               │    • Stage 2 (CLIP):     0.28 │
│    ┌──────────────────────────┬───────────────────────────┐  │      [ Threshold: >= 0.20 ]   │
│    │ Original Image + BBox    │ Target Crop (Stage 2 ROI) │  │    • Logic: AND Gate          │
│    │                          │                           │  │                               │
│    │                          │                           │  │ 4. AI Decision Output        │
│    │                          │                           │  │    [ ⚠️ ALERT CONFIRMED ]    │
│    └──────────────────────────┴───────────────────────────┘  │                               │
│                                                              │ 5. Human Workflow Controls    │
│    Prompt: [ "missing hardhat"                          ]    │    [ Acknowledge & Dispatch ] │
│    [ ⚡ Run Verification ]                                    │    [ Dismiss (False Alarm) ] │
│                                                              │    [ Flag for Review       ] │
└──────────────────────────────────────────────────────────────┴───────────────────────────────┘
```

### Detailed Component Specifications

1. **Benchmark KPI Banner:**
   - Displays static validated benchmark performance metrics across the top.
   - Distinct footnote clearly stating that metrics originate from the frozen 2,000-sample test set.
2. **Visual Inspection Workspace:**
   - **Primary Display:** Full input frame rendered with bounding box overlay from Stage 1 candidate generation.
   - **Secondary Display:** Cropped Region of Interest (ROI) passed to Stage 2, showing the exact visual evidence evaluated by CLIP.
3. **Evidence & Score Panel:**
   - Displays numerical $V1$ and $V2$ scores alongside their respective cutoff thresholds ($0.05$ and $0.20$).
   - Color-coded status pills indicating whether each gate passed or failed.
4. **AI Decision Banner:**
   - Large, unmistakable status badge:
     - **`⚠️ Alert Confirmed`** (Warning Amber/Red): Both verification conditions satisfied; alert escalated to operator queue.
     - **`✅ Not Confirmed (Filtered)`** (Muted Green/Gray): The alert does not pass the verification gate and is filtered from the primary human triage queue.
5. **Human Workflow Actions (Simulated Triage):**
   - Action buttons enabling operators to register human dispositions (`Acknowledge`, `Dismiss`, `Flag for Review`).
   - Logging audit trail demonstrating end-to-end human-in-the-loop integration.

---

## 6. Validated Benchmark KPI Definitions

All metrics presented in the UI header are locked to the Phase 2A frozen test set results:

| KPI Label | Validated Value | Benchmark Definition & Scope |
| :--- | :---: | :--- |
| **Recall Retention** | **90.75%** | The selected operating point detected 90.75% of positive benchmark samples (726 / 800 TP). |
| **False Alarm Reduction** | **76.75%** | Eliminates 76.75% of baseline false alarms (from 1,200 down to 279 FP). |
| **Review Workload Avoided** | **49.8%** | **995 of 2,000 benchmark cases** were filtered before the human-review stage under this workflow assumption. |
| **Precision** | **72.24%** | Among alerts confirmed by the system, the proportion that are genuine alerts (726 TP / 1,005 Total Alerts Confirmed on benchmark test set). |

*Mandatory Display Footnote:*  
> *"All metrics reflect empirical performance on a 2,000-sample frozen COCO-based benchmark dataset. Workplace Safety Monitoring and Quality Inspection are demonstration applications illustrating domain generalization. Field metrics in production will vary based on camera setup, lighting, and site conditions."*

---

## 7. Acceptance Criteria for Phase 2B-2 Implementation

Before writing code for the Gradio interface in Phase 2B-2, the implementation must strictly satisfy:

1. **Architecture Parity:** Both Safety and Quality tabs must call the exact same underlying verification function.
2. **Threshold Adherence:** Hardcoded decision parameters: $V1 \ge 0.05$, $V2 \ge 0.20$, rule: `AND`.
3. **Visual Transparency:** The interface must show both the full bounding box and the cropped patch.
4. **No Phantom Classes:** The AI model output must remain strictly binary (`Alert Confirmed` vs `Not Confirmed`). Any "Review" state must exist exclusively in the human workflow triage section.
5. **Zero Feature Creep:** No new models, no external APIs, no RTSP streams, and no dynamic threshold tuning sliders on the operational dashboard.
