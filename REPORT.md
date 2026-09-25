# Visual Verification System: Phase 1 & Phase 2 Technical & Application Report

## Phase 1 — Evaluation & Validation

The first phase focused on evaluating the two-stage visual verification system using a real dataset of 5,000 samples.

The evaluation compared the original detection stage (Stage 1) with several verification strategies, including:

* Stage 1 only
* Stage 1 + V1 verification
* Stage 1 + V2 verification
* Stage 1 + V1 + V2 using AND logic
* Stage 1 + V1 + V2 using OR logic

A separate development set and frozen test set were used to avoid tuning the system on the same data used for final evaluation.

The results showed that the verification stage can significantly reduce false alarms. However, a stronger filtering rule also increases the risk of missing genuine alerts.

For example:

* The strict AND rule reduced false alarms by **93.92%**, but recall dropped to **70.38%**.
* This showed that simply maximizing false-alarm reduction is not sufficient for a practical application.
* The system therefore needed a more balanced operating point between **false-alarm reduction** and **recall**.

### Key Finding

The main lesson from Phase 1 was:

> A practical visual verification system must reduce unnecessary alerts while still detecting most of the genuine alerts.

---

## Phase 2A — Operating Point Optimization

The second phase focused on finding a more practical decision rule.

A business constraint was defined:

> **Recall must remain at or above 90%.**

Different V1 and V2 thresholds, together with different decision logics, were then tested on the development set only.

The selected operating point was:

**AND Logic**

* V1 threshold = **0.05**
* V2 threshold = **0.20**
* Development-set Recall = **90.33%**
* Development-set False Alarm Reduction = **78.33%**

This operating point was then tested once on the previously frozen test set.

### Final Frozen Test Results

| Metric                         |          Result |
| ------------------------------ | --------------: |
| Recall                         |      **90.75%** |
| Precision                      |      **72.24%** |
| False Alarm Reduction          |      **76.75%** |
| Reviews Avoided                | **995 / 2,000** |
| Reduction in Human Review Load |       **49.8%** |

The final test showed that the selected operating point retained more than 90% of genuine alerts while removing a large portion of unnecessary alerts.

### Business Interpretation

In the test set, the original system generated **2,000 alerts**.

After applying the two-stage verification rule:

* **726** genuine alerts were correctly detected.
* **279** false alarms remained.
* **995 cases** did not need to be sent for human review.

This means the system could reduce the number of cases requiring human attention by approximately **49.8%**, while maintaining **90.75% recall** on this benchmark.

### Key Finding

Phase 2A demonstrated that the system can be configured for a more realistic business trade-off:

> **Reduce unnecessary alerts while keeping the ability to detect most genuine events.**

This operating point was used as the basis for the next phase: **Business Application and Dashboard Development**.

---

# Phase 2 — Business Application & Dashboard Implementation

## Objective

Phase 2 transformed the validated two-stage verification pipeline into a reusable business-oriented decision-support application.

The objective was not to develop a new model, but to demonstrate how the validated operating point could be integrated into practical visual-inspection workflows while keeping the AI decision logic, benchmark findings, and human review process clearly separated.

---

## Phase 2A Summary — Operating Point Validation

The operating point selected during Phase 2A was frozen for the application:

* **Decision Logic:** AND
* **Stage 1 — OWL-ViT:** V1 ≥ 0.05
* **Stage 2 — CLIP:** V2 ≥ 0.20

Frozen-test benchmark results:

| Metric                             |                  Result |
| ---------------------------------- | ----------------------: |
| Positive samples detected (Recall) |              **90.75%** |
| False Alarm Reduction              |              **76.75%** |
| Precision                          |              **72.24%** |
| Cases filtered before human review | **995 / 2,000 (49.8%)** |

These figures represent empirical results on the frozen 2,000-sample COCO-based benchmark and are not claims of production performance.

---

# Phase 2B-1 — Business Application Specification

A common workflow was designed for two application domains using the same underlying verification engine.

### Application 1 — Workplace Safety Monitoring

Example use cases:

* PPE / helmet compliance
* workplace hazard monitoring
* forklift or pedestrian-related scenarios
* walkway or aisle obstruction

### Application 2 — Quality & Assembly Inspection

Example use cases:

* missing fasteners
* assembly verification
* connector / component anomalies
* missing covers or other visible assembly issues

The important design principle is that both applications use the same:

**OWL-ViT → ROI Crop → CLIP → Frozen AND Gate**

pipeline.

This demonstrates reuse of the same Data Science system across different operational contexts rather than creating separate models for each domain.

---

# Phase 2B-2 — Gradio Dashboard Implementation

A functional Gradio dashboard was implemented with a shared `TwoStageVerificationEngine`.

### Core Engine

The engine performs the following sequence:

```text
Input Image + Target Prompt
          ↓
Stage 1: OWL-ViT Candidate Detection
          ↓
Highest-Scoring Candidate / Bounding Box
          ↓
10% Context-Padded ROI Crop
          ↓
Stage 2: CLIP Visual-Semantic Verification
          ↓
V1 ≥ 0.05 AND V2 ≥ 0.20
          ↓
Alert Confirmed / Not Confirmed (Filtered)
```

The operating parameters are defined centrally in `engine.py` so that the dashboard uses the same frozen thresholds validated during Phase 2A.

---

## Visual Evidence

The dashboard provides two visual outputs:

1. **Annotated original image**

   * Stage 1 bounding box
   * final decision
   * V1 and V2 scores

2. **Stage 2 verification crop**

   * ROI actually evaluated by CLIP
   * includes 10% spatial context padding

A resolution warning is also displayed when the detected ROI is smaller than 100 pixels in either dimension.

---

## Transparent Decision Evidence

The dashboard exposes the evidence behind the binary AI decision:

* Stage 1 V1 score
* V1 threshold
* Stage 1 PASS / FAIL
* Stage 2 V2 score
* V2 threshold
* Stage 2 PASS / FAIL
* AND decision rule
* ROI dimensions
* Stage latency

This makes the system more interpretable than a simple binary prediction interface.

---

## Human-in-the-Loop Workflow

The AI output is intentionally separated from the human operational decision.

### AI layer

* **Alert Confirmed**
* **Not Confirmed (Filtered)**

### Human layer

* **Acknowledge Alert**
* **Dismiss as False Alarm**
* **Flag for Review**

The human actions are presented as simulated operational workflow actions rather than additional AI predictions.

---

## Demo Applications

Four demonstration images were prepared:

```text
app/samples/
├── safety_violation_no_helmet.jpg
├── safety_compliant_with_helmet.jpg
├── quality_defect_missing_bolt.jpg
└── quality_compliant_assembly.jpg
```

These images are used to demonstrate the application workflow and UI behavior.

They are **not part of the frozen benchmark evaluation** and are not used to claim industrial model performance.

---

## Documentation Corrections

The business specification and dashboard wording were corrected to avoid overclaiming.

In particular:

* `Alert Confirmed` is described as satisfying both verification conditions, not as a probability of a genuine event.
* `Not Confirmed` is described as filtered from the primary triage queue rather than being considered automatically safe.
* `49.8%` is described as benchmark cases filtered before human review, rather than real-world labor savings.
* `72.24%` precision is explicitly defined as the proportion of confirmed alerts that are genuine.
* Safety and Quality applications are presented as demonstrations of domain reuse rather than validated industrial applications.

---

# Phase 2 Final Status

### Completed

* [x] Business application specification
* [x] Frozen operating point integration
* [x] Shared two-stage verification engine
* [x] Safety Monitoring application
* [x] Quality & Assembly application
* [x] Visual evidence display
* [x] Score and threshold transparency
* [x] Binary AI decision workflow
* [x] Simulated human-in-the-loop workflow
* [x] Demonstration sample images
* [x] Benchmark KPI presentation
* [x] Business/documentation wording audit
* [x] Application build and runtime testing

## Final Architecture

```text
                    ┌──────────────────────┐
                    │   Input Image        │
                    │   + Target Prompt    │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Stage 1: OWL-ViT     │
                    │ Candidate Detection  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Best Candidate ROI   │
                    │ + 10% Context        │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Stage 2: CLIP        │
                    │ Semantic Verification│
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Frozen AND Gate      │
                    │ V1 ≥ 0.05            │
                    │ V2 ≥ 0.20            │
                    └──────────┬───────────┘
                               ↓
                  ┌────────────┴────────────┐
                  ↓                         ↓
          Alert Confirmed          Not Confirmed
                  ↓                     (Filtered)
          Human Triage
                  ↓
       Acknowledge / Dismiss /
          Flag for Review
```

## Phase 2 Conclusion

Phase 2 successfully converted the validated experimental pipeline into a functional, reusable decision-support dashboard.

The project now contains three connected layers:

**Data Science Evaluation → Business Decision Logic → Interactive Application**

The core algorithm and operating point are now considered frozen. Further work should focus on **Phase 3: portfolio presentation, dashboard polish, documentation, architecture visualization, screenshots, and GitHub cleanup**, rather than adding new models or expanding the system scope.
