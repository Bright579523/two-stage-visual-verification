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

---

# Supplementary Progress Report — Deployment, Stress Testing & Project Freeze

**Date:** 26 September 2026  
**Status:** COMPLETED / FROZEN  
**Project:** Two-Stage Visual Verification for Reducing False Alarms  

## 1. Overview

Following completion of the benchmark evaluation and operating-point optimization, the project was extended through public deployment, industrial stress testing, failure-mode analysis, repeatability testing, and final portfolio packaging.

The project is now considered technically frozen. No further model, architecture, threshold, or feature changes are planned unless future evidence identifies a substantive issue.

The final system uses:
* **Stage 1:** OWL-ViT open-vocabulary object detection
* **Stage 2:** CLIP image-text verification
* **Decision logic:** AND
* **Frozen Stage 1 threshold:** $V_1 \ge 0.05$
* **Frozen Stage 2 threshold:** $V_2 \ge 0.20$
* **ROI context padding:** 10%
* **Candidate selection:** Top-1
* **Deployment:** Gradio application on Hugging Face Spaces

---

## 2. Public Deployment

The application was successfully deployed to a public Hugging Face Space:

* **Hugging Face Space:** [https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification](https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification)
* **Direct application:** [https://bright87-two-stage-visual-verification.hf.space](https://bright87-two-stage-visual-verification.hf.space/)

The deployed application successfully loaded the same two-stage verification workflow used during local testing.

This establishes that the project is not limited to a local prototype and can be packaged as a deployable interactive ML application.

---

## 3. Ten-Scenario Industrial Stress Test

A new set of 10 photorealistic industrial scenarios was created to test visual diversity beyond the original worker/helmet examples.

The test set covered:

**Safety / Industrial**
1. Forklift in warehouse
2. Forklift operator / safety boots close-up
3. Overhead crane lifting a steel component
4. Distant worker in a large manufacturing bay
5. Safety boots in a factory environment
6. Worker without a hardhat

**Manufacturing Quality**
7. Machined component with metal scrap attached
8. Batch inspection with one defective part among multiple normal parts
9. Macro view of burr/scrap on a machined component
10. Production tray containing multiple machined parts with one scrap-contaminated part

The purpose was not to establish industrial production accuracy, but to examine whether the architecture behaves sensibly across different industrial visual contexts.

---

## 4. Stress-Test Results

| # | Scenario | $V_1$ | $V_2$ | Decision |
| :-: | :--- | :---: | :---: | :--- |
| 1 | Forklift warehouse | 0.5799 | 0.3003 | Alert Confirmed |
| 2 | Safety boots close-up | 0.2511 | 0.3255 | Alert Confirmed |
| 3 | Overhead crane | 0.2753 | 0.3344 | Alert Confirmed |
| 4 | Distant worker | 0.0310 | 0.2453 | Not Confirmed |
| 5 | Safety boots low-angle | 0.2890 | 0.3141 | Alert Confirmed |
| 6 | Worker without hardhat | 0.0647 | 0.3264 | Alert Confirmed |
| 7 | Machined part with scrap | 0.1026 | 0.2111 | Alert Confirmed |
| 8 | Batch inspection / one defect | 0.1153 | 0.2216 | Alert Confirmed |
| 9 | Macro burr / scrap | 0.1389 | 0.2840 | Alert Confirmed |
| 10 | Production tray clutter | 0.0399 | 0.2491 | Not Confirmed |

The test demonstrated successful verification across multiple industrial concepts while also exposing meaningful failure modes.

---

## 5. Observed Failure Modes

### 5.1 Small / Distant Objects
Scenario #4 contained a worker at substantial distance from the camera.

The detected ROI was approximately:
$$22 \times 43 \text{ px}$$

Stage 1 produced:
$$V_1 = 0.0310 < 0.05$$
while Stage 2 produced:
$$V_2 = 0.2453 > 0.20$$

The final result was therefore:
**Not Confirmed (Filtered)**

This provides empirical evidence that very small or distant targets can reduce Stage 1 detection confidence.

The project does not claim a specific pixel cutoff or attribute the failure to a single architectural mechanism.

**Recommended documentation wording:**
> *Small or distant objects can reduce Stage 1 detection confidence, increasing the risk of missed alerts.*

---

## 6. Sequential Gate Bottleneck
Scenario #10 demonstrated an important architectural property.

Results:
* $V_1 = 0.0399 \rightarrow$ **FAIL**
* $V_2 = 0.2491 \rightarrow$ **PASS**
* Final decision $\rightarrow$ **Not Confirmed**

Because the system is sequential, Stage 2 cannot rescue an alert when Stage 1 fails to produce a candidate above the initial detection threshold.

This is documented as a structural limitation of the two-stage architecture rather than as a model bug.

**Recommended wording:**
> *Stage 2 cannot rescue an alert when Stage 1 fails to propose a candidate passing the initial threshold. Defects that fail the localization gate are therefore filtered regardless of their Stage 2 semantic alignment.*

---

## 7. Manufacturing Quality Stress Testing
The additional quality scenarios expanded the project beyond safety monitoring.

The system successfully verified:
* individual machined components containing scrap
* a batch containing one abnormal component
* burr/scrap defects in a close-up inspection view

However, the cluttered production-tray scenario produced a Stage 1 failure despite Stage 2 alignment.

This demonstrates that the architecture can support manufacturing-quality applications while also showing that production-like visual complexity can create detection failures.

These scenarios are demonstrations of architecture reuse, not validated industrial inspection performance.

---

## 8. Prompt Sensitivity
Previous testing demonstrated that semantically similar prompts can produce materially different scores.

Examples included successful prompts such as:
`a photo of a worker without a hardhat`
and lower-performing variants such as:
`person not wearing helmet`

This behavior is documented as both:

* **Prompting Guideline:** Use descriptive, image-grounded prompts and maintain canonical prompts for reproducible evaluation.
* **Known Limitation:** Because OWL-ViT relies on image-text similarity, semantically equivalent prompts can produce different detection scores and therefore different final decisions.

Threshold optimization was not repeated based on these observations. The canonical operating point remains frozen.

---

## 9. Repeatability and Determinism
A repeated local inference test was conducted using the same image and prompt three times.

**Input:**
* Image: `safety_violation_no_helmet.jpg`
* Prompt: `a photo of a worker without a hardhat`

**Results:**

| Run | $V_1$ | $V_2$ | Decision |
| :-: | :---: | :---: | :--- |
| 1 | 0.117256 | 0.275284 | Alert Confirmed |
| 2 | 0.117256 | 0.275284 | Alert Confirmed |
| 3 | 0.117256 | 0.275284 | Alert Confirmed |

The local inference results were identical across all three runs.
The public GPU deployment produced numerically close scores with the same final decision.

Therefore, the defensible conclusion is:
> *Local inference was deterministic across repeated runs, while public GPU inference produced numerically close scores with identical final decisions.*

The project does not claim bit-exact equivalence across different hardware environments.

---

## 10. Final Benchmark Evidence
The previously frozen benchmark remains unchanged.

**Frozen test set:**
2,000 samples

**Results for the selected operating point:**
* Recall: **90.75%**
* Precision: **72.24%**
* False Alarm Reduction: **76.75%**
* Recall Retention: **90.75%**
* Cases filtered before human-review stage: **995 / 2,000**
* Review workload avoided: **49.8%** of benchmark cases

The benchmark is COCO-based and uses synthetic/hard-negative construction. These figures therefore describe benchmark performance, not validated production performance.

---

## 11. Project Freeze
The following components are now frozen:

* **Model:** OWL-ViT + CLIP
* **Architecture:** Sequential two-stage verification, Top-1 candidate selection, 10% context-padded ROI
* **Decision Rule:** $V_1 \ge 0.05 \land V_2 \ge 0.20$ (AND Gate)
* **Application:** Gradio dashboard (Safety Monitoring, Quality & Assembly Inspection, Human-review simulation)
* **Evidence:** Frozen benchmark, 10-scenario stress test, failure-mode documentation, deployment test, repeatability test

No further optimization is planned for the current portfolio version.

---

## 12. Portfolio Positioning
The project is positioned as an **Applied Data Science / Computer Vision decision-support system**, rather than a novel computer-vision research contribution.

The main strengths demonstrated are:
1. Translating a business problem into a measurable ML workflow
2. Designing a two-stage verification architecture
3. Selecting an operating point using a recall constraint
4. Evaluating false-alarm reduction and recall trade-offs
5. Performing explicit failure-mode analysis
6. Building a reusable inference engine
7. Deploying the system publicly
8. Documenting limitations instead of presenting benchmark results as production validation

The project complements the user's manufacturing background by demonstrating the ability to connect industrial workflows with modern AI/ML techniques.

---

## 13. Final Status

**Project 2 — Two-Stage Visual Verification**  
**Status: DONE / FROZEN**

The project has now completed:
* [x] Model implementation
* [x] Benchmark evaluation
* [x] Operating-point optimization
* [x] Business application implementation
* [x] Industrial stress testing
* [x] Failure-mode analysis
* [x] Public deployment
* [x] Repeatability testing
* [x] README documentation
* [x] Portfolio packaging

Further work should focus on portfolio integration and interview preparation, rather than additional model experimentation.

