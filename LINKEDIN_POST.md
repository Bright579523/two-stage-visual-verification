# LinkedIn Article & Post Strategy: Two-Stage Visual Verification Platform

This document contains publication-ready LinkedIn copy and strategic framing for **Project 2 (Two-Stage Visual Verification)**. It connects academic research on AI hallucination verification with practical manufacturing operational experience (10+ years), positioning you as an applied Data Science / Computer Vision professional who designs evaluated, deployable solutions.

---

## Strategic Framing & Narrative Backbone

```text
University Research (Prompt Hallucination Detection)
         │
         ▼
Engineering Problem (How do we verify zero-shot vision outputs?)
         │
         ▼
Manufacturing Reality (Excessive false alarms swamp human triage teams)
         │
         ▼
Two-Stage Architecture (Detect broadly first, verify selectively second)
         │
         ▼
Evaluation-Driven Selection (Recall ≥ 90% constraint on 5,000 samples)
         │
         ▼
Interactive Deployment (Hugging Face Spaces + Multi-Domain Reuse)
         │
         ▼
Engineering Maturity (Transparent documentation of failure modes & limitations)
```

### Core Positioning Statement
> **"Coming from a manufacturing background, I was particularly interested in whether a computer vision model could be evaluated not only by model metrics, but also by the amount of unnecessary work it creates for human operators."**

---

## Option 1: Ready-to-Publish LinkedIn Post (Recommended for Main Feed)

*Copy and paste directly into the LinkedIn post creator. It is structured for readability, high engagement, and immediate credibility.*

```text
Can we evaluate computer vision models not just by benchmark accuracy, but by how much unnecessary work they create for human operators?

Coming from a 10+ year background in manufacturing, I’ve seen firsthand what happens when automated monitoring produces too many false alarms: operators experience alert fatigue, review queues back up, and teams either raise the threshold (risking missed hazards) or stop trusting the system altogether.

This project started from a university project on AI prompt hallucination detection. I wanted to take the core verification idea further: instead of only studying when a vision model produces false positives, can we turn that verification step into a practical decision-support workflow?

I built a two-stage computer vision pipeline:
1. Stage 1 (OWL-ViT): Proposes potential candidate regions at high sensitivity (V1 ≥ 0.05).
2. Stage 2 (CLIP): Examines the detected candidate on a 10% context-padded crop to verify whether the image semantics genuinely match the prompt (V2 ≥ 0.20).
3. AND Gate: Alerts escalate to the human review queue only when both gates pass.

To avoid tuning on test data, I evaluated the system across a 5,000-sample COCO-based benchmark, splitting it into a 60% development set and a 40% frozen test set (2,000 samples). Rather than blindly maximizing precision, I enforced a strict operational constraint: Recall must stay at or above 90.0%.

On the frozen test set, the final operating point achieved:
• 90.75% Recall (safety-critical recall retained)
• 76.75% False Alarm Reduction
• 72.24% Precision (up from 40.0% raw baseline)
• 49.8% of benchmark cases filtered before the primary human review stage (995 / 2,000)

To test domain reuse, I packaged the shared engine into a Gradio web application deployed on Hugging Face Spaces and tested it across two distinct industrial contexts:
• Workplace Safety Monitoring (PPE compliance, forklift & crane operations)
• Manufacturing Quality Inspection (missing fasteners, burr/scrap defects)

I also conducted a 10-scenario industrial stress test, which deliberately highlighted system limitations:
• Small or distant objects (<1,000 px² ROI) can reduce Stage 1 confidence below threshold.
• Stage 2 cannot rescue an alert when Stage 1 fails to propose a candidate—a structural property of sequential architectures.
• High visual clutter and specular highlights on reflective metal parts can suppress open-vocabulary embeddings.

Understanding these operational boundaries is just as important as knowing what the model gets right.

Live Demo on Hugging Face Spaces: https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification
Code & Full Benchmark Report on GitHub: https://github.com/Bright579523/hallucination-detection

#ComputerVision #DataScience #MachineLearning #IndustrialAI #Manufacturing #OpenVocabulary #MLOps #Python #HuggingFace
```

---

## Option 2: Long-Form LinkedIn Article / Pulse Newsletter

*Use this version if you wish to publish an in-depth LinkedIn Article to showcase deep technical thought leadership.*

```markdown
# Bridging Academic AI and Factory Operations: A Two-Stage Verification Approach to False Alarm Reduction

### By Watcharapon Anaketanaset

For over a decade, I worked in industrial manufacturing. One of the recurring challenges in any automation effort—whether mechanical, sensor-based, or visual—is false alarms. 

In theory, an automated detection system flags safety breaches or quality defects instantly. In reality, if a system triggers on shadows, perspective anomalies, or benign variations, human operators quickly suffer from **alert fatigue**. When that happens, teams face an uncomfortable dilemma:
1. **Lower the alert volume by raising the threshold:** You reduce false alarms, but you risk missing genuine accidents or critical component defects.
2. **Leave sensitivity high:** Operators spend hours triaging trivial noise, and genuine alerts get lost in the backlog.

When I started working on a university research project on **AI prompt hallucination detection**, I realized the underlying question was directly applicable to factory operations: 

> *When a zero-shot vision model responds to a prompt, how do we verify whether its output is a genuine detection or a visual false positive?*

I decided to take that core verification concept out of the classroom and turn it into a practical decision-support architecture.

---

### The Architecture: Detect Broadly First, Verify Selectively Second

Rather than trying to force a single model to both localize and verify with high precision, I built a two-stage sequential pipeline:

```
[Input Image + Query]
         │
         ▼
[Stage 1: OWL-ViT Candidate Proposal] ──► Localization (V1 ≥ 0.05)
         │
         ├─── If V1 < 0.05 ───► Filtered (No Operator Triage Needed)
         │
         ▼ (Passes Gate 1)
[10% Context-Padded Crop]
         │
         ▼
[Stage 2: CLIP Semantic Verification] ──► Fine Verification (V2 ≥ 0.20)
         │
         ├─── If V2 < 0.20 ───► Filtered (No Operator Triage Needed)
         │
         ▼ (Passes Gate 2)
[ALERT CONFIRMED (AND Gate)]          ──► Escalated to Human Operator Queue
```

* **Stage 1 (OWL-ViT):** Functions as a sensitive wide-angle candidate proposer. By keeping its threshold low ($V_1 \ge 0.05$), we avoid discarding potential hazards at the entrance of the pipeline.
* **10% Context Padding:** When cropping the candidate bounding box, an adaptive 10% context border is added. This ensures CLIP receives surrounding spatial context rather than an aggressively cropped, truncated shape.
* **Stage 2 (CLIP):** Evaluates the cropped visual-semantic alignment against the query prompt ($V_2 \ge 0.20$).
* **Decision Gate:** Only alerts satisfying **both** criteria (`AND` logic) are escalated to human triage queues.

---

### Evaluation Under an Operational Constraint

In applied Data Science, optimizing purely for F1-score or overall accuracy is rarely appropriate for safety-critical environments. If a system reaches 95% overall accuracy but misses 30% of workers without hardhats, it cannot be deployed.

I established an explicit operational rule:
> **Recall retention must remain at or above 90.0%.**

Using a 5,000-sample COCO-based benchmark (split into a 60% Development Set and a 40% Frozen Test Set), I evaluated five verification strategies:

| Strategy | Recall | False Alarm Reduction | Benchmark Precision | Reviews Avoided |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Baseline (Stage 1 Alone)** | 100.0% | 0.0% | 40.00% | 0 / 2,000 (0.0%) |
| **Strict AND ($t_1=0.09, t_2=0.25$)** | 70.37% | 93.92% | 88.52% | 1,364 / 2,000 (68.2%) |
| **OR Logic ($t_1=0.09, t_2=0.25$)** | 91.37% | 73.33% | 69.55% | 949 / 2,000 (47.5%) |
| **Selected Operating Point: AND ($t_1=0.05, t_2=0.20$)** | **90.75%** | **76.75%** | **72.24%** | **995 / 2,000 (49.8%)** |

While a strict AND rule eliminated 93.9% of false alarms, recall dropped to 70.37%—unacceptable for safety. By calibrating the operating point on the development set to $V_1 \ge 0.05$ and $V_2 \ge 0.20$, the pipeline:
* Retained **90.75% positive recall** on the untouched frozen test set
* Cut false alarms by **76.75%**
* Filtered **49.8% of incoming cases (995 out of 2,000)** before human operator triage
* Raised precision from 40.0% to **72.24%**

---

### Reusing the Architecture Across Industrial Domains

To prove that the verification logic was domain-agnostic, I built a modular Gradio dashboard sharing a single verification engine across two application areas:

1. **Workplace Safety Monitoring:** Detecting missing hardhats, footwear compliance during forklift dismounts, and personnel near suspended crane loads.
2. **Manufacturing Quality Inspection:** Detecting unseated fasteners, missing bolts, and metal burrs on machined components.

*(Note: These workflows are demonstrations of architecture reuse and are not production-validated inspection systems.)*

---

### Documenting Failure Modes: What Stress Testing Revealed

No computer vision system is perfect, and engineering credibility comes from knowing where a system breaks. I evaluated the deployed pipeline against 10 photorealistic industrial stress scenarios, uncovering three key operational limitations:

1. **Small & Distant Objects:** A factory surveillance view with a distant worker ($22 \times 43$ px ROI) scored $V_1 = 0.0310 < 0.05$, causing a missed alert. Very small or distant targets reduce Stage 1 confidence below the operating threshold.
2. **The Sequential Gate Bottleneck:** In a cluttered production tray with reflective steel parts, scrap debris scored $V_1 = 0.0399 < 0.05$, while Stage 2 scored $V_2 = 0.2491 > 0.20$. Because the pipeline is strictly sequential, Stage 2 cannot rescue an alert if Stage 1 misses the proposal.
3. **Prompt Sensitivity:** Text embeddings showed noticeable sensitivity to phrasing (`"a photo of a worker without a hardhat"` produced $V_1 = 0.117$, whereas colloquial shorthand like `"person not wearing helmet"` dropped to $V_1 = 0.018$). Production systems must enforce canonical prompt templates.

---

### Key Takeaway for Industrial AI

Connecting machine learning models to industrial workflows requires more than high benchmark scores. It requires:
* Framing the problem around operational human constraints (operator workload and recall limits)
* Setting clear trade-off boundaries before touching the test set
* Packaging the pipeline into an accessible, repeatable interface
* Being honest about failure modes and environmental boundaries

I invite you to test the live application and review the complete benchmark code:

* **Live Demo on Hugging Face Spaces:** https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification
* **GitHub Repository:** https://github.com/Bright579523/hallucination-detection
```

---

## Option 3: LinkedIn Profile "Featured Project" Snippet

*Use this snippet under the **Featured** or **Projects** section on your LinkedIn Profile.*

> **Two-Stage Visual Verification Platform (Industrial CV & Decision Support)**  
> *Applied Machine Learning • Open-Vocabulary Computer Vision • Gradio • PyTorch*  
> 
> • Extended a university research initiative on AI prompt hallucination verification into a practical two-stage computer vision decision-support pipeline (OWL-ViT + CLIP) designed to mitigate alert fatigue in industrial visual inspection.  
> • Formulated an evaluation protocol across a 5,000-sample benchmark under an explicit operational constraint (Recall $\ge 90\%$), reducing false alarms by **76.75%** and filtering **49.8% of review cases** while maintaining **90.75% recall** on a held-out test set.  
> • Packaged the modular engine into a multi-domain Gradio dashboard deployed to Hugging Face Spaces, demonstrating workflow reuse across Workplace Safety and Assembly Quality Inspection.  
> • Documented empirical failure modes (small/distant targets, visual clutter, and sequential gate limits) across 10 industrial stress scenarios.  
> 
> **Live Demo:** https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification  
> **Source Code:** https://github.com/Bright579523/hallucination-detection
