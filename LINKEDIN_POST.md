# LinkedIn Article: Two-Stage Visual Verification Platform

## Ready-to-Publish LinkedIn Post

```text
Can we evaluate computer vision models not just by benchmark accuracy, but by how much unnecessary work they create for human operators?

Coming from a 10+ year background in manufacturing, I've seen firsthand what happens when automated monitoring produces too many false alarms: operators experience alert fatigue, review queues back up, and teams either raise the threshold (risking missed hazards) or stop trusting the system altogether.

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
Code & Full Benchmark Report on GitHub: https://github.com/Bright579523/two-stage-visual-verification

#ComputerVision #DataScience #MachineLearning #IndustrialAI #Manufacturing #OpenVocabulary #MLOps #Python #HuggingFace
```

---

## LinkedIn Profile "Featured Project" Snippet

> **Two-Stage Visual Verification Platform (Industrial CV & Decision Support)**  
> *Applied Machine Learning • Open-Vocabulary Computer Vision • Gradio • PyTorch*  
> 
> • Extended a university research initiative on AI prompt hallucination verification into a practical two-stage computer vision decision-support pipeline (OWL-ViT + CLIP) designed to mitigate alert fatigue in industrial visual inspection.  
> • Formulated an evaluation protocol across a 5,000-sample benchmark under an explicit operational constraint (Recall ≥ 90%), reducing false alarms by **76.75%** and filtering **49.8% of review cases** while maintaining **90.75% recall** on a held-out test set.  
> • Packaged the modular engine into a multi-domain Gradio dashboard deployed to Hugging Face Spaces, demonstrating workflow reuse across Workplace Safety and Assembly Quality Inspection.  
> • Documented empirical failure modes (small/distant targets, visual clutter, and sequential gate limits) across 10 industrial stress scenarios.  
> 
> **Live Demo:** https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification  
> **Source Code:** https://github.com/Bright579523/two-stage-visual-verification
