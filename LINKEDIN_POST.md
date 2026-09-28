# LinkedIn Post — Two-Stage Visual Verification

```text
After 10+ years in manufacturing, I kept seeing the same problem: automated detection systems flood operators with false alarms until they stop paying attention.

During my studies, I worked on AI hallucination detection — basically asking "did the model actually see what it claims to see?" Turns out that's the exact same problem factory floors deal with every day.

So I built a system around it:

OWL-ViT scans the image for target objects first (Stage 1).
CLIP double-checks the cropped region against the original query (Stage 2).
Alerts only reach human reviewers when both stages agree.

Tested on 2,000 held-out samples:
• Cut false alarms by 76.75%
• Kept 90.75% recall — no trading away safety
• Nearly half the review queue filtered out before anyone has to look

I also ran it through 10 real factory scenarios — forklifts, cranes, PPE checks, defective parts. Found clear limits: small distant objects get missed, metal reflections confuse the model, and if Stage 1 drops a candidate, Stage 2 can't save it.

Knowing where your system breaks matters as much as knowing where it works.

Try it: https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification
Code: https://github.com/Bright579523/two-stage-visual-verification

#ComputerVision #DataScience #Manufacturing #IndustrialAI #Python #HuggingFace
```
