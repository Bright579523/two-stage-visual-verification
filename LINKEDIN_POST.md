# LinkedIn Post — Hallucination Detection in Zero-Shot Vision Models

```text
After 10+ years in manufacturing, I kept seeing the same problem: AI detection systems generate so many false alarms that operators stop trusting them.

During my studies, I decided to investigate why. Specifically — can zero-shot vision-language models (OWL-ViT, CLIP) reliably detect safety violations without any fine-tuning?

I built a two-stage verification pipeline to test this:

→ Stage 1: OWL-ViT scans for target objects using text prompts
→ Stage 2: CLIP re-evaluates each cropped detection region
→ Both stages must agree before an alert fires

Then I stress-tested it on real factory scenarios — PPE checks, forklifts, cranes, surface defects.

What I found:

• The system still generates false positives — CLIP can't reliably understand "without helmet" because it struggles with negation
• Small or distant objects get missed entirely — if Stage 1 doesn't detect it, Stage 2 can't fix it
• Adjusting thresholds to reduce false alarms increases missed detections, and vice versa

The honest conclusion: zero-shot VLMs are not ready for autonomous safety alerting. They hallucinate too often in safety-critical contexts.

But that finding itself has value. Understanding where AI breaks is just as important as showcasing where it works. I built a dashboard that visualizes exactly how and why each detection fails — making the hallucination problem visible and measurable.

For production use, you'd need fine-tuned models (YOLO + domain-specific classifier), not zero-shot inference.

Try the demo: https://huggingface.co/spaces/Bright87/Two_Stage_Visual_Verification
Code: https://github.com/Bright579523/two-stage-visual-verification

#ComputerVision #DataScience #Manufacturing #AIResearch #HallucinationDetection #Python
```
