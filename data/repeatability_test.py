import re
import time
from gradio_client import Client, handle_file

SPACE_NAME = "Bright87/Two_Stage_Visual_Verification"
SAMPLE_PATH = r"d:\Advance AI Hallucination detect\hf_space\samples\safety_violation_no_helmet.jpg"
PROMPT = "a photo of a worker without a hardhat"

def extract_scores(html):
    scores = re.findall(r">([0-9]+\.[0-9]+)</span>", html)
    decision = "ALERT CONFIRMED" if "ALERT CONFIRMED" in html else ("NOT CONFIRMED" if "NOT CONFIRMED" in html else "UNKNOWN")
    v1 = float(scores[0]) if len(scores) > 0 else None
    v2 = float(scores[1]) if len(scores) > 1 else None
    return v1, v2, decision

def main():
    print(f"[Repeatability Test] Connecting to Hugging Face Space: {SPACE_NAME}...")
    client = Client(SPACE_NAME)
    
    runs = []
    print(f"[Repeatability Test] Running 3 sequential identical requests on {SAMPLE_PATH}...\n")
    
    for i in range(1, 4):
        t0 = time.time()
        result = client.predict(
            image=handle_file(SAMPLE_PATH),
            prompt=PROMPT,
            api_name="/run_verification"
        )
        latency = (time.time() - t0) * 1000
        html = result[2]
        v1, v2, dec = extract_scores(html)
        runs.append({
            "run": i,
            "v1": v1,
            "v2": v2,
            "decision": dec,
            "latency_ms": latency
        })
        print(f"Run #{i}: V1 = {v1}, V2 = {v2}, Decision = {dec} (Latency: {latency:.1f}ms)")
        time.sleep(1)

    print("\n=== REPEATABILITY TEST SUMMARY ===")
    v1_set = set(r["v1"] for r in runs)
    v2_set = set(r["v2"] for r in runs)
    dec_set = set(r["decision"] for r in runs)
    
    is_deterministic = len(v1_set) == 1 and len(v2_set) == 1 and len(dec_set) == 1
    print(f"V1 unique values: {v1_set}")
    print(f"V2 unique values: {v2_set}")
    print(f"Decision unique values: {dec_set}")
    print(f"Deterministic Result: {'PASS (100% IDENTICAL)' if is_deterministic else 'VARIATION DETECTED'}")

if __name__ == "__main__":
    main()
