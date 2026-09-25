import os
import sys
import json
from PIL import Image

sys.path.insert(0, r"d:\Advance AI Hallucination detect\app")
from engine import TwoStageVerificationEngine

def main():
    engine = TwoStageVerificationEngine.get_instance()
    test_dir = r"d:\Advance AI Hallucination detect\data\stress_test_samples"

    tests = [
        ("safety_01_forklift_warehouse.jpg", "a photo of a forklift", "Safety / Forklift detection in warehouse"),
        ("safety_02_forklift_boots_step.jpg", "a photo of safety boots", "Safety / Safety boots close-up on forklift step"),
        ("safety_03_overhead_crane.jpg", "a photo of an overhead crane", "Safety / Overhead crane lifting steel beam"),
        ("safety_04_crane_worker_distant.jpg", "a photo of a factory worker", "Safety / Distant worker in factory bay (Resolution challenge)"),
        ("safety_05_safety_boots_lowangle.jpg", "a photo of safety boots", "Safety / Low-angle safety boots on workshop floor"),
        ("safety_06_ppe_missing_helmet.jpg", "a photo of a worker without a hardhat", "Safety / PPE breach (Worker without hardhat)"),
        ("quality_01_part_metal_scrap.jpg", "a photo of scrap metal debris", "Quality / Single part with metal scrap burr"),
        ("quality_02_batch_scrap_defect.jpg", "a photo of scrap metal debris", "Quality / Batch inspection (1 defect in 12 parts)"),
        ("quality_03_burr_scrap_macro.jpg", "a photo of a metal burr defect", "Quality / Macro burr on threaded hole"),
        ("quality_04_tray_clutter_defect.jpg", "a photo of scrap metal debris", "Quality / Production tray clutter defect")
    ]

    results = []
    print("=== RUNNING 10 TEST SCENARIOS ===\n")
    for img_name, prompt, desc in tests:
        path = os.path.join(test_dir, img_name)
        img = Image.open(path)
        res = engine.verify(img, prompt)
        v1 = res["v1_score"]
        v2 = res["v2_score"]
        dec = res["decision"]
        dim = res["box_dimensions"]
        lat_tot = res.get("latency_v1_ms", 0) + res.get("latency_v2_ms", 0)
        results.append({
            "file": img_name,
            "description": desc,
            "prompt": prompt,
            "v1_score": round(v1, 4),
            "v1_pass": res["v1_pass"],
            "v2_score": round(v2, 4),
            "v2_pass": res["v2_pass"],
            "decision": dec,
            "box_dimensions": dim,
            "latency_ms": round(lat_tot, 1)
        })
        print(f"[{img_name}] {desc}")
        print(f"   Prompt: \"{prompt}\"")
        print(f"   V1: {v1:.4f} ({'PASS' if res['v1_pass'] else 'FAIL'}) | V2: {v2:.4f} ({'PASS' if res['v2_pass'] else 'FAIL'}) -> {dec} (Box: {dim}, Latency: {lat_tot:.1f}ms)\n")

    out_json = os.path.join(test_dir, "stress_test_10_samples_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")

if __name__ == "__main__":
    main()
