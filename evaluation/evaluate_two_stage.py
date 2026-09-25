"""
evaluate_two_stage.py — Phase 1: Two-Stage Visual Verification Pipeline Evaluation

Business Goal:
Determine whether a second-stage visual verification layer (V2: CLIP) reduces false alarms
compared to the raw detector (Stage 1) and confidence thresholding (Stage 1 + V1),
while retaining acceptable true alert recall.

Configurations evaluated on frozen unseen test set:
- Config A: Stage 1 (Raw OWL-ViT Top-1 candidate, operational baseline)
- Config B: Stage 1 + V1 (OWL-ViT confidence threshold)
- Config C: Stage 1 + V2 (CLIP semantic similarity threshold)
- Config D: Stage 1 + V1 + V2 (Sequential AND: V1 PASS AND V2 PASS)
- Config E: Stage 1 + V1 + V2 (Relaxed OR: V1 PASS OR V2 PASS)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score
)


def load_and_prepare_data(file_path):
    """Load results CSV and map ground truth to operational business definitions."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)
    print(f"[Data] Loaded {len(df)} records from: {file_path}")

    # Operational mapping:
    # Original true_label: 0 = genuine object present, 1 = target absent (hallucination)
    # Business Alert definition:
    # 1 = Genuine Target Present (Ground truth positive for an alert)
    # 0 = Target Absent (Ground truth negative for an alert; alerting here is a False Alarm)
    if 'is_genuine' not in df.columns:
        if 'true_label' in df.columns:
            df['is_genuine'] = (df['true_label'] == 0).astype(int)
        else:
            raise ValueError("CSV must contain 'true_label' or 'is_genuine' column.")

    required_cols = ['pred_score', 'clip_similarity', 'is_genuine']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def optimize_threshold_f1(scores, labels, score_range=(0.01, 0.99), step=0.01):
    """Optimize threshold strictly on development/train split by maximizing F1."""
    thresholds = np.arange(score_range[0], score_range[1] + step, step)
    best_thresh = thresholds[0]
    best_f1 = -1.0

    for t in thresholds:
        preds = (scores >= t).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t

    return round(float(best_thresh), 4), round(float(best_f1), 4)


def evaluate_configuration(name, y_true, y_pred, baseline_fa=None, baseline_recall=None, baseline_alerts=None):
    """Calculate all model and operational metrics for a specific configuration."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm:
    # [[TN (Correct Rejection), FP (False Alarm)],
    #  [FN (Missed Alert),      TP (True Alert)]]
    tn, fp, fn, tp = cm.ravel()

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    total_alerts = int(tp + fp)
    fa = int(fp)
    true_alerts = int(tp)

    # Operational comparison vs Stage 1 Baseline
    if baseline_fa is not None and baseline_fa > 0:
        fa_reduction_pct = ((baseline_fa - fa) / baseline_fa) * 100.0
    else:
        fa_reduction_pct = 0.0

    if baseline_recall is not None and baseline_recall > 0:
        recall_retention_pct = (rec / baseline_recall) * 100.0
    else:
        recall_retention_pct = 100.0 if rec > 0 else 0.0

    if baseline_alerts is not None:
        reviews_avoided = max(0, baseline_alerts - total_alerts)
    else:
        reviews_avoided = 0

    return {
        "Configuration": name,
        "Total Alerts": total_alerts,
        "True Alerts (TP)": true_alerts,
        "False Alarms (FP)": fa,
        "Missed Alerts (FN)": int(fn),
        "Correct Rejections (TN)": int(tn),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "FPR": round(fpr, 4),
        "FNR": round(fnr, 4),
        "FA Reduction (%)": round(fa_reduction_pct, 2),
        "Recall Retention (%)": round(recall_retention_pct, 2),
        "Reviews Avoided": reviews_avoided,
        "_cm": cm
    }


def plot_confusion_matrices(results, output_path):
    """Plot confusion matrices for all evaluated configurations."""
    n_configs = len(results)
    fig, axes = plt.subplots(1, n_configs, figsize=(5 * n_configs, 4.5))
    if n_configs == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = res["_cm"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Reject (0)", "Alert (1)"],
            yticklabels=["Absent (0)", "Genuine (1)"]
        )
        ax.set_title(f"{res['Configuration']}\nF1={res['F1-Score']:.3f} | Prec={res['Precision']:.3f} | Rec={res['Recall']:.3f}", fontsize=10)
        ax.set_xlabel("System Decision")
        ax.set_ylabel("Ground Truth")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Plot] Confusion matrices saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fair Evaluation of Two-Stage Visual Verification Pipeline")
    parser.add_argument("--data_path", type=str, default="data/v2_clip_results.csv",
                        help="Path to results CSV (with pred_score and clip_similarity)")
    parser.add_argument("--test_size", type=float, default=0.40,
                        help="Fraction of data held out for frozen test set (default: 0.40)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split (default: 42)")
    parser.add_argument("--output_dir", type=str, default="evaluation",
                        help="Output directory for reports and plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    df = load_and_prepare_data(args.data_path)

    # 2. Stratified Train / Test Split
    strat_col = df['negative_type'] if 'negative_type' in df.columns else df['is_genuine']
    
    # Handle tiny mockup dataset gracefully if test_size causes single-element classes
    try:
        train_df, test_df = train_test_split(
            df, test_size=args.test_size, random_state=args.seed, stratify=strat_col
        )
    except Exception:
        print("[Notice] Dataset size too small for full stratification. Falling back to non-stratified split.")
        train_df, test_df = train_test_split(
            df, test_size=args.test_size, random_state=args.seed
        )

    print("\n" + "="*70)
    print("  DATA SPLIT SUMMARY (REPRODUCIBLE SPLIT)")
    print("="*70)
    print(f"  Total Samples     : {len(df)}")
    print(f"  Train / Dev Set   : {len(train_df)} ({len(train_df)/len(df)*100:.1f}%) - used ONLY for threshold tuning")
    print(f"  Frozen Test Set   : {len(test_df)} ({len(test_df)/len(df)*100:.1f}%) - evaluated strictly ONCE")
    print(f"  Test Genuine (P)  : {test_df['is_genuine'].sum()}")
    print(f"  Test Absent  (N)  : {len(test_df) - test_df['is_genuine'].sum()}")

    # 3. Threshold Optimization on Train/Dev Set ONLY
    print("\n" + "="*70)
    print("  THRESHOLD OPTIMIZATION (DEVELOPMENT SET ONLY)")
    print("="*70)
    opt_t_v1, train_f1_v1 = optimize_threshold_f1(
        train_df['pred_score'].values, train_df['is_genuine'].values,
        score_range=(0.01, 0.99), step=0.01
    )
    opt_t_v2, train_f1_v2 = optimize_threshold_f1(
        train_df['clip_similarity'].values, train_df['is_genuine'].values,
        score_range=(0.05, 0.50), step=0.005
    )
    print(f"  V1 (Confidence)  : Best Threshold = {opt_t_v1} (Train F1 = {train_f1_v1:.4f})")
    print(f"  V2 (CLIP Sim)    : Best Threshold = {opt_t_v2} (Train F1 = {train_f1_v2:.4f})")

    # 4. Evaluation on Frozen Test Set
    y_test = test_df['is_genuine'].values

    # Config A: Stage 1 Candidate (Raw Baseline - All candidates alerted)
    y_pred_A = np.ones(len(test_df), dtype=int)
    res_A = evaluate_configuration("A. Stage 1 (Raw Baseline)", y_test, y_pred_A)

    baseline_fa = res_A["False Alarms (FP)"]
    baseline_recall = res_A["Recall"]
    baseline_alerts = res_A["Total Alerts"]

    # Re-evaluate A with self as baseline
    res_A = evaluate_configuration(
        "A. Stage 1 (Raw Baseline)", y_test, y_pred_A,
        baseline_fa=baseline_fa, baseline_recall=baseline_recall, baseline_alerts=baseline_alerts
    )

    # Config B: Stage 1 + V1
    y_pred_B = (test_df['pred_score'].values >= opt_t_v1).astype(int)
    res_B = evaluate_configuration(
        f"B. Stage 1 + V1 (t={opt_t_v1})", y_test, y_pred_B,
        baseline_fa=baseline_fa, baseline_recall=baseline_recall, baseline_alerts=baseline_alerts
    )

    # Config C: Stage 1 + V2
    y_pred_C = (test_df['clip_similarity'].values >= opt_t_v2).astype(int)
    res_C = evaluate_configuration(
        f"C. Stage 1 + V2 (t={opt_t_v2})", y_test, y_pred_C,
        baseline_fa=baseline_fa, baseline_recall=baseline_recall, baseline_alerts=baseline_alerts
    )

    # Config D: Stage 1 + V1 + V2 (Sequential AND: V1 PASS and V2 PASS)
    y_pred_D = ((test_df['pred_score'].values >= opt_t_v1) & (test_df['clip_similarity'].values >= opt_t_v2)).astype(int)
    res_D = evaluate_configuration(
        "D. Stage 1 + V1 + V2 (AND)", y_test, y_pred_D,
        baseline_fa=baseline_fa, baseline_recall=baseline_recall, baseline_alerts=baseline_alerts
    )

    # Config E: Stage 1 + V1 + V2 (Relaxed OR: V1 PASS or V2 PASS)
    y_pred_E = ((test_df['pred_score'].values >= opt_t_v1) | (test_df['clip_similarity'].values >= opt_t_v2)).astype(int)
    res_E = evaluate_configuration(
        "E. Stage 1 + V1 + V2 (OR)", y_test, y_pred_E,
        baseline_fa=baseline_fa, baseline_recall=baseline_recall, baseline_alerts=baseline_alerts
    )

    all_results = [res_A, res_B, res_C, res_D, res_E]

    # 5. Format & Display Results
    output_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_cm"} for r in all_results])
    
    print("\n" + "="*95)
    print("  FINAL EVALUATION REPORT ON FROZEN TEST SET")
    print("="*95)
    print(output_df.to_string(index=False))
    print("="*95)

    # 6. Save Artifacts
    csv_out = os.path.join(args.output_dir, "final_comparison_table.csv")
    output_df.to_csv(csv_out, index=False)
    print(f"\n[Save] Metrics table saved to: {csv_out}")

    plot_out = os.path.join(args.output_dir, "confusion_matrices.png")
    plot_confusion_matrices(all_results, plot_out)

    # 7. Summary Interpretation
    print("\n" + "="*70)
    print("  EXECUTIVE SUMMARY & INTERPRETATION")
    print("="*70)
    fa_red_D = res_D["FA Reduction (%)"]
    rec_ret_D = res_D["Recall Retention (%)"]
    fa_red_B = res_B["FA Reduction (%)"]
    rec_ret_B = res_B["Recall Retention (%)"]

    print(f"* Baseline Stage 1 False Alarms : {baseline_fa} out of {len(test_df) - test_df['is_genuine'].sum()} non-target samples")
    print(f"* Stage 1 + V1 Reduction       : {fa_red_B:.1f}% FA reduction (Recall retained: {rec_ret_B:.1f}%)")
    print(f"* Two-Stage (AND) Reduction    : {fa_red_D:.1f}% FA reduction (Recall retained: {rec_ret_D:.1f}%)")
    
    if fa_red_D > fa_red_B:
        additional_drop = fa_red_D - fa_red_B
        print(f"[VERDICT] Stage 2 (CLIP) provides an additional {additional_drop:.1f}% false alarm reduction over V1 alone.")
    elif fa_red_D == fa_red_B:
        print("[VERDICT] Stage 2 (CLIP) achieves the same false alarm reduction as V1 on this test split.")
    else:
        print("[VERDICT] Stage 2 (CLIP) does NOT improve upon V1 alone under the current configuration.")

    print("="*70)


if __name__ == "__main__":
    main()
