"""
optimize_operating_point.py — Phase 2A: Operating Point Optimization

Business Goal:
Find a practical, principled operating point for the Two-Stage Visual Verification system
governed by an explicit business constraint (e.g., Recall >= 90% in Safety / Quality).
Optimization is performed strictly on the Development/Train set (3,000 rows),
and validated strictly ONCE on the Frozen Test set (2,000 rows).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def load_and_split_data(file_path, test_size=0.40, seed=42):
    """Load 5,000-row dataset and perform reproducible stratified split."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    # Business mapping: 1 = Genuine Target (Positive), 0 = Absent / Hallucination (Negative)
    df['is_genuine'] = (df['true_label'] == 0).astype(int)

    strat_col = df['negative_type'] if 'negative_type' in df.columns else df['is_genuine']
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strat_col
    )
    return train_df, test_df


def evaluate_decision_rule(y_true, y_pred, baseline_fa, baseline_rec, baseline_alerts):
    """Compute standard model and operational business metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    total_alerts = int(tp + fp)
    fa = int(fp)
    true_alerts = int(tp)

    fa_red = ((baseline_fa - fa) / baseline_fa) * 100.0 if baseline_fa > 0 else 0.0
    rec_ret = (rec / baseline_rec) * 100.0 if baseline_rec > 0 else 0.0
    reviews_avoided = max(0, baseline_alerts - total_alerts)

    return {
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
        "FA Reduction (%)": round(fa_red, 2),
        "Recall Retention (%)": round(rec_ret, 2),
        "Reviews Avoided": reviews_avoided,
        "_cm": cm
    }


def analyze_dev_operating_points(train_df, recall_target=0.90):
    """
    Explore threshold space across V1, V2, AND, and OR logics strictly on Dev Set.
    Identify the candidate rule that maximizes False Alarm Reduction subject to Recall >= recall_target.
    """
    y_dev = train_df['is_genuine'].values
    v1_scores = train_df['pred_score'].values
    v2_scores = train_df['clip_similarity'].values

    # Dev Baseline (Stage 1 Raw)
    y_pred_base = np.ones(len(y_dev), dtype=int)
    base_eval = evaluate_decision_rule(y_dev, y_pred_base,
                                       baseline_fa=np.sum(y_dev == 0),
                                       baseline_rec=1.0,
                                       baseline_alerts=len(y_dev))
    dev_base_fa = base_eval["False Alarms (FP)"]
    dev_base_rec = base_eval["Recall"]
    dev_base_alerts = base_eval["Total Alerts"]

    candidates = []

    # 1. Sweep V1-only (Confidence Threshold)
    for t1 in np.arange(0.01, 0.40, 0.01):
        pred = (v1_scores >= t1).astype(int)
        metrics = evaluate_decision_rule(y_dev, pred, dev_base_fa, dev_base_rec, dev_base_alerts)
        metrics["Logic"] = "V1 Only"
        metrics["Threshold_V1"] = round(t1, 3)
        metrics["Threshold_V2"] = np.nan
        candidates.append(metrics)

    # 2. Sweep V2-only (CLIP Similarity Threshold)
    for t2 in np.arange(0.05, 0.40, 0.01):
        pred = (v2_scores >= t2).astype(int)
        metrics = evaluate_decision_rule(y_dev, pred, dev_base_fa, dev_base_rec, dev_base_alerts)
        metrics["Logic"] = "V2 Only"
        metrics["Threshold_V1"] = np.nan
        metrics["Threshold_V2"] = round(t2, 3)
        candidates.append(metrics)

    # 3. Sweep Sequential AND Logic (V1 >= t1 AND V2 >= t2)
    for t1 in [0.03, 0.05, 0.07, 0.09, 0.11]:
        for t2 in [0.15, 0.18, 0.20, 0.22, 0.25]:
            pred = ((v1_scores >= t1) & (v2_scores >= t2)).astype(int)
            metrics = evaluate_decision_rule(y_dev, pred, dev_base_fa, dev_base_rec, dev_base_alerts)
            metrics["Logic"] = "AND Logic"
            metrics["Threshold_V1"] = round(t1, 3)
            metrics["Threshold_V2"] = round(t2, 3)
            candidates.append(metrics)

    # 4. Sweep Relaxed OR Logic (V1 >= t1 OR V2 >= t2)
    for t1 in [0.05, 0.07, 0.09, 0.12, 0.15, 0.18]:
        for t2 in [0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
            pred = ((v1_scores >= t1) | (v2_scores >= t2)).astype(int)
            metrics = evaluate_decision_rule(y_dev, pred, dev_base_fa, dev_base_rec, dev_base_alerts)
            metrics["Logic"] = "OR Logic"
            metrics["Threshold_V1"] = round(t1, 3)
            metrics["Threshold_V2"] = round(t2, 3)
            candidates.append(metrics)

    cand_df = pd.DataFrame([{k: v for k, v in c.items() if k != "_cm"} for c in candidates])

    # Filter by constraint: Recall >= recall_target (e.g. >= 0.90)
    valid_cands = cand_df[cand_df["Recall"] >= recall_target].copy()

    # Sort valid candidates by highest False Alarm Reduction
    if len(valid_cands) > 0:
        valid_cands = valid_cands.sort_values(by=["FA Reduction (%)", "Precision"], ascending=[False, False])
        best_cand = valid_cands.iloc[0].to_dict()
    else:
        # Fallback to candidate with closest recall
        cand_df = cand_df.sort_values(by=["Recall", "FA Reduction (%)"], ascending=[False, False])
        best_cand = cand_df.iloc[0].to_dict()

    return cand_df, best_cand, base_eval


def plot_pareto_curve(cand_df, best_cand, output_path):
    """Plot the Trade-off between Recall and False Alarm Reduction."""
    plt.figure(figsize=(10, 6))

    # Scatter plot by Logic
    palette = {"V1 Only": "#2563eb", "V2 Only": "#9333ea", "AND Logic": "#dc2626", "OR Logic": "#16a34a"}
    sns.scatterplot(
        data=cand_df, x="Recall", y="FA Reduction (%)", hue="Logic",
        style="Logic", s=60, alpha=0.75, palette=palette
    )

    # Highlight target line (Recall = 90%)
    plt.axvline(x=0.90, color="gray", linestyle="--", alpha=0.7, label="Business Constraint: Recall >= 90%")

    # Highlight Selected Best Operating Point
    plt.scatter(
        [best_cand["Recall"]], [best_cand["FA Reduction (%)"]],
        color="#d97706", s=180, zorder=10, edgecolor="black", linewidth=1.5,
        label=f"Selected: {best_cand['Logic']} (Rec={best_cand['Recall']:.2f}, FA Red={best_cand['FA Reduction (%)']:.1f}%)"
    )

    plt.title("Operating Point Trade-off on Development Set (Recall vs. False Alarm Reduction)", fontsize=13, fontweight="bold")
    plt.xlabel("True Alert Recall (Proportion of genuine targets kept)", fontsize=11)
    plt.ylabel("False Alarm Reduction vs. Stage 1 Baseline (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Plot] Trade-off curve saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Find Practical Operating Point for Visual Verification")
    parser.add_argument("--data_path", type=str, default="data/v2_clip_results.csv",
                        help="Path to real 5,000-row CSV")
    parser.add_argument("--recall_target", type=float, default=0.90,
                        help="Minimum recall constraint for genuine targets (default: 0.90)")
    parser.add_argument("--output_dir", type=str, default="evaluation",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data & Reproduce Split
    train_df, test_df = load_and_split_data(args.data_path, test_size=0.40, seed=42)
    print("="*75)
    print("  PHASE 2A: OPERATING POINT OPTIMIZATION UNDER BUSINESS CONSTRAINT")
    print("="*75)
    print(f"  Total Dataset     : {len(train_df) + len(test_df)} rows")
    print(f"  Dev Set (Tune)    : {len(train_df)} rows (40% Genuine, 60% Absent)")
    print(f"  Frozen Test Set   : {len(test_df)} rows (Untouched during exploration)")
    print(f"  Business Goal     : Maximize False Alarm Reduction subject to Recall >= {args.recall_target * 100:.0f}%")

    # 2. Explore Dev Set
    cand_df, best_cand, dev_base = analyze_dev_operating_points(train_df, recall_target=args.recall_target)

    # 3. Plot Trade-off Curve
    tradeoff_plot = os.path.join(args.output_dir, "operating_point_tradeoff.png")
    plot_pareto_curve(cand_df, best_cand, tradeoff_plot)

    # 4. Dev Set Top Candidates Table
    print("\n" + "="*75)
    print(f"  TOP 5 OPERATING POINTS SATISFYING RECALL >= {args.recall_target * 100:.0f}% (DEV SET)")
    print("="*75)
    top_cands = cand_df[cand_df["Recall"] >= args.recall_target].sort_values(
        by=["FA Reduction (%)", "Precision"], ascending=[False, False]
    ).head(5)
    print(top_cands[["Logic", "Threshold_V1", "Threshold_V2", "Precision", "Recall", "F1-Score", "FA Reduction (%)", "Reviews Avoided"]].to_string(index=False))

    print("\n" + "-"*75)
    print("  SELECTED CANDIDATE RULE (FROM DEV SET):")
    print(f"  Logic        : {best_cand['Logic']}")
    print(f"  Threshold V1 : {best_cand['Threshold_V1']}")
    print(f"  Threshold V2 : {best_cand['Threshold_V2']}")
    print(f"  Dev Metrics  : Recall = {best_cand['Recall']:.4f} | FA Reduction = {best_cand['FA Reduction (%)']:.2f}% | Precision = {best_cand['Precision']:.4f}")
    print("-"*75)

    # 5. Single Final Validation on Frozen Test Set
    print("\n" + "="*75)
    print("  SINGLE FINAL VALIDATION ON FROZEN TEST SET (N = 2,000)")
    print("="*75)
    y_test = test_df['is_genuine'].values
    test_v1 = test_df['pred_score'].values
    test_v2 = test_df['clip_similarity'].values

    # Test Baseline (Stage 1 Raw)
    y_test_base = np.ones(len(y_test), dtype=int)
    test_base_metrics = evaluate_decision_rule(
        y_test, y_test_base,
        baseline_fa=np.sum(y_test == 0),
        baseline_rec=1.0,
        baseline_alerts=len(y_test)
    )
    test_base_fa = test_base_metrics["False Alarms (FP)"]
    test_base_rec = test_base_metrics["Recall"]
    test_base_alerts = test_base_metrics["Total Alerts"]

    test_base_metrics["Configuration"] = "A. Stage 1 (Raw Baseline)"

    # Execute Selected Rule on Test Set
    t1 = best_cand['Threshold_V1']
    t2 = best_cand['Threshold_V2']

    if best_cand['Logic'] == "V1 Only":
        y_test_sel = (test_v1 >= t1).astype(int)
        rule_name = f"Recommended: V1 Only (t={t1})"
    elif best_cand['Logic'] == "V2 Only":
        y_test_sel = (test_v2 >= t2).astype(int)
        rule_name = f"Recommended: V2 Only (t={t2})"
    elif best_cand['Logic'] == "AND Logic":
        y_test_sel = ((test_v1 >= t1) & (test_v2 >= t2)).astype(int)
        rule_name = f"Recommended: AND Logic (t1={t1}, t2={t2})"
    else:  # OR Logic
        y_test_sel = ((test_v1 >= t1) | (test_v2 >= t2)).astype(int)
        rule_name = f"Recommended: OR Logic (t1={t1}, t2={t2})"

    sel_metrics = evaluate_decision_rule(
        y_test, y_test_sel,
        baseline_fa=test_base_fa,
        baseline_rec=test_base_rec,
        baseline_alerts=test_base_alerts
    )
    sel_metrics["Configuration"] = rule_name

    # Compare with Previous Phase 1 Default AND rule for context
    y_test_p1_and = ((test_v1 >= 0.09) & (test_v2 >= 0.25)).astype(int)
    p1_and_metrics = evaluate_decision_rule(
        y_test, y_test_p1_and,
        baseline_fa=test_base_fa,
        baseline_rec=test_base_rec,
        baseline_alerts=test_base_alerts
    )
    p1_and_metrics["Configuration"] = "Ref: Phase 1 Strict AND (t1=0.09, t2=0.25)"

    comparison_results = [test_base_metrics, p1_and_metrics, sel_metrics]
    final_test_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_cm"} for r in comparison_results])

    print(final_test_df[["Configuration", "Total Alerts", "True Alerts (TP)", "False Alarms (FP)", "Precision", "Recall", "FA Reduction (%)", "Recall Retention (%)", "Reviews Avoided"]].to_string(index=False))

    # Save validation table
    csv_out = os.path.join(args.output_dir, "operating_point_validation.csv")
    final_test_df.to_csv(csv_out, index=False)
    print(f"\n[Save] Validation table saved to: {csv_out}")

    # Plot final confusion matrix comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cm_base = test_base_metrics["_cm"]
    cm_sel = sel_metrics["_cm"]

    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Reject (0)", "Alert (1)"], yticklabels=["Absent (0)", "Genuine (1)"])
    axes[0].set_title(f"Baseline: Stage 1 Raw\nFA={test_base_metrics['False Alarms (FP)']} | Recall={test_base_metrics['Recall']:.1%}")
    axes[0].set_xlabel("System Decision")
    axes[0].set_ylabel("Ground Truth")

    sns.heatmap(cm_sel, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=["Reject (0)", "Alert (1)"], yticklabels=["Absent (0)", "Genuine (1)"])
    axes[1].set_title(f"{rule_name}\nFA Reduction={sel_metrics['FA Reduction (%)']:.1f}% | Recall={sel_metrics['Recall']:.1%}")
    axes[1].set_xlabel("System Decision")
    axes[1].set_ylabel("Ground Truth")

    plt.tight_layout()
    cm_plot_out = os.path.join(args.output_dir, "operating_point_cm.png")
    plt.savefig(cm_plot_out, dpi=200)
    plt.close()
    print(f"[Plot] Confusion matrix saved to: {cm_plot_out}")

    print("\n" + "="*75)
    print("  EXECUTIVE SUMMARY (PLAIN LANGUAGE)")
    print("="*75)
    print(f"* Business Target: Keep at least {args.recall_target * 100:.0f}% of genuine alerts to protect safety/quality.")
    print(f"* Selected Rule  : {rule_name}")
    print(f"* Test Results   : False alarms dropped by {sel_metrics['FA Reduction (%)']:.1f}% (from {test_base_fa} down to {sel_metrics['False Alarms (FP)']}).")
    print(f"                   Genuine alert recall is maintained at {sel_metrics['Recall']:.1%} ({sel_metrics['True Alerts (TP)']} out of 800 targets alerted).")
    print(f"                   Avoided {sel_metrics['Reviews Avoided']} unnecessary manual inspections ({sel_metrics['Reviews Avoided']/test_base_alerts*100:.1f}% workload reduction).")
    print("="*75)


if __name__ == "__main__":
    main()
