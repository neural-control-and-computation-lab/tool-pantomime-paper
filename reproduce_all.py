#!/usr/bin/env python3
"""
Single-command reproduction of all manuscript statistical results and figures.

Usage:
  python reproduce_all.py               # Recompute everything from raw kinematics + render Figures 1-4
  python reproduce_all.py --use-cache   # Reuse existing intermediate caches in figures/ if present
  python reproduce_all.py --stats-only  # Print all manuscript statistics without rendering figures
  python reproduce_all.py --no-header   # Render Matplotlib data panels without SVG schematic headers
"""

import argparse
import glob
import json
import os
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

import plot_fig1
import plot_fig2
import plot_fig3
import plot_fig4
from model import APRAXIA_CUTOFF, FEATURE_GROUPS, FEATURE_NAMES, evaluate_predictions

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "modelinputs")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")


def _clear_caches():
    for p in glob.glob(os.path.join(FIGURES_DIR, "*.json")):
        os.remove(p)


def _fmt_pathway_table(rows_by_name):
    header = (
        f"  {'Model Pathway':<15} {'r':>7} {'p-value':>10} {'rho':>7} "
        f"{'MAE':>7} {'RMSE':>7} {'AUC':>7} {'Sens':>11} {'Spec':>11}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, m in rows_by_name:
        sens_str = f"{m['sensitivity']:.2f} ({m['tp']}/{m['tp'] + m['fn']})"
        spec_str = f"{m['specificity']:.2f} ({m['tn']}/{m['tn'] + m['fp']})"
        print(
            f"  {label:<15} {m['r']:>7.3f} {m['r_p']:>10.2e} {m['rho']:>7.3f} "
            f"{m['mae']:>7.3f} {m['rmse']:>7.3f} {m['auc']:>7.3f} {sens_str:>11} {spec_str:>11}"
        )


def print_section_1():
    print("\n" + "=" * 88)
    print("1. COHORT & MANUAL PRAXIS CODING SUMMARY (Figure 1, N = 53)")
    print("=" * 88)
    with open(os.path.join(DATA_DIR, "metadata.json"), "r") as f:
        md = json.load(f)
    rows, item_scores = plot_fig1.load_participant_scores()
    n_valid = sum(r["n"] for r in rows.values())
    print(f"  Total recorded trials in metadata.json : {len(md)}")
    print(f"  Content-valid trials (content = 1)     : {n_valid} across {len(rows)} participants")
    print(f"  Clinical apraxia cutoff                : < {APRAXIA_CUTOFF:.2f} (0-3 scale)\n")

    print(f"  {'Group':<25} {'n':>4} {'Total (0-3)':>16} {'Hand Post.':>14} {'Arm Post.':>14} {'Amp./Timing':>14}")
    print("  " + "-" * 89)
    for g_key, g_label in (
        ("control", "Healthy control"),
        ("nonaprax", "Stroke: without apraxia"),
        ("apraxic", "Stroke: with apraxia"),
    ):
        members = [r for r in rows.values() if r["group"] == g_key]
        tot = [m["total"] for m in members]
        hp = [m["hp"] for m in members]
        ap = [m["ap"] for m in members]
        at = [m["at"] for m in members]
        print(
            f"  {g_label:<25} {len(members):>4d} "
            f"{np.mean(tot):>7.2f} ± {np.std(tot, ddof=1):.2f} "
            f"{np.mean(hp):>6.2f} ± {np.std(hp, ddof=1):.2f} "
            f"{np.mean(ap):>6.2f} ± {np.std(ap, ddof=1):.2f} "
            f"{np.mean(at):>6.2f} ± {np.std(at, ddof=1):.2f}"
        )

    aucs = plot_fig1.compute_item_aucs(rows, item_scores)
    top5 = ", ".join(f"{r['name']} ({r['auc']:.2f})" for r in aucs[:5])
    bot3 = ", ".join(f"{r['name']} ({r['auc']:.2f})" for r in aucs[-3:])
    print(f"\n  Item apraxia discriminability (ROC AUC across 43 stroke participants):")
    print(f"    Highest 5 items : {top5}")
    print(f"    Lowest 3 items  : {bot3}")


def print_section_2():
    print("\n" + "=" * 88)
    print("2. TRAINING SET CROSS-VALIDATION (Figure 2, n = 41 LOPO)")
    print("=" * 88)
    res = plot_fig2.compute_or_load_lopo()
    true_m = res["true"]
    m_part = evaluate_predictions(res["direct"], true_m)
    m_item = evaluate_predictions(res["loio"], true_m)
    m_blend = evaluate_predictions(res["blend"], true_m)
    print(
        f"  Cohort: {m_blend['n_participants']} participants "
        f"({m_blend['n_controls']} control, {m_blend['n_stroke']} stroke "
        f"[{m_blend['n_apraxic']} with apraxia, {m_blend['n_stroke'] - m_blend['n_apraxic']} without apraxia])\n"
    )
    _fmt_pathway_table([("Participant", m_part), ("Item", m_item), ("Blended", m_blend)])

    tot_feat = res["mean_mag"] + res["std_mag"]
    shares = 100.0 * tot_feat / tot_feat.sum()
    print("\n  Participant-level weight share by kinematic family (mean + SD |coefficient|):")
    for fam_name, idxs in FEATURE_GROUPS.items():
        print(f"    {fam_name:<18} ({len(idxs):>2d} features): {shares[idxs].sum():5.1f}%")

    top_f = np.argsort(-shares)[:5]
    print("\n  Top 5 kinematic features (training LOPO):")
    for idx in top_f:
        print(f"    - {FEATURE_NAMES[idx]:<30}: {shares[idx]:.2f}%")

    iw = res["item_weights"]
    top_i = sorted(iw, key=lambda k: iw[k], reverse=True)[:5]
    print("\n  Top 5 reliability-weighted items (training LOPO):")
    for k in top_i:
        print(f"    - {plot_fig2.item_names[k]:<20}: {100.0 * iw[k]:.2f}%")
    return res


def print_section_3():
    print("\n" + "=" * 88)
    print("3. HELD-OUT TEST SET EVALUATION (Figure 3, n = 12, trained on n = 41)")
    print("=" * 88)
    paths = plot_fig3.compute_or_load_pathways()
    true_m = paths["true"]
    m_part = evaluate_predictions(paths["part"], true_m)
    m_item = evaluate_predictions(paths["item"], true_m)
    m_blend = evaluate_predictions(paths["blend"], true_m)
    print(
        f"  Cohort: {m_blend['n_participants']} participants "
        f"({m_blend['n_controls']} control, {m_blend['n_stroke']} stroke "
        f"[{m_blend['n_apraxic']} with apraxia, {m_blend['n_stroke'] - m_blend['n_apraxic']} without apraxia])\n"
    )
    _fmt_pathway_table([("Participant", m_part), ("Item", m_item), ("Blended", m_blend)])

    print(f"\n  {'Participant':<12} {'Group':<25} {'True':>7} {'Part.':>8} {'Item':>8} {'Blended':>9} {'Error':>8}")
    print("  " + "-" * 81)
    for pid in sorted(true_m, key=lambda p: true_m[p]):
        if pid.startswith("CL"):
            grp = "Control"
        elif true_m[pid] < APRAXIA_CUTOFF:
            grp = "Stroke (with apraxia)"
        else:
            grp = "Stroke (without apraxia)"
        err = paths["blend"][pid] - true_m[pid]
        print(
            f"  {pid:<12} {grp:<25} {true_m[pid]:>7.3f} "
            f"{paths['part'][pid]:>8.3f} {paths['item'][pid]:>8.3f} "
            f"{paths['blend'][pid]:>9.3f} {err:>+8.3f}"
        )


def print_section_4(res41):
    print("\n" + "=" * 88)
    print("4. FULL-COHORT CROSS-VALIDATION & WEIGHT STABILITY (Figure 4, N = 53 LOPO)")
    print("=" * 88)
    direct53, loio53, blend53, true53, coef53, itemw53 = plot_fig4.compute_or_load()
    m_part = evaluate_predictions(direct53, true53)
    m_item = evaluate_predictions(loio53, true53)
    m_blend = evaluate_predictions(blend53, true53)
    print(
        f"  Cohort: {m_blend['n_participants']} participants "
        f"({m_blend['n_controls']} control, {m_blend['n_stroke']} stroke "
        f"[{m_blend['n_apraxic']} with apraxia, {m_blend['n_stroke'] - m_blend['n_apraxic']} without apraxia])\n"
    )
    _fmt_pathway_table([("Participant", m_part), ("Item", m_item), ("Blended", m_blend)])

    imp41 = plot_fig4._importance(np.concatenate([res41["mean_mag"], res41["std_mag"]]))
    imp53 = plot_fig4._importance(coef53)
    v41 = np.array([imp41[i] for i in range(len(FEATURE_NAMES))])
    v53 = np.array([imp53[i] for i in range(len(FEATURE_NAMES))])
    fr, fp = pearsonr(v41, v53)
    frho, frhop = spearmanr(v41, v53)

    ids = sorted(set(res41["item_weights"]) & set(itemw53))
    iw41 = np.array([res41["item_weights"][i] for i in ids])
    iw53 = np.array([itemw53[i] for i in ids])
    ir, ip = pearsonr(iw41, iw53)
    irho, irhop = spearmanr(iw41, iw53)

    print("\n  Weight stability between training LOPO (n = 41) and full-cohort LOPO (N = 53):")
    print(f"    Feature weights (46 features) : Pearson r = {fr:.3f} (p = {fp:.2e}), Spearman rho = {frho:.3f} (p = {frhop:.2e})")
    print(f"    Item weights    (39 items)    : Pearson r = {ir:.3f} (p = {ip:.2e}), Spearman rho = {irho:.3f} (p = {irhop:.2e})")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all manuscript statistics and figures from raw kinematic recordings."
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse cached intermediate JSON files in figures/ if already present.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation from raw kinematics (default behavior unless --use-cache is set).",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Compute and print all manuscript statistics without rendering SVG/PNG figures.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Render standalone Matplotlib subplot PNGs without SVG schematic headers.",
    )
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if args.no_header:
        os.environ["NO_HEADER"] = "1"

    if args.recompute or not args.use_cache:
        _clear_caches()

    print_section_1()
    res45 = print_section_2()
    print_section_3()
    print_section_4(res45)

    if not args.stats_only:
        print("\n" + "=" * 88)
        print("5. GENERATING MANUSCRIPT FIGURES (Figures 1-4)")
        print("=" * 88)
        plot_fig1.main()
        plot_fig2.main()
        plot_fig3.main()
        plot_fig4.main()

    elapsed = time.time() - t0
    print("\n" + "=" * 88)
    print(f"Completed in {elapsed:.1f} s.")
    if not args.stats_only:
        print("Generated figures in figures/:")
        for fn in ("fig1.png", "fig2.png", "fig3.png", "fig4.png", "figure1.svg", "figure2.svg", "figure3.svg", "figure4.svg"):
            p = os.path.join(FIGURES_DIR, fn)
            if os.path.exists(p):
                sz_mb = os.path.getsize(p) / (1024 * 1024)
                print(f"  - figures/{fn:<14} ({sz_mb:5.2f} MB)")
    print("=" * 88)


if __name__ == "__main__":
    main()
