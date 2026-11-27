#!/usr/bin/env python3
"""
Reproduce all manuscript statistics and publication figures (Figures 1-4)
from the raw 100 Hz kinematic recordings in `modelinputs/`.

Usage:
  python reproduce.py                 # Compute all stats + render Figures 1-4
  python reproduce.py --stats-only    # Print all statistical tables only
  python reproduce.py --figure 2      # Render a single figure (1, 2, 3, or 4)
  python reproduce.py --export-web    # Also export docs/data/explorer_data.json for GitHub Pages
"""

import argparse
import glob
import json
import os
import time
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from model import (
    APRAXIA_CUTOFF,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    ITEM_INFO,
    TEST_PARTICIPANTS,
    BlendModel,
    evaluate_predictions,
    get_motion_type,
    load_all_data,
)
from plotting import fig1 as plot_fig1
from plotting import fig2 as plot_fig2
from plotting import fig3 as plot_fig3
from plotting import fig4 as plot_fig4

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "modelinputs")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
WEB_DATA_PATH = os.path.join(PROJECT_DIR, "docs", "data", "explorer_data.json")


FEATURE_DESCRIPTIONS = [
    "Mean 3D thumb-index distance normalized by participant hand length",
    "Within-trial SD of hand-length-normalized 3D thumb-index aperture",
    "Minimum hand-length-normalized 3D thumb-index aperture during gesture",
    "Maximum hand-length-normalized 3D thumb-index aperture during gesture",
    "Correlation between normalized thumb and index fingertip speeds",
    "Correlation between normalized index and middle fingertip speeds",
    "Correlation between normalized index fingertip and wrist speeds",
    "Correlation between normalized wrist and ipsilateral shoulder speeds",
    "Mean distance between within-trial standardized thumb and index trajectories",
    "SD of distance between standardized thumb and index trajectories",
    "Minimum distance between standardized thumb and middle trajectories",
    "Maximum distance between standardized thumb and middle trajectories",
    "Mean distance between standardized middle fingertip and wrist trajectories",
    "Mean absolute frame-to-frame change in standardized thumb-middle distance",
    "Number of opening steps in standardized thumb-middle aperture",
    "Number of closing steps in standardized thumb-middle aperture",
    "Correlation between standardized index and knuckle excursion magnitudes",
    "Correlation between standardized knuckle and elbow excursion magnitudes",
    "Correlation between standardized thumb and wrist excursion magnitudes",
    "SD of thumb velocity relative to arm (wrist-shoulder midpoint) along X",
    "SD of thumb velocity relative to arm (wrist-shoulder midpoint) along Y",
    "SD of thumb velocity relative to arm (wrist-shoulder midpoint) along Z",
    "Mean 3D velocity difference magnitude between thumb and ipsilateral shoulder",
    "SD of 3D velocity difference magnitude between thumb and ipsilateral shoulder",
    "Peak 3D velocity difference magnitude between thumb and ipsilateral shoulder",
    "Mean 3D synergy angle between proximal arm-hand and distal thumb-middle velocity vectors",
    "SD of 3D synergy angle between proximal and distal velocity vectors",
    "Range of 3D synergy angle between proximal and distal velocity vectors",
    "Speed-weighted unit movement direction of index fingertip along X",
    "Speed-weighted unit movement direction of index fingertip along Y",
    "Speed-weighted unit movement direction of index fingertip along Z (vertical)",
    "Ratio of vertical (Z) to horizontal (XY) standardized index velocity",
    "Cumulative 3D path length of within-trial standardized index trajectory",
    "Standardized index excursion range along X",
    "Standardized index excursion range along Y",
    "Standardized index excursion range along Z (vertical)",
    "Euclidean norm of 3D standardized index excursion range",
    "Mean standardized index fingertip 3D speed",
    "SD of standardized index fingertip 3D speed",
    "Peak standardized index fingertip 3D speed",
    "Number of local speed peaks (submovements) in smoothed index speed profile",
    "Mean dimensionless jerk magnitude of standardized index trajectory",
    "SD of dimensionless jerk magnitude of standardized index trajectory",
    "First non-zero lag peak of normalized index speed autocorrelation (rhythm)",
    "Segmented movement duration in seconds",
    "Category-matched kinematic feature (cumulative twist, downward strike, face lift, poke speed, or pinch modulation)",
]


def _clear_caches():
    for p in glob.glob(os.path.join(FIGURES_DIR, "*cache*.json")):
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
    print(f"  Clinical apraxia cut-off               : < {APRAXIA_CUTOFF:.2f} (0-3 scale)\n")

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
    print("\n  Item apraxia discriminability (ROC AUC across 43 stroke participants):")
    print(f"    Highest 5 items : {top5}")
    print(f"    Lowest 3 items  : {bot3}")
    return rows, item_scores, aucs


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

    top_f = np.argsort(-shares)[:8]
    print("\n  Top 8 kinematic features (training LOPO):")
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
    return paths


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
    return direct53, loio53, blend53, true53, coef53, itemw53, imp41, imp53


def export_web_data(rows, item_scores, aucs, res41, paths_test, full53):
    """Export compact, rich JSON bundle for the interactive GitHub Pages explorer (`docs/`)."""
    direct53, loio53, blend53, true53, coef53, itemw53, imp41, imp53 = full53
    X_all, sc_all, items_all, parts_all = load_all_data(DATA_DIR, downsample=15)
    valid_mask = np.array([c == 1 and t is not None for c, t in zip(sc_all["content"], sc_all["total"])])
    X = X_all[valid_mask]
    items = items_all[valid_mask]
    parts = [p for p, m in zip(parts_all, valid_mask) if m]
    hp_arr = [int(v) for v, m in zip(sc_all["hp"], valid_mask) if m]
    ap_arr = [int(v) for v, m in zip(sc_all["ap"], valid_mask) if m]
    at_arr = [int(v) for v, m in zip(sc_all["at"], valid_mask) if m]
    tot_arr = [int(v) for v, m in zip(sc_all["total"], valid_mask) if m]

    # Fit full N=53 model for signed standardized coefficients & compute LOPO item-level kinematic AUCs
    y_num = np.array(tot_arr, dtype=float)
    full_mdl = BlendModel(alpha_item=10, alpha_direct=10, blend_weight=0.5, calibrate=True)
    full_mdl.fit(X, y_num, items, parts)
    signed_beta53 = full_mdl.direct_model.coef_.tolist()

    # Out-of-sample item-level kinematic predictions across 53 LOPO folds (for kinematic AUC)
    parts_np = np.array(parts)
    lopo_trial_preds = np.zeros(len(X))
    for pid in sorted(set(parts)):
        tr = parts_np != pid
        ho = ~tr
        m_fold = BlendModel(alpha_item=10, alpha_direct=10, blend_weight=0.5, calibrate=True)
        m_fold.fit(X[tr], y_num[tr], items[tr], [p for p, k in zip(parts, tr) if k])
        lopo_trial_preds[ho] = m_fold._predict_trials(X[ho], items[ho])

    kin_auc_by_item = {}
    stroke_set = {p for p, r in rows.items() if not r["is_control"]}
    for item_id in sorted(ITEM_INFO.keys()):
        by_p = defaultdict(list)
        for pred_v, it_v, p_v in zip(lopo_trial_preds, items, parts):
            if int(it_v) == item_id and p_v in stroke_set:
                by_p[p_v].append(float(pred_v))
        if len(by_p) >= 10:
            p_list = sorted(by_p.keys())
            labs = np.array([1 if rows[p]["group"] == "apraxic" else 0 for p in p_list])
            vals = np.array([np.mean(by_p[p]) for p in p_list])
            kin_auc_by_item[item_id] = float(roc_auc_score(labs, -vals)) if len(set(labs)) > 1 else 0.5

    manual_auc_map = {r["id"]: r["auc"] for r in aucs}

    # Build item metadata + score distributions
    non_pids = [p for p, r in rows.items() if r["group"] == "nonaprax"]
    apr_pids = [p for p, r in rows.items() if r["group"] == "apraxic"]
    items_out = []
    for item_id in sorted(ITEM_INFO.keys()):
        info = ITEM_INFO[item_id]
        ps = item_scores.get(item_id, {})

        def dist_and_mean(pids):
            counts = [0, 0, 0, 0]
            vals = []
            for p in pids:
                tr_list = ps.get(p, [])
                if not tr_list:
                    continue
                m_val = float(np.mean(tr_list))
                vals.append(m_val)
                lvl = int(round(m_val))
                if 0 <= lvl <= 3:
                    counts[lvl] += 1
            n_p = len(vals)
            props = [round(c / n_p, 4) if n_p > 0 else 0.0 for c in counts]
            return props, (round(float(np.mean(vals)), 4) if n_p > 0 else None)

        dist_non, mean_non = dist_and_mean(non_pids)
        dist_apr, mean_apr = dist_and_mean(apr_pids)
        items_out.append(
            {
                "id": item_id,
                "name": info["name"],
                "category": info["category"],
                "motion_type": get_motion_type(item_id),
                "hand_posture": info["hand_posture"],
                "arm_posture": info["arm_posture"],
                "amplitude": info["amplitude"],
                "key_motion": info["key_motion"],
                "manual_auc": round(manual_auc_map.get(item_id, 0.5), 4),
                "kinematic_auc": round(kin_auc_by_item.get(item_id, 0.5), 4),
                "weight_train41": round(100.0 * res41["item_weights"].get(item_id, 0.0), 4),
                "weight_full53": round(100.0 * itemw53.get(item_id, 0.0), 4),
                "mean_nonaprax": mean_non,
                "mean_apraxic": mean_apr,
                "dist_nonaprax": dist_non,
                "dist_apraxic": dist_apr,
            }
        )

    # Feature metadata (46 features)
    fam_meta = [
        ("Hand posture", 0, 19, "#882255"),
        ("Arm posture & trajectory", 19, 32, "#E69F00"),
        ("Amplitude & timing", 32, 45, "#44AA99"),
        ("Motion-specific", 45, 46, "#7A7A84"),
    ]
    top8_train = set(int(i) for i in np.argsort(-np.array([imp41[k] for k in range(46)]))[:8])
    top8_full = set(int(i) for i in np.argsort(-np.array([imp53[k] for k in range(46)]))[:8])
    features_out = []
    for idx, fname in enumerate(FEATURE_NAMES):
        fam_label, fam_col = "Motion-specific", "#7A7A84"
        for fl, lo, hi, fc in fam_meta:
            if lo <= idx < hi:
                fam_label, fam_col = fl, fc
                break
        disp_name = "Motion-specific feature" if idx == 45 else fname
        features_out.append(
            {
                "index": idx,
                "name": disp_name,
                "family": fam_label,
                "color": fam_col,
                "description": FEATURE_DESCRIPTIONS[idx],
                "mean_mag_41": round(float(res41["mean_mag"][idx]), 5),
                "std_mag_41": round(float(res41["std_mag"][idx]), 5),
                "mean_mag_53": round(float(coef53[idx]), 5),
                "std_mag_53": round(float(coef53[46 + idx]), 5),
                "signed_beta_mean_53": round(float(signed_beta53[idx]), 5),
                "signed_beta_sd_53": round(float(signed_beta53[46 + idx]), 5),
                "rel_weight_41": round(float(imp41[idx]), 4),
                "rel_weight_53": round(float(imp53[idx]), 4),
                "in_top8_train": idx in top8_train,
                "in_top8_full": idx in top8_full,
            }
        )

    # Participant table (53 participants)
    test_set = set(TEST_PARTICIPANTS)
    participants_out = []
    for pid in sorted(rows.keys()):
        r = rows[pid]
        is_test = pid in test_set
        participants_out.append(
            {
                "id": pid,
                "group": r["group"],
                "split": "test" if is_test else "train",
                "n_trials": r["n"],
                "hp": round(r["hp"], 4),
                "ap": round(r["ap"], 4),
                "at": round(r["at"], 4),
                "true_score": round(r["total"], 4),
                "train_lopo": None
                if is_test
                else {
                    "part": round(res41["direct"][pid], 4),
                    "item": round(res41["loio"][pid], 4),
                    "blend": round(res41["blend"][pid], 4),
                },
                "heldout_test": {
                    "part": round(paths_test["part"][pid], 4),
                    "item": round(paths_test["item"][pid], 4),
                    "blend": round(paths_test["blend"][pid], 4),
                }
                if is_test
                else None,
                "full_lopo": {
                    "part": round(direct53[pid], 4),
                    "item": round(loio53[pid], 4),
                    "blend": round(blend53[pid], 4),
                },
            }
        )

    # Representative 3D trajectories across 4 diverse tools (Corkscrew, Hammer, Lipstick, Scissors)
    with open(os.path.join(DATA_DIR, "metadata.json"), "r") as f:
        md_raw = json.load(f)

    showcase_items = [
        (33, "Corkscrew", [("Control", "CL030", "trial253"), ("Stroke: without apraxia", "ST040", "trial744"), ("Stroke: with apraxia", "ST057", "trial1763")]),
        (11, "Hammer", None),
        (31, "Lipstick", None),
        (32, "Scissors", None),
    ]
    md_by_trial = {e["trial_name"]: e for e in md_raw}

    def pick_rep_trials(item_id):
        cands = {"control": [], "nonaprax": [], "apraxic": []}
        for e in md_raw:
            if e["item"] != item_id:
                continue
            p = e["participant"]
            if p not in rows:
                continue
            sc = e["score"][0] if isinstance(e["score"][0], list) else e["score"]
            if len(sc) < 4 or sc[0] != 1 or any(v is None or (isinstance(v, float) and np.isnan(v)) for v in sc[1:4]):
                continue
            tot = int(sc[1] + sc[2] + sc[3])
            cands[rows[p]["group"]].append((tot, rows[p]["total"], p, e["trial_name"]))
        chosen = []
        if cands["control"]:
            cands["control"].sort(key=lambda x: (-x[0], -x[1]))
            _, _, p, tn = cands["control"][0]
            chosen.append(("Control", p, tn))
        if cands["nonaprax"]:
            cands["nonaprax"].sort(key=lambda x: (-x[0], -x[1]))
            _, _, p, tn = cands["nonaprax"][0]
            chosen.append(("Stroke: without apraxia", p, tn))
        if cands["apraxic"]:
            cands["apraxic"].sort(key=lambda x: (x[0], x[1]))
            _, _, p, tn = cands["apraxic"][0]
            chosen.append(("Stroke: with apraxia", p, tn))
        return chosen

    sample_traces = {}
    for item_id, item_label, preset in showcase_items:
        triples = preset if preset is not None else pick_rep_trials(item_id)
        item_list = []
        for grp_label, pid, tname in triples:
            e = md_by_trial[tname]
            sc = e["score"][0] if isinstance(e["score"][0], list) else e["score"]
            arr100 = np.load(os.path.join(DATA_DIR, f"{tname}.npy")).astype(np.float64)
            t = arr100[:, 0] - arr100[0, 0]
            raw = arr100[:, 1:]
            step = max(1, int(round(len(t) / 180)))  # ~180 frames per trace for smooth interactive 3D
            ts = t[::step]
            rs = raw[::step]
            wrist_cm = (rs[:, 12:15] - rs[:, 12:15].mean(axis=0, keepdims=True)) * 100.0
            index_cm = (rs[:, 3:6] - rs[:, 12:15].mean(axis=0, keepdims=True)) * 100.0
            thumb_cm = (rs[:, 0:3] - rs[:, 12:15].mean(axis=0, keepdims=True)) * 100.0
            speed_cms = np.linalg.norm(rs[:, 36:39], axis=1) * 100.0
            aperture_cm = np.linalg.norm(rs[:, 0:3] - rs[:, 3:6], axis=1) * 100.0
            v1 = rs[:, 18:21] - rs[:, 15:18]
            v2 = rs[:, 12:15] - rs[:, 15:18]
            cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
            elbow_deg = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            tnorm = 100.0 * (ts - ts[0]) / max(ts[-1] - ts[0], 1e-6)
            item_list.append(
                {
                    "label": grp_label,
                    "participant": pid,
                    "trial_name": tname,
                    "score_total": int(sc[1] + sc[2] + sc[3]),
                    "score_hp": int(sc[1]),
                    "score_ap": int(sc[2]),
                    "score_at": int(sc[3]),
                    "duration": round(float(t[-1] - t[0]), 2),
                    "tnorm": [round(float(v), 2) for v in tnorm],
                    "wrist_xyz": [[round(float(c), 2) for c in row] for row in wrist_cm],
                    "index_xyz": [[round(float(c), 2) for c in row] for row in index_cm],
                    "thumb_xyz": [[round(float(c), 2) for c in row] for row in thumb_cm],
                    "aperture": [round(float(v), 2) for v in aperture_cm],
                    "elbow_angle": [round(float(v), 2) for v in elbow_deg],
                    "wrist_speed": [round(float(v), 2) for v in speed_cms],
                }
            )
        sample_traces[item_label] = item_list

    # Compact trial feature matrix (1,950 valid trials x 46 features) for live browser Ridge modelling
    trials_compact = {
        "participant": parts,
        "item": [int(i) for i in items],
        "hp": hp_arr,
        "ap": ap_arr,
        "at": at_arr,
        "total": tot_arr,
        "X": [[round(float(v), 4) for v in row] for row in X],
    }

    os.makedirs(os.path.dirname(WEB_DATA_PATH), exist_ok=True)
    web_bundle = {
        "cutoff": APRAXIA_CUTOFF,
        "test_participants": TEST_PARTICIPANTS,
        "features": features_out,
        "items": items_out,
        "participants": participants_out,
        "sample_traces": sample_traces,
        "trials": trials_compact,
    }
    with open(WEB_DATA_PATH, "w") as f:
        json.dump(web_bundle, f, separators=(",", ":"))
    sz_kb = os.path.getsize(WEB_DATA_PATH) / 1024.0
    print(f"\nExported interactive explorer dataset to docs/data/explorer_data.json ({sz_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all manuscript statistics and figures from raw kinematic recordings."
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Compute and print all manuscript statistics without rendering figures.",
    )
    parser.add_argument(
        "--figure",
        type=int,
        choices=[1, 2, 3, 4],
        help="Render only a specific figure (1, 2, 3, or 4).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse cached fold JSON files in figures/ if present.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep temporary fold JSON cache files in figures/ after completion.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Render standalone Matplotlib subplot PNGs without SVG schematic headers.",
    )
    parser.add_argument(
        "--export-web",
        action="store_true",
        help="Export docs/data/explorer_data.json for the interactive GitHub Pages companion site.",
    )
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if args.no_header:
        os.environ["NO_HEADER"] = "1"

    if not args.use_cache:
        _clear_caches()

    if args.figure is not None and not args.export_web:
        runners = {1: plot_fig1.main, 2: plot_fig2.main, 3: plot_fig3.main, 4: plot_fig4.main}
        runners[args.figure]()
        if not args.keep_cache and not args.use_cache:
            _clear_caches()
        return

    rows, item_scores, aucs = print_section_1()
    res41 = print_section_2()
    paths_test = print_section_3()
    full53 = print_section_4(res41)

    if args.export_web:
        export_web_data(rows, item_scores, aucs, res41, paths_test, full53)

    if not args.stats_only:
        print("\n" + "=" * 88)
        print("5. GENERATING MANUSCRIPT FIGURES (Figures 1-4)")
        print("=" * 88)
        targets = [args.figure] if args.figure else [1, 2, 3, 4]
        runners = {1: plot_fig1.main, 2: plot_fig2.main, 3: plot_fig3.main, 4: plot_fig4.main}
        for fig_num in targets:
            runners[fig_num]()

    if not args.keep_cache and not args.use_cache:
        _clear_caches()

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
