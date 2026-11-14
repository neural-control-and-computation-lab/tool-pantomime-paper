"""
Figure 3 — Out-of-sample evaluation on the 12 held-out test participants
(3 healthy controls, 6 stroke without apraxia, 3 stroke with apraxia; model
trained once on all 41 training participants).

Panels:
  A  Model evaluation pipeline (vector header)
  B  Held-out test true vs. predicted praxis scores (n = 12)
  C  Pathway comparison (Participant, Item, Blended: Pearson r, AUC, MAE)
  D  Stroke apraxia confusion matrix, sensitivity, and specificity
  E  Participant-level model score contributions across the 46 features for the
     three test patients with apraxia (ST028, ST034, ST067) vs. stroke without apraxia range
  F  Item-level prediction errors (predicted - true) across the 39 items

Rows in Panels E and F are ordered from top to bottom by how far the missed
borderline patient (ST028) lies above the mean of the two detected patients
with apraxia (ST034, ST067).

Outputs:
  figures/figure3.svg
  figures/fig3.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from item_info import ITEM_INFO
from model import APRAXIA_CUTOFF, FEATURE_NAMES, BlendModel, load_test_data, load_train_data

plt.rcParams["font.family"] = "DejaVu Sans"

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_dir, "modelinputs")
figures_dir = os.path.join(project_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)
path_cache = os.path.join(figures_dir, "fig3_pathway_cache.json")

cmat_color = "#2E7D32"
apraxia_cutoff = APRAXIA_CUTOFF
ax_lim = (1.7, 3.1)
c_control = "#0A1F4D"
c_nonaprax = "#1976D2"
c_aprax = "#D32F2F"
dark_red = "#7F0000"
dark_blue = "#0A2E5C"

FS_AXIS_LABEL = 14.0
FS_TICK = 12.0
FS_ITEM = 10.8
FS_FEAT = 9.6
FS_ANNOT = 10.5
FS_HEADER = 13.5
FS_QUADRANT = 13.5

missed = ["ST028"]
caught = ["ST034", "ST067"]
c_miss = "#D32F2F"
c_caught = "#A34234"
c_caught2 = "#4E0F2E"

full_names = list(FEATURE_NAMES)
full_names[45] = "Motion-specific feature"
item_names = {k: v["name"] for k, v in ITEM_INFO.items()}


def compute_or_load_pathways():
    """Fit `BlendModel` on all 41 training participants and evaluate the 12 test participants."""
    if os.path.exists(path_cache):
        with open(path_cache, "r") as f:
            return json.load(f)

    print("Fitting BlendModel on training set (n = 41) and evaluating held-out test set (n = 12)...")
    xtr, sctr, itr, ptr = load_train_data(data_dir, downsample=15)
    ytr = np.array([t if t is not None else np.nan for t in sctr["total"]])
    m = ~np.isnan(ytr)
    xtr, ytr, itr = xtr[m], ytr[m], itr[m]
    ptr = [p for p, k in zip(ptr, m) if k]

    mdl = BlendModel(alpha_item=10, alpha_direct=10, blend_weight=0.5, calibrate=True)
    mdl.fit(xtr, ytr, itr, ptr)

    xte, scte, ite, pte = load_test_data(data_dir, downsample=15)
    yte = np.array([t if t is not None else np.nan for t in scte["total"]])
    m = ~np.isnan(yte)
    xte, yte, ite = xte[m], yte[m], ite[m]
    pte = np.array([p for p, k in zip(pte, m) if k])

    def aggv_of(X, parts, who):
        f = X[np.array([p == who for p in parts])]
        return np.concatenate([f.mean(0), f.std(0)])

    A = np.array([aggv_of(xtr, np.array(ptr), p) for p in sorted(set(ptr))])
    mu, sd = A.mean(0), A.std(0) + 1e-9
    beta = mdl.direct_model.coef_
    nf = len(FEATURE_NAMES)

    out = {
        "part": {},
        "item": {},
        "blend": {},
        "true": {},
        "feat_contrib": {},
        "item_preds": {},
        "item_w": {},
        "item_detail": {},
        "names": list(FEATURE_NAMES),
        "global_mean": float(mdl.global_mean),
    }
    for p in sorted(set(pte)):
        k = pte == p
        tp = mdl._predict_trials(xte[k], ite[k])
        Iv = ite[k]
        w = np.array([mdl.item_weights.get(i, 0.1) for i in Iv])
        w = w / w.sum()
        raw = float(np.sum(tp * w))
        out["item"][p] = float(np.clip(mdl.global_mean + (raw - mdl.global_mean) * (1 + mdl.bias_factor), 0, 3))
        aggv = aggv_of(xte, pte, p)
        out["part"][p] = float(
            np.clip(mdl.direct_model.predict(mdl.direct_scaler.transform(np.nan_to_num(aggv.reshape(1, -1))))[0], 0, 3)
        )
        out["blend"][p] = float(mdl.predict_participant(xte[k], ite[k], [p] * int(k.sum()))[p])
        out["true"][p] = float(yte[k].mean())

        # Participant-level pathway linear contribution (mean + SD halves summed per feature)
        z = (aggv - mu) / sd
        c = beta * z
        out["feat_contrib"][p] = (c[:nf] + c[nf:]).tolist()
        out["item_preds"][p] = tp.tolist()
        out["item_w"][p] = w.tolist()
        out["item_detail"][p] = {
            "items": [int(v) for v in Iv],
            "pred": tp.tolist(),
            "true": yte[k].tolist(),
        }

    with open(path_cache, "w") as f:
        json.dump(out, f)
    return out


def draw_scatter(ax, predictions, true_means):
    pids = sorted(predictions)
    pred = np.array([predictions[p] for p in pids])
    true = np.array([true_means[p] for p in pids])
    is_ctrl = np.array([p.startswith("CL") for p in pids])
    mis = np.array(
        [(not is_ctrl[i]) and ((true[i] < apraxia_cutoff) != (pred[i] < apraxia_cutoff)) for i in range(len(pids))]
    )
    mn, mx = ax_lim
    ax.add_patch(
        Rectangle((mn, mn), apraxia_cutoff - mn, apraxia_cutoff - mn, facecolor=c_aprax, alpha=0.10, zorder=0)
    )
    ax.add_patch(
        Rectangle(
            (apraxia_cutoff, apraxia_cutoff),
            mx - apraxia_cutoff,
            mx - apraxia_cutoff,
            facecolor=c_nonaprax,
            alpha=0.10,
            zorder=0,
        )
    )
    ax.axhline(apraxia_cutoff, color="#888", lw=1.2, ls="--", zorder=1)
    ax.axvline(apraxia_cutoff, color="#888", lw=1.2, ls="--", zorder=1)
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.25, lw=1.5, zorder=1)
    for i, p in enumerate(pids):
        if is_ctrl[i]:
            color, marker, size = c_control, "o", 150
        else:
            color = c_aprax if true[i] < apraxia_cutoff else c_nonaprax
            marker = "X" if mis[i] else "o"
            size = 230 if mis[i] else 150
        ax.scatter(
            true[i],
            pred[i],
            c=color,
            edgecolors="black",
            s=size,
            marker=marker,
            linewidths=1.2,
            alpha=0.95,
            zorder=4,
        )
        if (not is_ctrl[i]) and true[i] < apraxia_cutoff:
            col = {missed[0]: c_miss, caught[0]: c_caught, caught[1]: c_caught2}.get(p, "#333")
            side_x, ha_al = (11, "left") if p == caught[1] else (-11, "right")
            ax.annotate(
                p,
                (true[i], pred[i]),
                xytext=(side_x, 0),
                textcoords="offset points",
                ha=ha_al,
                va="center",
                fontsize=FS_ANNOT,
                fontweight="bold",
                color=col,
            )
    ax.set_xlabel("True Score", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.set_ylabel("Predicted Score", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_linewidth(1.1)
        sp.set_color("#333")
    r = pearsonr(pred, true)[0]
    rho = spearmanr(pred, true)[0]
    mae = float(np.mean(np.abs(pred - true)))
    lab = (true[~is_ctrl] < apraxia_cutoff).astype(int)
    auc = roc_auc_score(lab, -pred[~is_ctrl]) if len(set(lab)) > 1 else float("nan")
    return r, rho, mae, auc


def pathway_metrics(pred, true):
    pids = sorted(pred)
    p = np.array([pred[k] for k in pids])
    t = np.array([true[k] for k in pids])
    r = pearsonr(p, t)[0]
    mae = float(np.mean(np.abs(p - t)))
    pats = [k for k in pids if k.startswith("ST")]
    tp = np.array([true[k] for k in pats])
    pp = np.array([pred[k] for k in pats])
    lab = (tp < apraxia_cutoff).astype(int)
    auc = roc_auc_score(lab, -pp) if len(set(lab)) > 1 else float("nan")
    return r, mae, auc


def draw_perf(ax, paths):
    models = ["part", "item", "blend"]
    display = ["Participant", "Item", "Blended"]
    mets = [pathway_metrics(paths[m], paths["true"]) for m in models]
    r = [x[0] for x in mets]
    mae = [x[1] for x in mets]
    auc = [x[2] for x in mets]
    xpos = np.arange(3)
    wbar = 0.38
    b1 = ax.bar(xpos - wbar / 2, r, wbar, color="#4d4d4d", label="$r$")
    b2 = ax.bar(xpos + wbar / 2, auc, wbar, color="#bdbdbd", label="AUC")
    for bars, vals in ((b1, r), (b2, auc)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=FS_ANNOT)
    ax.set_xticks(xpos)
    ax.set_xticklabels(display, fontsize=FS_TICK, fontweight="bold")
    for xp, e in zip(xpos, mae):
        ax.text(xp, -0.225, f"MAE {e:.3f}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS_TICK)
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Performance", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def draw_part_dots(ax, paths):
    true = paths["true"]
    names = full_names
    nf = len(names)
    apr = sorted([p for p in paths["feat_contrib"] if p.startswith("ST") and true[p] < apraxia_cutoff], key=lambda q: true[q])
    non = [p for p in paths["feat_contrib"] if p.startswith("ST") and true[p] >= apraxia_cutoff]
    st = np.array(paths["feat_contrib"][missed[0]])
    cm = np.mean([paths["feat_contrib"][p] for p in apr if p != missed[0]], axis=0)
    order = np.argsort(-(st - cm))
    y = np.arange(nf)

    N = np.array([paths["feat_contrib"][p] for p in non])
    lo, hi = N.min(0)[order], N.max(0)[order]
    ax.barh(y, hi - lo, left=lo, height=0.66, color="#9e9e9e", alpha=0.24, edgecolor="none", zorder=0)
    ax.axvline(0, color="#333", lw=1.1, zorder=1)

    cols = {missed[0]: c_miss, caught[0]: c_caught, caught[1]: c_caught2}
    prof = {p: np.array(paths["feat_contrib"][p])[order] for p in apr}
    apr_top_order = sorted(apr, key=lambda q: prof[q][0])
    top_offsets = {
        apr_top_order[0]: (-8, 9, "right"),
        apr_top_order[1]: (2, 9, "left"),
        apr_top_order[2]: (0, 9, "center"),
    }
    for p in apr:
        v = prof[p]
        star = p == missed[0]
        ax.plot(
            v,
            y,
            "-",
            color=cols[p],
            lw=2.4 if star else 1.5,
            alpha=1.0 if star else 0.70,
            zorder=5 if star else 3,
            solid_capstyle="round",
        )
        ax.scatter(
            v,
            y,
            s=42 if star else 20,
            c=cols[p],
            marker="X" if star else "o",
            edgecolors="black" if star else "none",
            linewidths=0.9 if star else 0,
            alpha=1.0 if star else 0.75,
            zorder=6 if star else 4,
        )
        dx_pt, dy_pt, ha_align = top_offsets[p]
        ax.annotate(
            p,
            (v[0], y[0]),
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            ha=ha_align,
            va="bottom",
            fontsize=FS_ANNOT,
            fontweight="bold",
            color=cols[p],
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4) if p == apr_top_order[1] else None,
            zorder=7,
        )

    vmax = max(float(np.abs(N).max()), max(float(np.abs(prof[p]).max()) for p in apr))
    xlim_val = max(0.08, float(np.ceil(vmax * 1.15 * 20.0) / 20.0))
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=FS_FEAT)
    ax.set_xlim(-xlim_val, xlim_val)
    ax.set_ylim(-1.6, nf + 0.3)
    ax.invert_yaxis()
    jj = 1 + int(np.argmax(hi[1 : min(12, nf)]))
    ax.text(min(hi[jj], xlim_val * 0.98), y[jj] - 0.42, "Stroke: without apraxia", ha="right", va="bottom", fontsize=FS_ANNOT, color="#6E6E78")
    ax.set_xlabel("Score Contribution", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.tick_params(axis="x", labelsize=FS_TICK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def draw_item_err(ax, paths):
    true_p = paths["true"]
    det = paths["item_detail"]
    apr = sorted([p for p in det if p.startswith("ST") and true_p[p] < apraxia_cutoff], key=lambda q: true_p[q])
    non = [p for p in det if p.startswith("ST") and true_p[p] >= apraxia_cutoff]
    err = {p: {i: pr - tr for i, pr, tr in zip(det[p]["items"], det[p]["pred"], det[p]["true"])} for p in det}

    items = sorted(set.intersection(*[set(err[p]) for p in apr]))
    st = np.array([err[missed[0]][i] for i in items])
    cm = np.mean([[err[p][i] for i in items] for p in apr if p != missed[0]], axis=0)
    order = np.argsort(-(st - cm))
    items = [items[k] for k in order]
    y = np.arange(len(items))

    band = np.array([[err[p].get(i, np.nan) for i in items] for p in non])
    lo, hi = np.nanmin(band, axis=0), np.nanmax(band, axis=0)
    ax.barh(y, hi - lo, left=lo, height=0.66, color="#9e9e9e", alpha=0.24, edgecolor="none", zorder=0)
    ax.axvline(0, color="#333", lw=1.1, zorder=1)
    mj = int(np.argmin(lo))
    ax.text(lo[mj] + 0.16, y[mj] - 0.38, "Stroke:\nwithout apraxia", ha="left", va="bottom", fontsize=FS_ANNOT, color="#6E6E78", linespacing=0.95)

    cols = {missed[0]: c_miss, caught[0]: c_caught, caught[1]: c_caught2}
    for p in apr:
        v = np.array([err[p][i] for i in items])
        star = p == missed[0]
        ax.plot(
            v,
            y,
            "-",
            color=cols[p],
            lw=2.8 if star else 1.8,
            alpha=1.0 if star else 0.70,
            zorder=5 if star else 3,
            solid_capstyle="round",
        )
        ax.scatter(
            v,
            y,
            s=64 if star else 30,
            c=cols[p],
            marker="X" if star else "o",
            edgecolors="black" if star else "none",
            linewidths=1.0 if star else 0,
            alpha=1.0 if star else 0.75,
            zorder=6 if star else 4,
        )
        others = np.mean([[err[q][i] for i in items] for q in apr if q != p], axis=0)
        j = int(np.argmax(np.abs(v - others)))
        side = 1 if v[j] >= others[j] else -1
        ax.annotate(
            p,
            (v[j], y[j]),
            xytext=(10 * side, 0),
            textcoords="offset points",
            ha="left" if side > 0 else "right",
            va="center",
            fontsize=FS_ANNOT,
            fontweight="bold",
            color=cols[p],
        )

    ax.set_yticks(y)
    ax.set_yticklabels([item_names.get(i, f"Item {i}") for i in items], fontsize=FS_ITEM)
    ax.set_ylim(-1.0, len(items) + 0.3)
    ax.invert_yaxis()
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels(["-2", "-1", "0", "1", "2"])
    ax.set_xlabel("Prediction Error", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def draw_confusion(ax, paths):
    pred, true = paths["blend"], paths["true"]
    pats = [p for p in sorted(pred) if p.startswith("ST")]
    t = np.array([true[p] < apraxia_cutoff for p in pats])
    q = np.array([pred[p] < apraxia_cutoff for p in pats])
    cm = np.array([[int((t & ~q).sum()), int((~t & ~q).sum())], [int((t & q).sum()), int((~t & q).sum())]])
    fn, tn = cm[0, 0], cm[0, 1]
    tp, fp = cm[1, 0], cm[1, 1]
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    ax.set_anchor("W")
    cmap = LinearSegmentedColormap.from_list("cmat", ["#ffffff", cmat_color])
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=FS_HEADER + 3, fontweight="bold", color="#1F2328")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["With\napraxia", "Without\napraxia"], fontsize=FS_ITEM, ma="center")
    ax.set_yticklabels(["Without\napraxia", "With\napraxia"], fontsize=FS_ITEM, rotation=90, va="center", ma="center")
    ax.set_xlabel("True", fontsize=FS_AXIS_LABEL, fontweight="bold", labelpad=6)
    ax.set_ylabel("Predicted", fontsize=FS_AXIS_LABEL, fontweight="bold", labelpad=6)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(length=0)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return sens, spec


def main():
    paths = compute_or_load_pathways()
    predictions = paths["blend"]
    true_means = paths["true"]

    fig = plt.figure(figsize=(17, 18.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.20, 1.0], wspace=0.04)
    gsL = outer[0, 0].subgridspec(2, 1, height_ratios=[0.82, 1.44], hspace=0.20)
    gsB = gsL[1, 0].subgridspec(1, 2, width_ratios=[1.0, 0.13], wspace=0.02)
    axA = fig.add_subplot(gsL[0, 0])
    axB = fig.add_subplot(gsB[0, 0])
    axC = fig.add_subplot(outer[0, 1])

    r, rho, mae, auc = draw_scatter(axA, predictions, true_means)
    print(f"  Held-out test (n=12): r={r:.3f}  rho={rho:.3f}  MAE={mae:.3f}  AUC={auc:.3f}")
    mn, mx = ax_lim
    axA.text(mn + 0.06, mn + 0.06, "With\nApraxia", fontsize=FS_QUADRANT, color=dark_red, ha="left", va="bottom", fontstyle="italic", fontweight="bold")
    axA.text(mx - 0.06, mx - 0.06, "Without\nApraxia", fontsize=FS_QUADRANT, color=dark_blue, ha="right", va="top", fontstyle="italic", fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_control, markeredgecolor="black", markersize=11, label="Control"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_nonaprax, markeredgecolor="black", markersize=11, label="Stroke: without apraxia"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_aprax, markeredgecolor="black", markersize=11, label="Stroke: with apraxia"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#888", markeredgecolor="black", markersize=12, label="Misclassified"),
    ]
    axA.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.988), borderaxespad=0, fontsize=FS_ANNOT, frameon=False)

    draw_part_dots(axB, paths)
    draw_item_err(axC, paths)

    axA.set_anchor("W")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    lx = lambda ax: inv.transform((ax.yaxis.label.get_window_extent(renderer=rend).x0, 0))[0]
    tickx = min(inv.transform((t.get_window_extent(renderer=rend).x0, 0))[0] for t in axB.get_yticklabels())
    delta = tickx - lx(axA)
    pA = axA.get_position()
    axA.set_position([pA.x0 + delta, pA.y0, pA.width, pA.height])
    fig.canvas.draw()
    pA = axA.get_position()

    pR = outer[0, 0].get_position(fig)
    rx0 = pA.x1 + 0.052
    axP = fig.add_axes([rx0 + 0.024, pA.y1 - pA.height * 0.33, (pR.x1 - rx0) * 0.74, pA.height * 0.33])
    draw_perf(axP, paths)

    cm_w = 1.55 / fig.get_figwidth()
    cm_h = 1.55 / fig.get_figheight()
    axM = fig.add_axes([rx0 + 0.024, pA.y0, cm_w, cm_h])
    sens, spec = draw_confusion(axM, paths)
    stexts = []
    for y, lab, val in ((0.60, "Sensitivity:", sens), (0.42, "Specificity:", spec)):
        axM.text(1.84, y, lab, transform=axM.transAxes, ha="right", va="center", fontsize=FS_HEADER, fontweight="bold", color="#333")
        stexts.append(axM.text(1.89, y, f"{val:.2f}", transform=axM.transAxes, ha="left", va="center", fontsize=FS_HEADER, color="#333"))

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tgt = max(inv.transform((t.get_window_extent(renderer=rend).x1, 0))[0] for t in stexts)
    for ax_ in (axP, axB):
        pp = ax_.get_position()
        ax_.set_position([pp.x0, pp.y0, max(tgt - pp.x0, 0.05), pp.height])

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pM = axM.get_position()
    ref_x = -0.5
    axM.yaxis.set_label_coords(ref_x, 0.5)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    rbx = lambda ax: inv.transform((ax.yaxis.label.get_window_extent(renderer=rend).x1, 0))[0]
    axM.yaxis.set_label_coords(ref_x + (rbx(axP) - rbx(axM)) / pM.width, 0.5)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    lbl_left = min(inv.transform((t.get_window_extent(renderer=rend).x0, 0))[0] for t in axC.get_yticklabels())
    shift = (tgt + 0.024) - lbl_left
    pC = axC.get_position()
    axC.set_position([pC.x0 + shift, pC.y0, (pC.width - shift) * 0.90, pC.height])

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ylb = lambda ax: inv.transform((0, ax.xaxis.label.get_window_extent(renderer=rend).y0))[1]
    pMM = axM.get_position()
    axM.set_position([pMM.x0, pMM.y0 + (ylb(axA) - ylb(axM)), pMM.width, pMM.height])

    out = os.path.join(figures_dir, "fig3.png")
    if os.environ.get("NO_HEADER"):
        fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
        print(f"Saved {out}")
    else:
        from svg_composer import compose_fig3
        compose_fig3(fig, axA, axP, axM, axB, axC, out, dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
