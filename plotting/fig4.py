"""
Figure 4 — Full-cohort Leave-One-Participant-Out (LOPO) cross-validation across
all 53 participants (10 healthy controls, 26 stroke without apraxia, 17 stroke
with apraxia) and stability of learned feature and item weights relative to the
41-participant training cohort.

Panels:
  A  Full-cohort LOPO cross-validation pipeline (vector header)
  B  Participant-level relative feature weight (%) comparison (N = 41 vs. N = 53)
  C  Item-level relative reliability weight (%) comparison (N = 41 vs. N = 53)
  D  Cross-validated true vs. predicted praxis scores (N = 53)
  E  Pathway comparison (Participant, Item, Blended: Pearson r, AUC, MAE)
  F  Stroke apraxia confusion matrix, sensitivity, and specificity

Outputs:
  figures/figure4.svg
  figures/fig4.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from model import FEATURE_NAMES, BlendModel, load_all_data
from plotting.fig2 import (
    DEFAULT_AX_LIM,
    FS_ANNOT,
    FS_AXIS_LABEL,
    FS_QUADRANT,
    FS_TICK,
    apraxia_cutoff,
    c_aprax,
    c_control,
    c_nonaprax,
    compute_or_load_lopo,
    dark_blue,
    dark_red,
    draw_comparison,
    draw_confusion,
    draw_scatter,
    item_names,
    loio_color,
    model_metrics,
)

plt.rcParams["font.family"] = "DejaVu Sans"

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_dir, "modelinputs")
figures_dir = os.path.join(project_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)
cache_path = os.path.join(figures_dir, "fig4_cache.json")

FAMILIES = [
    ("Hand posture", 0, 19, "#882255"),
    ("Arm posture", 19, 32, "#E69F00"),
    ("Amplitude & timing", 32, 45, "#44AA99"),
    ("Motion type", 45, 46, "#7A7A84"),
]

DISPLAY_NAMES = {45: "Motion-specific feature"}


def compute_or_load():
    """Run (or load cached) 53-fold full-cohort LOPO cross-validation."""
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            d = json.load(f)
        return (
            d["direct"],
            d["loio"],
            d["blend"],
            d["true"],
            np.array(d["coef_mag"]),
            {int(k): v for k, v in d["item_w"].items()},
        )

    print("Running 53-fold full-cohort LOPO cross-validation...")
    x, sc, items, parts = load_all_data(data_dir, downsample=15)
    y = np.array([t if t is not None else np.nan for t in sc["total"]])
    m = ~np.isnan(y)
    x, y, items = x[m], y[m], items[m]
    parts = np.array([p for p, k in zip(parts, m) if k])
    part_true = {p: float(y[parts == p].mean()) for p in set(parts)}

    direct, loio, blend = {}, {}, {}
    coefs, iws = [], []
    item_ids = sorted(set(int(i) for i in items))
    pids = sorted(set(parts))
    for pid in pids:
        tr = parts != pid
        ho = parts == pid
        mdl = BlendModel(alpha_item=10, alpha_direct=10, blend_weight=0.5, calibrate=True)
        mdl.fit(x[tr], y[tr], items[tr], list(parts[tr]))

        d_pred, l_pred, b_pred = mdl.predict_pathways(x[ho], items[ho], list(parts[ho]))
        direct[pid] = d_pred[pid]
        loio[pid] = l_pred[pid]
        blend[pid] = b_pred[pid]

        coefs.append(np.abs(mdl.direct_model.coef_))
        iws.append([mdl.item_weights.get(i, np.nan) for i in item_ids])

    coef_mag = np.array(coefs).mean(0)
    item_w = {int(i): float(v) for i, v in zip(item_ids, np.nanmean(np.array(iws), 0))}
    with open(cache_path, "w") as f:
        json.dump(
            {
                "direct": direct,
                "loio": loio,
                "blend": blend,
                "true": part_true,
                "coef_mag": coef_mag.tolist(),
                "item_w": {str(k): v for k, v in item_w.items()},
            },
            f,
        )
    return direct, loio, blend, part_true, coef_mag, item_w


def _importance(coef):
    """Percentage of total |coefficient| (mean + SD halves summed) per feature."""
    nf = len(coef) // 2
    tot = np.abs(coef[:nf]) + np.abs(coef[nf:])
    return {i: float(v) for i, v in enumerate(100.0 * tot / tot.sum())}


def draw_feature_compare(sf, imp45, imp57):
    rows = []
    for _, lo, hi, colour in FAMILIES:
        members = sorted(
            range(lo, hi),
            key=lambda f: max(imp45.get(f, 0), imp57.get(f, 0)),
            reverse=True,
        )
        for f in members:
            rows.append((f, colour))
    n = len(rows)
    ax = sf.add_axes([0.52, 0.09, 0.45, 0.80])
    vmax_w = max(max(imp45.values()), max(imp57.values()))
    xmax = max(6.0, float(np.ceil(vmax_w * 1.15)))

    for r, (f, colour) in enumerate(rows):
        y = n - 1 - r
        ax.plot([0, xmax], [y, y], color="#E6E6EA", lw=0.7, zorder=0)
        a, b = imp45[f], imp57[f]
        ax.plot([a, b], [y, y], color=colour, lw=1.6, alpha=0.5, zorder=2)
        ax.scatter(a, y, s=42, facecolor="white", edgecolor=colour, linewidth=1.5, zorder=3)
        ax.scatter(b, y, s=42, facecolor=colour, edgecolor=colour, linewidth=1.5, zorder=3)

    ax.set_yticks(range(n))
    ax.set_yticklabels([DISPLAY_NAMES.get(f, FEATURE_NAMES[f]) for f, _ in rows][::-1], fontsize=8.6, color="black")
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Relative weight (%)", fontsize=FS_AXIS_LABEL, fontweight="bold", labelpad=9)
    tick_step = 1 if xmax <= 7 else (2 if xmax <= 12 else 3)
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.tick_params(axis="y", length=0, pad=5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#444", markeredgewidth=1.5, markersize=7, label="46 features (41 training)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#444", markeredgecolor="#444", markersize=7, label="46 features (all 53)"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.080),
        frameon=False,
        fontsize=FS_ANNOT,
        handletextpad=0.4,
        ncol=2,
        columnspacing=1.5,
    )
    return ax


def draw_item_compare(sf, w45, w57):
    ids = sorted(set(w45) & set(w57), key=lambda i: w57[i], reverse=True)
    n = len(ids)
    ax = sf.add_axes([0.40, 0.09, 0.56, 0.80])
    xmax = max(8.0, float(np.ceil(max(max(w45.values()), max(w57.values())) * 100.0 * 1.12)))

    for r, i in enumerate(ids):
        y = n - 1 - r
        a, b = w45[i] * 100, w57[i] * 100
        ax.plot([0, xmax], [y, y], color="#E6E6EA", lw=0.7, zorder=0)
        ax.plot([a, b], [y, y], color=loio_color, lw=1.5, alpha=0.6, zorder=2)
        ax.scatter(a, y, s=44, facecolor="white", edgecolor=loio_color, linewidth=1.5, zorder=3)
        ax.scatter(b, y, s=44, facecolor=loio_color, edgecolor=loio_color, linewidth=1.5, zorder=3)

    ax.set_yticks(range(n))
    ax.set_yticklabels([item_names.get(i, f"Item {i}") for i in ids][::-1], fontsize=9.4, color="black")
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Relative weight (%)", fontsize=FS_AXIS_LABEL, fontweight="bold", labelpad=9)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.tick_params(axis="y", length=0, pad=5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=loio_color, markeredgewidth=1.5, markersize=7, label="46 features (41 training)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=loio_color, markeredgecolor=loio_color, markersize=7, label="46 features (all 53)"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.080),
        frameon=False,
        fontsize=FS_ANNOT,
        handletextpad=0.4,
        ncol=2,
        columnspacing=1.5,
    )
    return ax


def main():
    direct, loio, blend, true_means, coef57, itemw57 = compute_or_load()
    predictions = blend
    mets = {
        "Direct": model_metrics(direct, true_means),
        "LOIO": model_metrics(loio, true_means),
        "Blend": model_metrics(blend, true_means),
    }

    lo = min(min(predictions.values()), min(true_means.values()))
    hi = max(max(predictions.values()), max(true_means.values()))
    fig4_lim = (min(DEFAULT_AX_LIM[0], lo - 0.05), max(DEFAULT_AX_LIM[1], hi + 0.05))

    f2_res = compute_or_load_lopo()
    imp45 = _importance(np.concatenate([f2_res["mean_mag"], f2_res["std_mag"]]))
    imp57 = _importance(coef57)
    w45 = f2_res["item_weights"]

    fig = plt.figure(figsize=(12.0, 14.8))
    sfTop, sfBot = fig.subfigures(2, 1, height_ratios=[1.62, 1.0], hspace=0.135)
    sfG, sfH = sfTop.subfigures(1, 2, width_ratios=[1.31, 1.0], wspace=0.3569)
    sfD, sfRC = sfBot.subfigures(1, 2, width_ratios=[1.31, 1.0], wspace=0.3569)

    axG = draw_feature_compare(sfG, imp45, imp57)
    axH = draw_item_compare(sfH, w45, itemw57)

    axD = sfD.add_axes([0.14, 0.11, 0.84, 0.86])
    draw_scatter(axD, predictions, true_means, lim=fig4_lim)
    axD.set_anchor("W")
    mn, mx = fig4_lim
    axD.text(mn + 0.02, mn + 0.02, "With\nApraxia", fontsize=FS_QUADRANT, color=dark_red, ha="left", va="bottom", fontstyle="italic", fontweight="bold")
    axD.text(mx - 0.02, mx - 0.02, "Without\nApraxia", fontsize=FS_QUADRANT, color=dark_blue, ha="right", va="top", fontstyle="italic", fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_control, markeredgecolor="black", markersize=9, label="Control"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_nonaprax, markeredgecolor="black", markersize=9, label="Stroke: without apraxia"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_aprax, markeredgecolor="black", markersize=9, label="Stroke: with apraxia"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#888", markeredgecolor="black", markersize=10, label="Misclassified"),
    ]
    axD.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.988), frameon=False, fontsize=FS_ANNOT)

    sfE, sfF = sfRC.subfigures(2, 1, height_ratios=[1.22, 1.0], hspace=0.06)
    axE = draw_comparison(sfE, mets)
    axF = draw_confusion(sfF, predictions, true_means)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    invE = sfE.transSubfigure.inverted()
    dy = invE.transform((0, axD.get_window_extent(renderer=rend).y1))[1] - invE.transform(
        (0, axE.get_window_extent(renderer=rend).y1)
    )[1]
    pe = axE.get_position()
    axE.set_position([pe.x0, pe.y0 + dy, pe.width, pe.height])

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    invF = sfF.transSubfigure.inverted()
    ylb = lambda a: a.xaxis.label.get_window_extent(renderer=rend).y0
    dyF = invF.transform((0, ylb(axD)))[1] - invF.transform((0, ylb(axF)))[1]
    pf = axF.get_position()
    axF.set_position([pf.x0, pf.y0 + dyF, pf.width, pf.height])

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    eH = axH.get_window_extent(renderer=rend)
    eD2 = axD.get_window_extent(renderer=rend)
    eG = axG.get_window_extent(renderer=rend)
    name_w = eG.x0 - min(t.get_window_extent(renderer=rend).x0 for t in axG.get_yticklabels())
    left_edge = axD.yaxis.label.get_window_extent(renderer=rend).x0
    panel_w = eD2.x1 - (left_edge + name_w)

    invG = sfG.transSubfigure.inverted()
    p0 = invG.transform((left_edge + name_w, eH.y0))
    p1 = invG.transform((left_edge + name_w + panel_w, eH.y1))
    axG.set_position([p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1]])

    eE2 = axE.get_window_extent(renderer=rend)
    right = eE2.x0 + panel_w
    invH = sfH.transSubfigure.inverted()
    q0 = invH.transform((eE2.x0, eH.y0))
    q1 = invH.transform((right, eH.y1))
    axH.set_position([q0[0], q0[1], q1[0] - q0[0], q1[1] - q0[1]])

    invE2 = sfE.transSubfigure.inverted()
    pe2 = axE.get_position()
    axE.set_position([pe2.x0, pe2.y0, invE2.transform((right, 0))[0] - pe2.x0, pe2.height])

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    stats = [t for t in axF.texts if t.get_transform() is axF.transAxes and t.get_position()[0] > 1.0]
    if stats:
        shift = (right - max(t.get_window_extent(renderer=rend).x1 for t in stats)) / axF.get_window_extent(renderer=rend).width
        for t in stats:
            x, y = t.get_position()
            t.set_position((x + shift, y))

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    lx = lambda a: inv.transform((a.yaxis.label.get_window_extent(renderer=rend).x0, 0))[0]
    wfig = lambda a: inv.transform((a.get_window_extent(renderer=rend).x1, 0))[0] - inv.transform(
        (a.get_window_extent(renderer=rend).x0, 0)
    )[0]
    ref_x = -0.30
    for a in (axE, axF):
        a.yaxis.set_label_coords(ref_x, 0.5)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ticks_l = min(t.get_window_extent(renderer=rend).x0 for a in (axE, axF) for t in a.get_yticklabels())
    target = inv.transform((ticks_l - 0.24 * fig.dpi, 0))[0]
    for a in (axE, axF):
        a.yaxis.set_label_coords(ref_x + (target - lx(a)) / wfig(a), 0.5)

    out = os.path.join(figures_dir, "fig4.png")
    if os.environ.get("NO_HEADER"):
        fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
        print(f"Saved {out}")
    else:
        from plotting.svg_composer import compose_fig4
        compose_fig4(fig, axG, axH, axD, axE, axF, out)
    plt.close(fig)

    n_ap = sum(1 for p, t in true_means.items() if p.startswith("ST") and t < apraxia_cutoff)
    print(
        f"  Full cohort (N={len(true_means)}): "
        f"{sum(p.startswith('CL') for p in true_means)} control, "
        f"{sum(p.startswith('ST') for p in true_means)} stroke ({n_ap} with apraxia)"
    )
    for k, (r, mae, auc) in mets.items():
        print(f"    {k:8s} r={r:.3f}  MAE={mae:.3f}  AUC={auc:.3f}")


if __name__ == "__main__":
    main()
