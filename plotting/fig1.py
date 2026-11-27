"""
Figure 1 — Task overview, manual coding scheme, and descriptive praxis scores
across all 53 participants (10 healthy controls, 26 stroke without apraxia,
17 stroke with apraxia; content = 1 trials).

Panels:
  A-B  Task illustration and manual scoring hierarchy (vector header template)
  C    Overall participant praxis scores (0-3) by group with 2.52 apraxia cutoff
  D    Mean component scores (Hand Posture, Arm Posture & Trajectory, Amplitude & Timing)
  E    Row-normalized item score distributions for stroke groups without vs. with apraxia
  F    Item-level ROC AUC for distinguishing stroke patients with vs. without apraxia

Outputs:
  figures/figure1.svg
  figures/fig1.png
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

from model import APRAXIA_CUTOFF, get_item_name, parse_component_scores

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "modelinputs")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CMAP_VMIN = 0.01

FS_AXIS_LABEL = 11
FS_TICK = 10
FS_ITEM = 8.5
FS_ANNOT = 8.5
FS_HEADER = 12
FS_CBAR = 8

C_CONTROL = "#0A1F4D"
DARK_CONTROL = "#040E26"
C_NONAPRAXIC = "#1976D2"
DARK_BLUE = "#0A2E5C"
C_APRAXIC = "#D32F2F"
DARK_RED = "#7F0000"
BAR_COLOUR = "#A399C6"


def load_participant_scores():
    with open(os.path.join(DATA_DIR, "metadata.json"), "r") as f:
        md = json.load(f)
    by_pid = defaultdict(lambda: {"hp": [], "ap": [], "at": [], "total": []})
    item_scores = defaultdict(lambda: defaultdict(list))
    for e in md:
        c, hp, ap, at, total = parse_component_scores(e["score"])
        if c != 1 or total is None:
            continue
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (hp, ap, at)):
            continue
        p = e["participant"]
        by_pid[p]["hp"].append(hp)
        by_pid[p]["ap"].append(ap)
        by_pid[p]["at"].append(at)
        by_pid[p]["total"].append(total)
        item_scores[e["item"]][p].append(total)

    rows = {}
    for p, d in by_pid.items():
        total_mean = float(np.mean(d["total"]))
        is_ctrl = p.startswith("CL")
        rows[p] = {
            "hp": float(np.mean(d["hp"])),
            "ap": float(np.mean(d["ap"])),
            "at": float(np.mean(d["at"])),
            "total": total_mean,
            "n": len(d["total"]),
            "is_control": is_ctrl,
            "group": "control" if is_ctrl else ("apraxic" if total_mean < APRAXIA_CUTOFF else "nonaprax"),
        }
    return rows, item_scores


def panel_a(ax, rows):
    groups = ["control", "nonaprax", "apraxic"]
    gcolor = {"control": C_CONTROL, "nonaprax": C_NONAPRAXIC, "apraxic": C_APRAXIC}
    gdark = {"control": DARK_CONTROL, "nonaprax": DARK_BLUE, "apraxic": DARK_RED}
    glabel = {
        "control": "Control",
        "nonaprax": "Stroke:\nwithout apraxia",
        "apraxic": "Stroke:\nwith apraxia",
    }
    positions = {"control": 1, "nonaprax": 2, "apraxic": 3}
    data = {g: [r["total"] for r in rows.values() if r["group"] == g] for g in groups}
    rng = np.random.default_rng(1)
    for g in groups:
        vals = np.array(data[g])
        xc = positions[g]
        xs = xc + rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(xs, vals, c=gcolor[g], edgecolors="none", marker="o", s=40, alpha=0.6, zorder=3)
        ax.hlines(vals.mean(), xc - 0.22, xc + 0.22, color=gdark[g], lw=2.8, zorder=5)
    ax.axhline(APRAXIA_CUTOFF, color="#000", ls="--", lw=1.2, alpha=0.65, zorder=2)
    ax.text(
        0.55,
        APRAXIA_CUTOFF - 0.02,
        f"Apraxia cut-off ({APRAXIA_CUTOFF})",
        fontsize=FS_ANNOT,
        ha="left",
        va="top",
        color="#000",
        style="italic",
    )
    counts = {g: len(data[g]) for g in groups}
    ax.set_xticks([positions[g] for g in groups])
    ax.set_xticklabels([f"{glabel[g]}\n(n={counts[g]})" for g in groups], fontsize=FS_TICK)
    ax.set_xlabel("Group", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.set_ylabel("Praxis Score (0-3)", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.set_ylim(1.5, 3.08)
    ax.set_xlim(0.4, 3.6)
    ax.tick_params(labelsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
        sp.set_color("#333")


def panel_b(ax, rows):
    groups = ["control", "nonaprax", "apraxic"]
    gcolor = {"control": C_CONTROL, "nonaprax": C_NONAPRAXIC, "apraxic": C_APRAXIC}
    gdark = {"control": DARK_CONTROL, "nonaprax": DARK_BLUE, "apraxic": DARK_RED}
    glabel = {
        "control": "Control",
        "nonaprax": "Stroke: without apraxia",
        "apraxic": "Stroke: with apraxia",
    }
    comps = ["hp", "ap", "at"]
    comp_label = {
        "hp": "Hand posture",
        "ap": "Arm posture\n& trajectory",
        "at": "Amplitude\n& timing",
    }
    comp_x = {c: i for i, c in enumerate(comps)}
    offset = {"control": -0.26, "nonaprax": 0.0, "apraxic": 0.26}
    rng = np.random.default_rng(3)
    for comp in comps:
        for g in groups:
            members = [r for r in rows.values() if r["group"] == g]
            vals = np.array([m[comp] for m in members])
            xc = comp_x[comp] + offset[g]
            xs = xc + rng.uniform(-0.07, 0.07, size=len(vals))
            ax.scatter(xs, vals, c=gcolor[g], edgecolors="none", marker="o", s=32, alpha=0.5, zorder=3)
            ax.hlines(vals.mean(), xc - 0.13, xc + 0.13, color=gdark[g], lw=2.6, zorder=5)
    ax.set_xticks([comp_x[c] for c in comps])
    ax.set_xticklabels([comp_label[c] for c in comps], fontsize=FS_TICK)
    ax.set_xlabel("Component", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.set_ylabel("Mean Score (0-1)", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.55, 2.55)
    ax.tick_params(labelsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
        sp.set_color("#333")
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=gcolor[g],
            markeredgecolor=gdark[g],
            markersize=9,
            label=glabel[g],
        )
        for g in groups
    ]
    ax.legend(
        handles=legend,
        fontsize=FS_ANNOT,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        ncol=1,
        frameon=False,
        handletextpad=0.5,
        labelspacing=0.5,
        alignment="left",
    )


def compute_item_aucs(rows, item_scores):
    patients = [p for p, r in rows.items() if not r["is_control"]]
    part_apraxia = {p: (1 if rows[p]["group"] == "apraxic" else 0) for p in patients}
    results = []
    for item_id, pid_scores in item_scores.items():
        scores, labels = [], []
        for p in patients:
            if p in pid_scores and len(pid_scores[p]) > 0:
                scores.append(np.mean(pid_scores[p]))
                labels.append(part_apraxia[p])
        if len(scores) < 10:
            continue
        scores = np.array(scores)
        labels = np.array(labels)
        auc = float(roc_auc_score(labels, -scores)) if len(np.unique(labels)) > 1 else 0.5
        results.append({"id": item_id, "name": get_item_name(item_id), "auc": auc})
    results.sort(key=lambda r: r["auc"], reverse=True)
    return results


def panel_c(ax_non, ax_apr, rows, item_scores, ordered_results):
    items = [r["id"] for r in ordered_results]
    names = [r["name"] for r in ordered_results]
    non_pids = [p for p, r in rows.items() if r["group"] == "nonaprax"]
    apr_pids = [p for p, r in rows.items() if r["group"] == "apraxic"]

    def build(pids):
        M = np.zeros((len(items), 4))
        for i, it in enumerate(items):
            ps = item_scores.get(it, {})
            counts = np.zeros(4)
            n_part = 0
            for p in pids:
                trials = ps.get(p, [])
                if len(trials) == 0:
                    continue
                lvl = int(round(np.mean(trials)))
                if 0 <= lvl <= 3:
                    counts[lvl] += 1
                    n_part += 1
            if n_part > 0:
                M[i] = counts / n_part
        return M

    M_non = build(non_pids)
    M_apr = build(apr_pids)
    vmax = 1.0
    n = len(items)
    grey0 = "#e8e8e8"
    blue_cmap = LinearSegmentedColormap.from_list("nonaprax", ["#ffffff", C_NONAPRAXIC])
    red_cmap = LinearSegmentedColormap.from_list("aprax", ["#ffffff", C_APRAXIC])
    blue_cmap.set_bad(grey0)
    red_cmap.set_bad(grey0)
    xedges = np.arange(5) - 0.5
    yedges = np.arange(n + 1) - 0.5
    for ax, M, cmap in [(ax_non, M_non, blue_cmap), (ax_apr, M_apr, red_cmap)]:
        Mm = np.ma.masked_where(M == 0, M)
        ax.pcolormesh(
            xedges,
            yedges,
            Mm,
            cmap=cmap,
            vmin=CMAP_VMIN,
            vmax=vmax,
            edgecolors="white",
            linewidth=1.2,
        )
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["0", "1", "2", "3"], fontsize=FS_TICK)
        ax.set_xlabel("Score", fontsize=FS_AXIS_LABEL, fontweight="bold")
        ax.tick_params(length=0, labelsize=FS_TICK)
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax_non.set_yticks(range(n))
    ax_non.set_yticklabels(names, fontsize=FS_ITEM)
    ax_apr.set_yticks(range(n))
    ax_apr.set_yticklabels([])
    return vmax, blue_cmap, red_cmap, grey0


def panel_d(ax, ordered_results):
    names = [r["name"] for r in ordered_results]
    aucs = [r["auc"] for r in ordered_results]
    n = len(aucs)
    y = np.arange(n)
    ax.barh(y, aucs, color=BAR_COLOUR, edgecolor="none", height=0.72, zorder=3)
    ax.axvline(0.5, color="#000", ls="--", lw=1.5, alpha=0.9, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=FS_ITEM)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel("AUC", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
        sp.set_color("#333")


def main():
    rows, item_scores = load_participant_scores()
    n_ctrl = sum(1 for r in rows.values() if r["group"] == "control")
    n_non = sum(1 for r in rows.values() if r["group"] == "nonaprax")
    n_apr = sum(1 for r in rows.values() if r["group"] == "apraxic")
    print(f"Participants: {len(rows)}  (control={n_ctrl}, without apraxia={n_non}, with apraxia={n_apr})")

    ordered_results = compute_item_aucs(rows, item_scores)

    fig_w, fig_h = 18, 11
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        2, 3, width_ratios=[1.0, 0.7, 0.68], height_ratios=[1, 1], wspace=0.30, hspace=0.42
    )
    axA = fig.add_subplot(outer[0, 0])
    axB = fig.add_subplot(outer[1, 0])
    axD = fig.add_subplot(outer[:, 2])

    panel_a(axA, rows)
    panel_b(axB, rows)
    panel_d(axD, ordered_results)

    n = len(ordered_results)
    posD = axD.get_position()
    posA = axA.get_position()
    y0, height = posD.y0, (posD.y1 - posD.y0)
    tile_y = height / n
    tile_x = tile_y * (fig_h / fig_w)
    hw = 4 * tile_x
    gap = 0.042
    total = 2 * hw + gap
    mid = (posA.x1 + posD.x0) / 2
    x0 = mid - total / 2
    x_apr = x0 + hw + gap
    axC1 = fig.add_axes([x0, y0, hw, height])
    axC2 = fig.add_axes([x_apr, y0, hw, height])
    vmax, blue_cmap, red_cmap, grey0 = panel_c(axC1, axC2, rows, item_scores, ordered_results)

    top_y = y0 + height
    fig.text(
        x0 + hw / 2,
        top_y + 0.003,
        "Without\napraxia",
        ha="center",
        va="bottom",
        fontsize=FS_HEADER - 2.6,
        fontweight="bold",
        linespacing=0.92,
    )
    fig.text(
        x_apr + hw / 2,
        top_y + 0.003,
        "With\napraxia",
        ha="center",
        va="bottom",
        fontsize=FS_HEADER - 2.6,
        fontweight="bold",
        linespacing=0.92,
    )

    cbar_y = y0 - 0.057
    cax1 = fig.add_axes([x0, cbar_y, hw, 0.012])
    cax2 = fig.add_axes([x_apr, cbar_y, hw, 0.012])
    cb1 = fig.colorbar(ScalarMappable(norm=Normalize(CMAP_VMIN, vmax), cmap=blue_cmap), cax=cax1, orientation="horizontal")
    cb2 = fig.colorbar(ScalarMappable(norm=Normalize(CMAP_VMIN, vmax), cmap=red_cmap), cax=cax2, orientation="horizontal")
    for cb in (cb1, cb2):
        cb.set_ticks([CMAP_VMIN, 0.5, 1.0])
        cb.set_ticklabels(["0.01", "0.5", "1"])
        cb.ax.tick_params(labelsize=FS_CBAR)
        cb.outline.set_visible(False)

    # Native crisp '0' grey swatches beside both colorbars (matching colorbar height & tick typography)
    sw_w = 0.012 * (fig_h / fig_w)
    for x_left in (x0, x_apr):
        gax = fig.add_axes([x_left - sw_w - 0.009, cbar_y, sw_w, 0.012])
        gax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=grey0, edgecolor="none"))
        gax.set_xlim(0, 1)
        gax.set_ylim(0, 1)
        gax.set_xticks([0.5])
        gax.set_xticklabels(["0"])
        gax.set_yticks([])
        gax.tick_params(axis="x", labelsize=FS_CBAR)
        for sp in gax.spines.values():
            sp.set_visible(False)

    fig.text(mid, cbar_y - 0.032, "Proportion of Participants", ha="center", va="top", fontsize=FS_AXIS_LABEL, fontweight="bold")

    # Align Panel C's y-label left edge with Panel D's y-label left edge
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    lx = lambda ax: inv.transform((ax.yaxis.label.get_window_extent(renderer=rend).x0, 0))[0]
    target = lx(axB)
    ref_x = -0.16
    axA.yaxis.set_label_coords(ref_x, 0.5)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axA.yaxis.set_label_coords(ref_x + (target - lx(axA)) / axA.get_position().width, 0.5)

    out = os.path.join(FIGURES_DIR, "fig1.png")
    if os.environ.get("NO_HEADER"):
        fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
        print(f"Saved {out} (600 dpi)")
    else:
        from plotting.svg_composer import compose_fig1
        compose_fig1(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
