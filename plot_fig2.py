"""
Figure 2 — Model architecture, sample kinematics, cross-validated feature and
item weights, and training-set LOPO cross-validation performance (n = 41:
7 healthy controls, 20 stroke without apraxia, 14 stroke with apraxia).

Panels:
  A  Model development and LOPO cross-validation pipeline (vector header)
  B  Sample 3D wrist trajectories and kinematic profiles (Item: Corkscrew)
  C  Participant-level Ridge |coefficient| (mean and SD inputs) across 41 folds
  D  Item-level reliability weights across 41 folds
  E  Cross-validated true vs. predicted praxis scores (n = 41)
  F  Pathway comparison (Participant, Item, Blended: Pearson r, AUC, MAE)
  G  Stroke apraxia confusion matrix, sensitivity, and specificity

Outputs:
  figures/figure2.svg
  figures/fig2.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from item_info import ITEM_INFO
from model import APRAXIA_CUTOFF, FEATURE_NAMES, BlendModel, load_train_data

plt.rcParams["font.family"] = "DejaVu Sans"

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_dir, "modelinputs")
figures_dir = os.path.join(project_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)
lopo_cache_path = os.path.join(figures_dir, "fig2_lopo_cache.json")

apraxia_cutoff = APRAXIA_CUTOFF
loio_color = "#A399C6"
cmat_color = "#2E7D32"
name_color = "black"

FS_AXIS_LABEL = 12.0
FS_TICK = 10.8
FS_ITEM = 10.2
FS_FEAT = 9.6
FS_ANNOT = 10.0
FS_HEADER = 12.8
FS_QUADRANT = 12.8

c_control = "#0A1F4D"
dark_control = "#040E26"
c_nonaprax = "#1976D2"
dark_blue = "#0A2E5C"
c_aprax = "#D32F2F"
dark_red = "#7F0000"
DEFAULT_AX_LIM = (1.7, 3.1)

# Marker column slices in time-stripped 100 Hz trial array
thumb_pos = slice(0, 3)
index_pos = slice(3, 6)
wrist_pos = slice(12, 15)
elbow_pos = slice(15, 18)
lshoulder_pos = slice(18, 21)
wrist_vel = slice(36, 39)

# Representative Corkscrew trials: (label, color, participant, trial, manual_score)
trials = [
    ("Control", c_control, "CL030", "trial253", 3),
    ("Without Apraxia", c_nonaprax, "ST040", "trial744", 3),
    ("With Apraxia", c_aprax, "ST057", "trial1763", 1),
]

groups = [
    ("Hand posture", 0, 19, "#882255"),
    ("Arm posture", 19, 32, "#E69F00"),
    ("Amplitude & timing", 32, 45, "#44AA99"),
    ("Motion type", 45, 46, "#7A7A84"),
]
full_names = list(FEATURE_NAMES)
full_names[45] = "Motion-specific feature"

item_names = {k: v["name"] for k, v in ITEM_INFO.items()}


def load_trial(trial_name):
    d = np.load(os.path.join(data_dir, f"{trial_name}.npy")).astype(np.float64)
    t = d[:, 0] - d[0, 0]
    return t, d[:, 1:]


def compute(t, raw):
    wpos = raw[:, wrist_pos]
    speed = np.linalg.norm(raw[:, wrist_vel], axis=1) * 100.0
    aperture = np.linalg.norm(raw[:, thumb_pos] - raw[:, index_pos], axis=1) * 100.0
    v1 = raw[:, lshoulder_pos] - raw[:, elbow_pos]
    v2 = raw[:, wrist_pos] - raw[:, elbow_pos]
    cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    elbow_ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    tnorm = 100.0 * (t - t[0]) / (t[-1] - t[0])
    return dict(
        wpos=wpos,
        speed=speed,
        aperture=aperture,
        elbow_ang=elbow_ang,
        tnorm=tnorm,
        dur=float(t[-1] - t[0]),
    )


def gradient_line_3d(ax, pts, base_color, lw=2.8):
    base = np.array(to_rgb(base_color))
    light = base + (1.0 - base) * 0.72
    cmap = LinearSegmentedColormap.from_list("t", [light, base])
    p = pts.reshape(-1, 1, 3)
    segs = np.concatenate([p[:-1], p[1:]], axis=1)
    tt = np.linspace(0, 1, len(pts))
    lc = Line3DCollection(segs, cmap=cmap, norm=Normalize(0, 1), linewidths=lw)
    lc.set_array(0.5 * (tt[:-1] + tt[1:]))
    ax.add_collection3d(lc)
    return lc


def draw_scatter(ax, predictions, true_means, lim=DEFAULT_AX_LIM):
    pids = sorted(predictions)
    pred = np.array([predictions[p] for p in pids])
    true = np.array([true_means[p] for p in pids])
    is_ctrl = np.array([p.startswith("CL") for p in pids])
    r = pearsonr(pred, true)[0]
    rho = spearmanr(pred, true)[0]
    mae = float(np.mean(np.abs(pred - true)))
    mis = np.array(
        [(not is_ctrl[i]) and ((true[i] < apraxia_cutoff) != (pred[i] < apraxia_cutoff)) for i in range(len(pids))]
    )
    mn, mx = lim
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
    for i in range(len(pids)):
        if is_ctrl[i]:
            color, marker, size = c_control, "o", 85
        else:
            color = c_aprax if true[i] < apraxia_cutoff else c_nonaprax
            marker = "X" if mis[i] else "o"
            size = 125 if mis[i] else 85
        ax.scatter(
            true[i],
            pred[i],
            c=color,
            edgecolors="black",
            s=size,
            marker=marker,
            linewidths=1.1,
            alpha=0.85,
            zorder=4,
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
    return r, rho, mae


def compute_or_load_lopo():
    """
    Run (or load cached) 41-fold LOPO cross-validation on the training cohort,
    returning pathway predictions, true scores, and fold-averaged feature/item weights.
    """
    if os.path.exists(lopo_cache_path):
        with open(lopo_cache_path, "r") as f:
            d = json.load(f)
        return {
            "direct": d["direct"],
            "loio": d["loio"],
            "blend": d["blend"],
            "true": d["true"],
            "mean_mag": np.array(d["mean_mag"]),
            "mean_mag_sd": np.array(d["mean_mag_sd"]),
            "std_mag": np.array(d["std_mag"]),
            "std_mag_sd": np.array(d["std_mag_sd"]),
            "item_weights": {int(k): v for k, v in d["item_weights"].items()},
            "item_weights_sd": {int(k): v for k, v in d["item_weights_sd"].items()},
        }

    print("Running 41-fold training LOPO cross-validation...")
    x, sc, items, parts = load_train_data(data_dir, downsample=15)
    y = np.array([t if t is not None else np.nan for t in sc["total"]])
    m = ~np.isnan(y)
    x, y, items = x[m], y[m], items[m]
    parts = np.array([p for p, k in zip(parts, m) if k])

    pids = sorted(set(parts))
    item_ids = sorted(set(int(i) for i in items))
    nf = x.shape[1]
    part_true = {p: float(y[parts == p].mean()) for p in pids}

    direct, loio, blend = {}, {}, {}
    mean_mags, std_mags, iws = [], [], []
    for pid in pids:
        tr = parts != pid
        ho = parts == pid
        mdl = BlendModel(alpha_item=10, alpha_direct=10, blend_weight=0.5, calibrate=True)
        mdl.fit(x[tr], y[tr], items[tr], list(parts[tr]))

        d_pred, l_pred, b_pred = mdl.predict_pathways(x[ho], items[ho], list(parts[ho]))
        direct[pid] = d_pred[pid]
        loio[pid] = l_pred[pid]
        blend[pid] = b_pred[pid]

        c = np.abs(mdl.direct_model.coef_)
        mean_mags.append(c[:nf])
        std_mags.append(c[nf:])
        iws.append([mdl.item_weights.get(i, np.nan) for i in item_ids])

    mean_mags = np.array(mean_mags)
    std_mags = np.array(std_mags)
    iws = np.array(iws)

    mean_mag = mean_mags.mean(0)
    mean_mag_sd = mean_mags.std(0, ddof=1)
    std_mag = std_mags.mean(0)
    std_mag_sd = std_mags.std(0, ddof=1)
    item_w = {int(i): float(v) for i, v in zip(item_ids, np.nanmean(iws, 0))}
    item_w_sd = {int(i): float(v) for i, v in zip(item_ids, np.nanstd(iws, 0, ddof=1))}

    payload = {
        "n_folds": len(pids),
        "direct": direct,
        "loio": loio,
        "blend": blend,
        "true": part_true,
        "mean_mag": mean_mag.tolist(),
        "mean_mag_sd": mean_mag_sd.tolist(),
        "std_mag": std_mag.tolist(),
        "std_mag_sd": std_mag_sd.tolist(),
        "item_weights": {str(k): v for k, v in item_w.items()},
        "item_weights_sd": {str(k): v for k, v in item_w_sd.items()},
    }
    with open(lopo_cache_path, "w") as f:
        json.dump(payload, f)

    return {
        "direct": direct,
        "loio": loio,
        "blend": blend,
        "true": part_true,
        "mean_mag": mean_mag,
        "mean_mag_sd": mean_mag_sd,
        "std_mag": std_mag,
        "std_mag_sd": std_mag_sd,
        "item_weights": item_w,
        "item_weights_sd": item_w_sd,
    }


def model_metrics(pred, true):
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


def draw_traces_portrait(sf):
    data = []
    for label, colr, pid, tn, score in trials:
        t, raw = load_trial(tn)
        f = compute(t, raw)
        f.update(label=label, col=colr, pid=pid, score=score)
        data.append(f)
    for f in data:
        f["w"] = (f["wpos"] - f["wpos"].mean(axis=0, keepdims=True)) * 100.0
    cube = max((f["w"].max(axis=0) - f["w"].min(axis=0)).max() for f in data)
    smax = max(f["speed"].max() for f in data)
    amin = min(f["aperture"].min() for f in data)
    amax = max(f["aperture"].max() for f in data)
    emin = min(f["elbow_ang"].min() for f in data)
    emax = max(f["elbow_ang"].max() for f in data)
    apad, epad = 0.06 * (amax - amin), 0.06 * (emax - emin)
    lblkw = dict(fontsize=FS_AXIS_LABEL, fontweight="bold")
    label_x = -0.30
    measures = [
        ("aperture", (amin - apad, amax + apad), "Thumb-index\naperture (cm)"),
        ("elbow_ang", (emin - epad, emax + epad), "Elbow angle (°)"),
        ("speed", (0, smax * 1.05), "Wrist speed (cm/s)"),
    ]

    gs = sf.add_gridspec(
        4, 3, left=0.16, right=0.90, top=0.92, bottom=0.047, height_ratios=[0.8, 1.0, 1.0, 1.0], wspace=0.22, hspace=0.28
    )
    ylabels = []
    for j, f in enumerate(data):
        colr = f["col"]
        first = j == 0
        ax3 = sf.add_subplot(gs[0, j], projection="3d")
        gradient_line_3d(ax3, f["w"], colr)
        ax3.scatter(*f["w"][0], facecolor="white", edgecolor=colr, s=26, linewidth=1.2, zorder=6)
        ax3.set_xlim(-cube / 2, cube / 2)
        ax3.set_ylim(-cube / 2, cube / 2)
        ax3.set_zlim(-cube / 2, cube / 2)
        ax3.view_init(elev=18, azim=-62)
        ax3.grid(False)
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis.set_pane_color((1, 1, 1, 0))
            axis.pane.set_edgecolor("none")
            axis.set_major_locator(MaxNLocator(nbins=3))
        ax3.tick_params(labelsize=FS_TICK)
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_zticklabels([])
        try:
            ax3.set_box_aspect(None, zoom=1.3)
        except TypeError:
            pass
        p = ax3.get_position()
        ax3.set_position([p.x0, p.y0 - 0.022, p.width, p.height])
        ax3.set_title(
            f"{f['label']}\n({f['score']}/3, {f['dur']:.1f} s)",
            color=colr,
            fontweight="bold",
            fontsize=FS_HEADER,
            pad=6,
        )
        if first:
            t3 = ax3.text2D(
                label_x,
                0.5,
                "3D Wrist\nPosition (cm)",
                transform=ax3.transAxes,
                rotation=90,
                va="center",
                ha="center",
                ma="center",
                **lblkw,
            )
            ylabels.append(("3d", ax3, t3))
        for r, (key, ylim, ylab) in enumerate(measures, start=1):
            ax = sf.add_subplot(gs[r, j])
            ax.plot(f["tnorm"], f[key], color=colr, lw=2.4, alpha=0.75)
            ax.set_xlim(-6, 100)
            ax.set_ylim(*ylim)
            ax.set_xticks([])
            for sp in ("top", "right", "bottom"):
                ax.spines[sp].set_visible(False)
            if first:
                ax.spines["left"].set_visible(True)
                ax.tick_params(axis="y", labelsize=FS_TICK)
                ax.set_ylabel(ylab, **lblkw)
                ax.yaxis.set_label_coords(label_x, 0.5)
                ylabels.append(("2d", ax, ax.yaxis.label))
            else:
                ax.spines["left"].set_visible(False)
                ax.set_yticks([])
    return ylabels


def draw_coeff(sf, mean_mag, std_mag, name_fs=FS_FEAT, width_ratios=(1.02, 2.73, 1.02)):
    n = len(full_names)
    bar_colors = np.empty(n, dtype=object)
    for _, lo, hi, c in groups:
        for i in range(lo, hi):
            bar_colors[i] = c
    xmax = 1.08 * max(mean_mag.max(), std_mag.max())
    tick_step = 0.01 if xmax <= 0.032 else (0.02 if xmax <= 0.065 else 0.04)
    y = np.arange(n)
    a = 0.8
    gs = sf.add_gridspec(1, 3, width_ratios=list(width_ratios), wspace=0.0, left=0.02, right=0.99, top=0.88, bottom=0.09)
    axl = sf.add_subplot(gs[0, 0])
    axm = sf.add_subplot(gs[0, 1])
    axr = sf.add_subplot(gs[0, 2])
    axl.barh(y, mean_mag, color=bar_colors, edgecolor="white", linewidth=0.5, alpha=a)
    axl.set_xlim(xmax, 0)
    axr.barh(y, std_mag, color=bar_colors, edgecolor="white", linewidth=0.5, alpha=a)
    axr.set_xlim(0, xmax)
    zero_fmt = matplotlib.ticker.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:.2f}")
    for ax in (axl, axr):
        ax.set_ylim(-0.6, n - 0.4)
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.tick_params(labelsize=9.0)
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.xaxis.set_major_formatter(zero_fmt)
        for sp in ("top", "left" if ax is axl else "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["right" if ax is axl else "left"].set_visible(False)
        ax.set_xlabel("|coefficient|", fontsize=FS_AXIS_LABEL, fontweight="bold", labelpad=9)
    axl.text(0.5, 1.01, "Mean", transform=axl.transAxes, ha="center", va="bottom", fontsize=FS_HEADER, fontweight="bold", color="#333")
    axr.text(0.5, 1.01, "SD", transform=axr.transAxes, ha="center", va="bottom", fontsize=FS_HEADER, fontweight="bold", color="#333")
    axm.set_xlim(0, 1)
    axm.set_ylim(-0.6, n - 0.4)
    axm.invert_yaxis()
    axm.axis("off")
    for i, nm in enumerate(full_names):
        axm.text(0.5, i, nm, ha="center", va="center", fontsize=name_fs, color=name_color)
    return axl, axm, axr


def draw_loio(ax, weights):
    item_ids = sorted(weights, key=lambda k: weights[k])
    w = np.array([weights[i] for i in item_ids])
    names = [item_names.get(int(i), f"Item {i}") for i in item_ids]
    n = len(item_ids)
    yy = np.arange(n)
    ax.barh(yy, w, color=loio_color, edgecolor="white", linewidth=0.5)
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=FS_ITEM, color=name_color)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, max(0.06, 1.08 * float(w.max())))
    ax.set_xlabel("Weight", fontsize=FS_AXIS_LABEL, fontweight="bold", labelpad=9)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.0)
        sp.set_color("#333")


def draw_comparison(sf, mets):
    models = ["Direct", "LOIO", "Blend"]
    r = [mets[m][0] for m in models]
    auc = [mets[m][2] for m in models]
    mae = [mets[m][1] for m in models]
    ax = sf.add_axes([0.16, 0.315, 0.80, 0.5004])
    xpos = np.arange(3)
    wbar = 0.38
    b1 = ax.bar(xpos - wbar / 2, r, wbar, color="#4d4d4d", label="$r$")
    b2 = ax.bar(xpos + wbar / 2, auc, wbar, color="#bdbdbd", label="AUC")
    for bars, vals in ((b1, r), (b2, auc)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=FS_ANNOT)
    display = ["Participant", "Item", "Blended"]
    ax.set_xticks(xpos)
    ax.set_xticklabels(display, fontsize=FS_TICK, fontweight="bold")
    for xp, e in zip(xpos, mae):
        ax.text(xp, -0.24, f"MAE {e:.3f}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS_TICK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Performance", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax


def draw_confusion(sf, predictions, true_means):
    pats = [p for p in sorted(predictions) if p.startswith("ST")]
    t = np.array([true_means[p] < apraxia_cutoff for p in pats])
    q = np.array([predictions[p] < apraxia_cutoff for p in pats])
    cm = np.array([[int((t & ~q).sum()), int((~t & ~q).sum())], [int((t & q).sum()), int((~t & q).sum())]])
    fn, tn = cm[0, 0], cm[0, 1]
    tp, fp = cm[1, 0], cm[1, 1]
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    ax = sf.add_axes([0.16, 0.139, 0.50, 0.6303])
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
    for y, lab, val in ((0.60, "Sensitivity:", sens), (0.42, "Specificity:", spec)):
        ax.text(1.78, y, lab, transform=ax.transAxes, ha="right", va="center", fontsize=FS_HEADER, fontweight="bold", color="#333")
        ax.text(1.83, y, f"{val:.2f}", transform=ax.transAxes, ha="left", va="center", fontsize=FS_HEADER, color="#333")
    return ax


def main():
    res = compute_or_load_lopo()
    predictions = res["blend"]
    true_means = res["true"]
    mean_mag = res["mean_mag"]
    std_mag = res["std_mag"]
    item_weights = res["item_weights"]
    direct = res["direct"]
    loio = res["loio"]

    fig = plt.figure(figsize=(17.3, 15.6))
    sfL, sfR = fig.subfigures(1, 2, width_ratios=[0.82, 1.18], wspace=0.02)
    trace_labels = draw_traces_portrait(sfL)

    sfR_top, sfR_bot = sfR.subfigures(2, 1, height_ratios=[1.68, 1.0], hspace=0.11)
    sfB, sfC = sfR_top.subfigures(1, 2, width_ratios=[1.33, 0.79], wspace=0.13)
    axl, axm, axr = draw_coeff(sfB, mean_mag, std_mag)
    axC = sfC.add_axes([0.42, 0.0475, 0.56, 0.87])
    draw_loio(axC, item_weights)

    sfD, sfRC = sfR_bot.subfigures(1, 2, width_ratios=[1.52, 0.88], wspace=0.13)
    axD = sfD.add_axes([0.085, 0.0926, 0.85, 0.8795])
    draw_scatter(axD, predictions, true_means, lim=DEFAULT_AX_LIM)
    axD.set_anchor("W")

    axD.text(1.74, 1.74, "With\nApraxia", fontsize=FS_QUADRANT, color=dark_red, ha="left", va="bottom", fontstyle="italic", fontweight="bold")
    axD.text(3.07, 3.07, "Without\nApraxia", fontsize=FS_QUADRANT, color=dark_blue, ha="right", va="top", fontstyle="italic", fontweight="bold")
    mec = "black"
    sc_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_control, markeredgecolor=mec, markersize=9, label="Control"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_nonaprax, markeredgecolor=mec, markersize=9, label="Stroke: without apraxia"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_aprax, markeredgecolor=mec, markersize=9, label="Stroke: with apraxia"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#888", markeredgecolor=mec, markersize=10, label="Misclassified"),
    ]
    axD.legend(handles=sc_handles, loc="upper left", bbox_to_anchor=(0.012, 0.988), frameon=False, fontsize=FS_ANNOT)

    mets = {
        "Direct": model_metrics(direct, true_means),
        "LOIO": model_metrics(loio, true_means),
        "Blend": model_metrics(predictions, true_means),
    }
    sfE, sfF = sfRC.subfigures(2, 1, height_ratios=[1.22, 1.0], hspace=0.06)
    axE = draw_comparison(sfE, mets)
    axF = draw_confusion(sfF, predictions, true_means)

    # Cross-subfigure display-space alignment pass
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    invE = sfE.transSubfigure.inverted()
    dy = invE.transform((0, axD.get_window_extent(renderer=rend).y1))[1] - invE.transform(
        (0, axE.get_window_extent(renderer=rend).y1)
    )[1]
    pe = axE.get_position()
    axE.set_position([pe.x0, pe.y0 + dy, pe.width, pe.height])

    ref_x = -0.32
    for a in (axE, axF):
        a.yaxis.set_label_coords(ref_x, 0.5)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    lx = lambda a: inv.transform((a.yaxis.label.get_window_extent(renderer=rend).x0, 0))[0]
    lw_fig = lambda a: inv.transform((a.yaxis.label.get_window_extent(renderer=rend).x1, 0))[0] - lx(a)
    tick_min = lambda a: min(
        inv.transform((t.get_window_extent(renderer=rend).x0, 0))[0] for t in a.get_yticklabels() if t.get_text()
    )
    target = min(
        min(inv.transform((t.get_window_extent(renderer=rend).x0, 0))[0] for t in axC.get_yticklabels()),
        tick_min(axE) - lw_fig(axE) - 0.005,
        tick_min(axF) - lw_fig(axF) - 0.005,
    )
    wfig = lambda a: inv.transform((a.get_window_extent(renderer=rend).x1, 0))[0] - inv.transform(
        (a.get_window_extent(renderer=rend).x0, 0)
    )[0]
    for a in (axE, axF):
        a.yaxis.set_label_coords(ref_x + (target - lx(a)) / wfig(a), 0.5)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    invF = sfF.transSubfigure.inverted()
    ylb = lambda a: a.xaxis.label.get_window_extent(renderer=rend).y0
    dyF = invF.transform((0, ylb(axD)))[1] - invF.transform((0, ylb(axF)))[1]
    pf = axF.get_position()
    axF.set_position([pf.x0, pf.y0 + dyF, pf.width, pf.height])

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    invB = sfB.transSubfigure.inverted()
    eD = axD.get_window_extent(renderer=rend)
    xl_edge, xr_edge = eD.x0, eD.x1
    eC = axC.get_window_extent(renderer=rend)
    exts = [a.get_window_extent(renderer=rend) for a in (axl, axm, axr)]
    curL, curR = exts[0].x0, exts[2].x1
    scale = (xr_edge - xl_edge) / (curR - curL)
    for a, e in zip((axl, axm, axr), exts):
        nx0 = xl_edge + (e.x0 - curL) * scale
        nx1 = xl_edge + (e.x1 - curL) * scale
        p0 = invB.transform((nx0, eC.y0))
        p1 = invB.transform((nx1, eC.y1))
        a.set_position([p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1]])

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    cx = [sum(a.get_window_extent(renderer=rend).intervalx) / 2 for _, _, a in trace_labels]
    target_cx = min(cx)
    for (kind, ax, art), c in zip(trace_labels, cx):
        dx = (target_cx - c) / ax.get_window_extent(renderer=rend).width
        x, y = art.get_position()
        if kind == "3d":
            art.set_position((x + dx, y))
        else:
            ax.yaxis.set_label_coords(x + dx, y)

    out = os.path.join(figures_dir, "fig2.png")
    if os.environ.get("NO_HEADER"):
        fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
        print(f"Saved {out}")
    else:
        from svg_composer import compose_fig2
        compose_fig2(fig, sfL, axl, axr, axD, axE, axF, out, dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
