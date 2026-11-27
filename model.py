"""
Automated kinematic scoring of tool-use pantomime gestures in limb apraxia.

Extracts 46 scale-free kinematic features from 8-marker 100 Hz electromagnetic
motion-capture recordings (Thumb, Index, Middle, MCP2/Knuckle, Wrist, Elbow,
Left Shoulder, Right Shoulder; 3D position + 3D velocity) and predicts
participant-level praxis scores (0-3 scale) via a two-pathway Ridge blend:

  - Item-level pathway: 39 item-specific L2 Ridge regressions (alpha = 10)
    predicting trial-level gesture scores, aggregated per participant via
    inverse cross-validated MSE reliability weights and rescaled around the
    training-fold global mean.
  - Participant-level pathway: L2 Ridge regression (alpha = 10) on participant
    aggregate kinematics (concatenated mean and SD across trials; 92 inputs).
  - Blended prediction: Equal-weight average (0.5 / 0.5) of the two pathways
    with linear variance-matching calibration fit on the training fold.

Usage:
  python model.py          # Training set LOPO cross-validation (n = 41)
  python model.py --test   # Out-of-sample held-out test evaluation (n = 12)
  python model.py --all    # Full-cohort LOPO cross-validation (N = 53)
"""

import argparse
import json
import os
import warnings
from collections import defaultdict
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

# =============================================================================
# Constants
# =============================================================================

THUMB, INDEX, MIDDLE, KNUCKLE = 0, 1, 2, 3
WRIST, ELBOW, LEFT_SHOULDER, RIGHT_SHOULDER = 4, 5, 6, 7

APRAXIA_CUTOFF = 2.52

# Held-out test cohort (n = 12: 3 control + 9 left-hemisphere stroke)
TEST_PARTICIPANTS = [
    "CL010", "CL019", "CL030",
    "ST027", "ST028", "ST034", "ST038", "ST056",
    "ST060", "ST061", "ST067", "ST069",
]

# Canonical names for all 46 scale-free kinematic features
FEATURE_NAMES = [
    # Hand posture & distal individuation (19 features, indices 0-18)
    "Thumb-Index aperture mean",      # 0
    "Thumb-Index aperture std",       # 1
    "Thumb-Index aperture min",       # 2
    "Thumb-Index aperture max",       # 3
    "Thumb-Index speed corr",         # 4
    "Index-Middle speed corr",        # 5
    "Index-Wrist speed corr",         # 6
    "Wrist-Shoulder speed corr",      # 7
    "Thumb-Index norm dist mean",     # 8
    "Thumb-Index norm dist std",      # 9
    "Thumb-Middle norm dist min",     # 10
    "Thumb-Middle norm dist max",     # 11
    "Middle-Wrist norm dist mean",    # 12
    "Thumb-Middle change rate",       # 13
    "Thumb-Middle opening count",     # 14
    "Thumb-Middle closing count",     # 15
    "Index-Knuckle excursion corr",   # 16
    "Knuckle-Elbow excursion corr",   # 17
    "Thumb-Wrist excursion corr",     # 18
    # Arm-hand coordination & velocity direction (13 features, indices 19-31)
    "Thumb-Arm rel vel std X",        # 19
    "Thumb-Arm rel vel std Y",        # 20
    "Thumb-Arm rel vel std Z",        # 21
    "Thumb-Shoulder vel diff mean",   # 22
    "Thumb-Shoulder vel diff std",    # 23
    "Thumb-Shoulder vel diff max",    # 24
    "Arm-Hand vel angle mean",        # 25
    "Arm-Hand vel angle std",         # 26
    "Arm-Hand vel angle range",       # 27
    "Index vel-wt dir X",             # 28
    "Index vel-wt dir Y",             # 29
    "Index vel-wt dir Z",             # 30
    "Vert/Horiz vel ratio",           # 31
    # Amplitude, smoothness, & timing (13 features, indices 32-44)
    "Index norm path length",         # 32
    "Index norm range X",             # 33
    "Index norm range Y",             # 34
    "Index norm range Z",             # 35
    "Index norm range norm",          # 36
    "Index norm speed mean",          # 37
    "Index norm speed std",           # 38
    "Index norm speed max",           # 39
    "Speed peaks count",              # 40
    "Dimensionless jerk mean",        # 41
    "Dimensionless jerk std",         # 42
    "Speed autocorr peak",            # 43
    "Duration",                       # 44
    # Motion type (1 feature, index 45)
    "Motion-type feature",            # 45
]

ALL_FEATURE_NAMES = FEATURE_NAMES

FEATURE_GROUPS = {
    "Hand Posture": list(range(0, 19)),
    "Arm Posture": list(range(19, 32)),
    "Amplitude/Timing": list(range(32, 45)),
    "Motion Type": [45],
}


# =============================================================================
# Tool-Use Pantomime Item Metadata (39 Items)
# =============================================================================

ITEM_INFO = {
    1: {
        "name": "Beer Mug",
        "category": "drinking",
        "hand_posture": "Fist (grasping handle) or C-shape (grasping mug)",
        "arm_posture": "Lift to side of mouth (fist) or directly to mouth (C-shape)",
        "amplitude": "From table/space to face",
        "key_motion": "lifting, tilting",
    },
    2: {
        "name": "Ping-Pong Paddle",
        "category": "sports",
        "hand_posture": "Fist or open hand",
        "arm_posture": "Swinging back and forth in front of body",
        "amplitude": "Enough to move ball forward",
        "key_motion": "swinging, back-and-forth",
    },
    3: {
        "name": "Bolt Lock",
        "category": "tool",
        "hand_posture": "Fingers curled, thumb pinch with index/middle finger",
        "arm_posture": "Reach out in front, arm twists",
        "amplitude": "Hand must twist (not shake), 1-3 twists",
        "key_motion": "twisting, rotating",
    },
    4: {
        "name": "Tea Bag",
        "category": "food/drink",
        "hand_posture": "Pinch with fingertips",
        "arm_posture": "Move hand up and down (dipping motion)",
        "amplitude": "Hand shouldn't go too far into cup",
        "key_motion": "dipping, up-down",
    },
    5: {
        "name": "Comb",
        "category": "grooming",
        "hand_posture": "Clenched hand or fist",
        "arm_posture": "Must gesture to head",
        "amplitude": "Usually okay, can comb in same location",
        "key_motion": "combing, brushing head",
    },
    6: {
        "name": "Eraser",
        "category": "writing",
        "hand_posture": "Thumb to fingertips, not pendulum motion",
        "arm_posture": "Aim at table, within 4 inches, horizontal plane",
        "amplitude": "Not larger than sheet of paper, 3+ back-forth motions",
        "key_motion": "erasing, back-forth horizontal",
    },
    7: {
        "name": "Paint Roller",
        "category": "tool",
        "hand_posture": "Fist grip, roller perpendicular to surface",
        "arm_posture": "Paint toward table or wall",
        "amplitude": "Some arm movement required, one up-down OK",
        "key_motion": "rolling, up-down",
    },
    8: {
        "name": "Knife",
        "category": "kitchen",
        "hand_posture": "Fist grip, no wrist twist, angled at table",
        "arm_posture": "Down at table, within 18 inches",
        "amplitude": "Cannot hit table",
        "key_motion": "chopping, slicing",
    },
    9: {
        "name": "Axe",
        "category": "tool",
        "hand_posture": "Fist grip, wrist OK with arm movement",
        "arm_posture": "Chopping from shoulder height to table",
        "amplitude": "Full chop motion, cannot hit table",
        "key_motion": "chopping, large vertical",
    },
    10: {
        "name": "Bottle Opener",
        "category": "kitchen",
        "hand_posture": "Prying motion, must hold opener",
        "arm_posture": "Arm elevated from table",
        "amplitude": "Opener moves less than 12 inches, max 3 times",
        "key_motion": "prying, levering",
    },
    11: {
        "name": "Hammer",
        "category": "tool",
        "hand_posture": "Fist or slightly open, no pinching",
        "arm_posture": "Vertical hammering, wrist movement OK",
        "amplitude": "Hand below head, cannot touch table",
        "key_motion": "hammering, vertical striking",
    },
    12: {
        "name": "Iron",
        "category": "household",
        "hand_posture": "Holding posture or fist, no flagrant wrist",
        "arm_posture": "Parallel to table, within 12 inches",
        "amplitude": "At least one up-back motion",
        "key_motion": "ironing, back-forth horizontal",
    },
    13: {
        "name": "Saw",
        "category": "tool",
        "hand_posture": "Fist, back of hand facing left",
        "arm_posture": "Move arm away from body and back",
        "amplitude": "Extend at least 4 inches, one back-forth motion",
        "key_motion": "sawing, push-pull",
    },
    14: {
        "name": "Paper Clip",
        "category": "office",
        "hand_posture": "Pinch with thumb/index/middle, no duck hands",
        "arm_posture": "Hand in front of body, arm extends outward",
        "amplitude": "1-3 pinches OK",
        "key_motion": "pinching, clipping",
    },
    15: {
        "name": "Razor",
        "category": "grooming",
        "hand_posture": "Closed hand grip, no face touching",
        "arm_posture": "Toward shavable body part, within 4 inches",
        "amplitude": "Two razor strokes minimum",
        "key_motion": "shaving, short strokes on face",
    },
    16: {
        "name": "Fork",
        "category": "eating",
        "hand_posture": "Fist, pencil grip, or two-finger pinch",
        "arm_posture": "From table/space to mouth",
        "amplitude": "Within 4 inches of mouth",
        "key_motion": "eating, lifting to mouth",
    },
    17: {
        "name": "Cigarette",
        "category": "smoking",
        "hand_posture": "Pointer/middle finger OR thumb-index pinch",
        "arm_posture": "From space to mouth or mouth to space",
        "amplitude": "Usually okay, may do only once",
        "key_motion": "smoking, to/from mouth",
    },
    18: {
        "name": "Deodorant",
        "category": "grooming",
        "hand_posture": "Not too broad grip, no fist",
        "arm_posture": "Toward armpit area",
        "amplitude": "Some oscillation required",
        "key_motion": "applying, up-down under arm",
    },
    19: {
        "name": "Match",
        "category": "fire",
        "hand_posture": "Wrist flick OK, must hold match",
        "arm_posture": "Some arm movement required",
        "amplitude": "Reasonable amount to light",
        "key_motion": "striking, flicking",
    },
    20: {
        "name": "Toothbrush",
        "category": "grooming",
        "hand_posture": "At least 3 fingers",
        "arm_posture": "Up-down or back-forth, within 4 inches of face",
        "amplitude": "Only where teeth are, multiple oscillations",
        "key_motion": "brushing, oscillating near mouth",
    },
    21: {
        "name": "Clamp",
        "category": "tool",
        "hand_posture": "Pinch with thumb/index/middle, no duck hands",
        "arm_posture": "In front of body, rotation from arm",
        "amplitude": "Rotation 2-12 inches, at least 2 rotations",
        "key_motion": "clamping, rotating",
    },
    22: {
        "name": "Wrench",
        "category": "tool",
        "hand_posture": "Some wrist movement, most in arm",
        "arm_posture": "Torque motion, back-forth or side-side",
        "amplitude": "Generally okay",
        "key_motion": "wrenching, torque motion",
    },
    23: {
        "name": "Clothing Pin",
        "category": "household",
        "hand_posture": "Two-finger pinch",
        "arm_posture": "Oriented outward and upward if extended",
        "amplitude": "Fingers within 3 inches, max 3 pinches",
        "key_motion": "pinching, releasing",
    },
    24: {
        "name": "Screwdriver",
        "category": "tool",
        "hand_posture": "Fist grip, thumb curled or on top",
        "arm_posture": "Turn into space or table",
        "amplitude": "Some rotation, at least 2 twists",
        "key_motion": "screwing, rotating",
    },
    25: {
        "name": "Nail Clipper",
        "category": "grooming",
        "hand_posture": "Thumb to fingertips, open-close motion",
        "arm_posture": "Toward body or into space",
        "amplitude": "Cannot open more than 3 inches",
        "key_motion": "clipping, small pinching",
    },
    26: {
        "name": "Car Jack",
        "category": "tool",
        "hand_posture": "Fist grip",
        "arm_posture": "Extend and pull back, or crank motion",
        "amplitude": "8-14 inches, at least 2 pulls",
        "key_motion": "cranking, pumping",
    },
    27: {
        "name": "Camera",
        "category": "device",
        "hand_posture": "Hold camera and press button",
        "arm_posture": "In front of face, within 6 inch sphere",
        "amplitude": "Finger cannot go far into camera",
        "key_motion": "holding, pressing button",
    },
    28: {
        "name": "Remote Control",
        "category": "device",
        "hand_posture": "Not fist, use thumb to press",
        "arm_posture": "Aimed forward, minimal twisting",
        "amplitude": "Minimal arm movement",
        "key_motion": "pressing buttons with thumb",
    },
    29: {
        "name": "Stapler",
        "category": "office",
        "hand_posture": "Push with palm or bottom of fist",
        "arm_posture": "Aimed at table",
        "amplitude": "Enough force to staple",
        "key_motion": "pressing down",
    },
    30: {
        "name": "Key",
        "category": "household",
        "hand_posture": "Thumb to curled pointer or fingertips",
        "arm_posture": "Turn arm in space",
        "amplitude": "At least 45 degrees rotation, under 8 inches",
        "key_motion": "turning, rotating",
    },
    31: {
        "name": "Lipstick",
        "category": "grooming",
        "hand_posture": "Thumb to fingertips, not fist",
        "arm_posture": "Aimed at lips, within 2 inches",
        "amplitude": "Enough to cover lips",
        "key_motion": "applying, horizontal near lips",
    },
    32: {
        "name": "Scissors",
        "category": "office",
        "hand_posture": "Open-close motion, two-finger pinch OK",
        "arm_posture": "Forward motion required",
        "amplitude": "Scissor at least twice, in sync with arm",
        "key_motion": "cutting, open-close with forward motion",
    },
    33: {
        "name": "Corkscrew",
        "category": "kitchen",
        "hand_posture": "Twist with entire hand, fist OK",
        "arm_posture": "Elevated at least 4 inches, below head",
        "amplitude": "Twist at least 90 degrees, at least twice",
        "key_motion": "twisting, screwing motion",
    },
    34: {
        "name": "Lighter",
        "category": "fire",
        "hand_posture": "Thumb flick down while holding",
        "arm_posture": "Generally correct, minimal movement",
        "amplitude": "Movement mainly in thumb",
        "key_motion": "flicking, thumb motion",
    },
    35: {
        "name": "Syringe",
        "category": "medical",
        "hand_posture": "Inject with thumb, hold with 2 fingers",
        "arm_posture": "Into space or toward arm/leg",
        "amplitude": "Max 3 times",
        "key_motion": "injecting, pressing plunger",
    },
    36: {
        "name": "Soap Dispenser",
        "category": "household",
        "hand_posture": "Press with hand/finger/thumb",
        "arm_posture": "4-10 inches from table",
        "amplitude": "Reasonable press depth",
        "key_motion": "pressing down, pumping",
    },
    37: {
        "name": "Calculator",
        "category": "device",
        "hand_posture": "Poke with one or multiple fingers",
        "arm_posture": "Toward table",
        "amplitude": "At least 2 presses",
        "key_motion": "poking, pressing buttons",
    },
    38: {
        "name": "Keyboard",
        "category": "device",
        "hand_posture": "Keyboard hands, all fingers must move",
        "arm_posture": "Toward table, side-to-side movement",
        "amplitude": "Press multiple times (3+)",
        "key_motion": "typing, multiple finger motion",
    },
    39: {
        "name": "Watch",
        "category": "device",
        "hand_posture": "Pinching to wind",
        "arm_posture": "Generally static",
        "amplitude": "No excessively large movements",
        "key_motion": "winding, small twisting",
    },
}

CATEGORIES = {
    "tool": [3, 7, 9, 11, 13, 21, 22, 24, 26],
    "grooming": [5, 15, 18, 20, 25, 31],
    "kitchen": [8, 10, 33],
    "eating/drinking": [1, 4, 16, 17],
    "office": [6, 14, 29, 32],
    "household": [12, 23, 30, 36],
    "device": [27, 28, 37, 38, 39],
    "fire": [19, 34],
    "medical": [35],
    "sports": [2],
}

MOTION_TYPES = {
    "vertical_striking": [9, 11],
    "horizontal_back_forth": [6, 12, 13],
    "rotating_twisting": [3, 21, 22, 24, 30, 33, 39],
    "lifting_to_face": [1, 4, 5, 15, 16, 17, 20, 31],
    "pressing_poking": [27, 28, 29, 34, 35, 36, 37, 38],
    "pinching": [14, 23, 25, 32],
    "swinging": [2],
    "up_down": [7, 18, 26],
    "prying": [10],
    "other": [8, 19],
}


def get_item_name(item_id):
    """Return the display name for `item_id` (1-39)."""
    if item_id in ITEM_INFO:
        return ITEM_INFO[item_id]["name"]
    return f"Item {item_id}"


def get_motion_type(item_id):
    """Return the primary kinematic motion category for `item_id`."""
    for motion_type, items in MOTION_TYPES.items():
        if item_id in items:
            return motion_type
    return "unknown"


def print_all_items():
    """Print a table of all 39 tool-use pantomime items."""
    for item_id in sorted(ITEM_INFO.keys()):
        info = ITEM_INFO[item_id]
        motion = get_motion_type(item_id)
        print(f"{item_id:2d}. {info['name']:<20} | {motion:<22} | {info['key_motion']}")



# =============================================================================
# Kinematic Feature Extraction (46 Scale-Free Features)
# =============================================================================

def _zscore_3d(x):
    """Within-trial channel-wise z-score for a single marker's (T, 3) stream."""
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    s = np.where(s > 1e-8, s, 1.0)
    return (x - m) / s


def _safe_corr(a, b):
    """Pearson correlation with zero-variance and short-trial guard."""
    if len(a) < 5 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def _angle_stats(u, v):
    """Mean, SD, and range of the 3D angle (radians) between two vector streams."""
    cos_a = np.sum(u * v, axis=1) / (
        np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-8
    )
    ang = np.arccos(np.clip(cos_a, -1.0, 1.0))
    return [float(ang.mean()), float(ang.std()), float(ang.max() - ang.min())]


def _prepare_streams(raw, downsample=15):
    """
    Reshape 100 Hz (T, 48) trial array into 3D position (cols 0:24) and
    3D velocity (cols 24:48), compute within-trial z-scored streams at 100 Hz,
    and decimate by `downsample` (default 15 -> 6.67 Hz).
    """
    raw100 = raw.astype(np.float64)
    pos100 = raw100[:, :24].reshape(-1, 8, 3)
    vel100 = raw100[:, 24:].reshape(-1, 8, 3)

    zp100 = np.empty_like(pos100)
    zv100 = np.empty_like(vel100)
    for m in range(8):
        zp100[:, m] = _zscore_3d(pos100[:, m])
        zv100[:, m] = _zscore_3d(vel100[:, m])

    pos = pos100[::downsample]
    zp = zp100[::downsample]
    zv = zv100[::downsample]
    hand_len = float(np.median(np.linalg.norm(pos[:, KNUCKLE] - pos[:, WRIST], axis=1))) + 1e-3
    return pos, zp, zv, hand_len


def compute_hand_posture_features(pos, zp, zv, hand_len):
    """
    Compute 19 scale-free hand posture and distal individuation features:
      - Hand-length-normalized 3D Thumb-Index grasp aperture (mean, std, min, max)
      - Multi-joint normalized speed correlations (Thumb-Index, Index-Middle,
        Index-Wrist, Wrist-LeftShoulder)
      - Within-trial normalized trajectory decoupling and distal-proximal excursion
        correlations.
    """
    d_ti = np.linalg.norm(pos[:, THUMB] - pos[:, INDEX], axis=1) / hand_len

    spd_th = np.linalg.norm(zv[:, THUMB], axis=1)
    spd_ix = np.linalg.norm(zv[:, INDEX], axis=1)
    spd_md = np.linalg.norm(zv[:, MIDDLE], axis=1)
    spd_wr = np.linalg.norm(zv[:, WRIST], axis=1)
    spd_ls = np.linalg.norm(zv[:, LEFT_SHOULDER], axis=1)
    spd_corrs = [
        _safe_corr(spd_th, spd_ix),
        _safe_corr(spd_ix, spd_md),
        _safe_corr(spd_ix, spd_wr),
        _safe_corr(spd_wr, spd_ls),
    ]

    zp_ti = np.linalg.norm(zp[:, THUMB] - zp[:, INDEX], axis=1)
    zp_tm = np.linalg.norm(zp[:, THUMB] - zp[:, MIDDLE], axis=1)
    zp_mw = np.linalg.norm(zp[:, MIDDLE] - zp[:, WRIST], axis=1)
    zp_tm_chg = np.diff(zp_tm)
    chg_stats = [
        float(np.abs(zp_tm_chg).mean()) if len(zp_tm_chg) > 0 else 0.0,
        float((zp_tm_chg > 0).sum()) if len(zp_tm_chg) > 0 else 0.0,
        float((zp_tm_chg < 0).sum()) if len(zp_tm_chg) > 0 else 0.0,
    ]
    exc_corrs = [
        _safe_corr(np.linalg.norm(zp[:, INDEX], axis=1), np.linalg.norm(zp[:, KNUCKLE], axis=1)),
        _safe_corr(np.linalg.norm(zp[:, KNUCKLE], axis=1), np.linalg.norm(zp[:, ELBOW], axis=1)),
        _safe_corr(np.linalg.norm(zp[:, THUMB], axis=1), np.linalg.norm(zp[:, WRIST], axis=1)),
    ]

    return [
        float(d_ti.mean()), float(d_ti.std()), float(d_ti.min()), float(d_ti.max()),
        *spd_corrs,
        float(zp_ti.mean()), float(zp_ti.std()),
        float(zp_tm.min()), float(zp_tm.max()),
        float(zp_mw.mean()),
        *chg_stats,
        *exc_corrs,
    ]


def compute_arm_posture_features(zv):
    """
    Compute 13 scale-free arm-hand coordination and 3D velocity direction features
    using ipsilateral LEFT_SHOULDER and vertical axis Z (axis 2).
    """
    zv_rel = zv[:, THUMB] - 0.5 * (zv[:, WRIST] + zv[:, LEFT_SHOULDER])
    zv_ts_dist = np.linalg.norm(zv[:, THUMB] - zv[:, LEFT_SHOULDER], axis=1)
    vel_synergy_ang = _angle_stats(
        zv[:, MIDDLE] - zv[:, LEFT_SHOULDER],
        zv[:, THUMB] - zv[:, MIDDLE],
    )

    zv_ix = zv[:, INDEX]
    sp = np.linalg.norm(zv_ix, axis=1)
    if sp.sum() > 0:
        w = sp / (sp.sum() + 1e-8)
        avg_d = (zv_ix.T * w).sum(axis=1)
        avg_d = avg_d / (np.linalg.norm(avg_d) + 1e-8)
        pol = avg_d.tolist()
    else:
        pol = [0.0, 0.0, 0.0]

    # Vertical axis in trakSTAR is Z (axis 2); horizontal plane is (X, Y)
    vert = float(np.abs(zv_ix[:, 2]).sum())
    horiz = float(np.sqrt(zv_ix[:, 0] ** 2 + zv_ix[:, 1] ** 2).sum())
    vh_ratio = vert / (horiz + 1e-8)

    return [
        float(zv_rel[:, 0].std()), float(zv_rel[:, 1].std()), float(zv_rel[:, 2].std()),
        float(zv_ts_dist.mean()), float(zv_ts_dist.std()), float(zv_ts_dist.max()),
        *vel_synergy_ang,
        *pol,
        vh_ratio,
    ]


def compute_amplitude_timing_features(zp, zv, downsample=15, fs=100.0):
    """
    Compute 13 dimensionless trajectory smoothness, range, rhythm, and timing
    features from normalized Index fingertip position/velocity plus duration (s).
    """
    z_traj = zp[:, INDEX]
    z_deriv = zv[:, INDEX]
    n_frames = len(z_traj)
    features = []

    if n_frames > 1:
        features.append(float(np.linalg.norm(np.diff(z_traj, axis=0), axis=1).sum()))
    else:
        features.append(0.0)

    rng = z_traj.max(axis=0) - z_traj.min(axis=0)
    features.extend(rng.tolist())
    features.append(float(np.linalg.norm(rng)))

    sp = np.linalg.norm(z_deriv, axis=1)
    features.extend([float(sp.mean()), float(sp.std()), float(sp.max())])

    if len(sp) > 5:
        sm = np.convolve(sp, np.ones(3) / 3.0, mode="same")
        peaks = int(((sm[1:-1] > sm[:-2]) & (sm[1:-1] > sm[2:])).sum())
        features.append(float(peaks))
    else:
        features.append(0.0)

    if n_frames > 2:
        acc = np.diff(z_deriv, axis=0)
        if len(acc) > 1:
            jrk = np.linalg.norm(np.diff(acc, axis=0), axis=1)
            features.extend([float(jrk.mean()), float(jrk.std())])
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])

    if len(sp) > 10:
        c = sp - sp.mean()
        ac = np.correlate(c, c, mode="full")[len(c) - 1:]
        ac = ac / (ac[0] + 1e-8)
        pk = 0
        if len(ac) > 5:
            for i in range(2, min(len(ac) - 1, 20)):
                if ac[i] > ac[i - 1] and ac[i] > ac[i + 1]:
                    pk = i
                    break
        features.append(float(ac[pk]) if pk > 0 else 0.0)
    else:
        features.append(0.0)

    duration = float(n_frames / (fs / downsample))
    features.append(duration)
    return features


def compute_motion_type_features(pos, zv, motion_type, hand_len):
    """
    Compute 1 scale-free motion-type-specific feature tailored to the gesture's
    kinematic category.
    """
    n_frames = len(pos)
    p_th, p_ix = pos[:, THUMB], pos[:, INDEX]
    p_wr, p_ls = pos[:, WRIST], pos[:, LEFT_SHOULDER]
    spd_th = np.linalg.norm(zv[:, THUMB], axis=1)
    spd_ix = np.linalg.norm(zv[:, INDEX], axis=1)

    if motion_type == "rotating_twisting":
        hv = p_ix - p_th
        if n_frames > 1:
            v1, v2 = hv[:-1], hv[1:]
            ca = np.sum(v1 * v2, axis=1) / (
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
            )
            return [float(np.sum(np.arccos(np.clip(ca, -1.0, 1.0))))]
        return [0.0]

    elif motion_type == "vertical_striking":
        return [float(np.abs(np.minimum(zv[:, INDEX, 2], 0.0)).max())]

    elif motion_type == "lifting_to_face":
        return [float(((p_wr[:, 2] - p_ls[:, 2]) / hand_len).max())]

    elif motion_type == "pressing_poking":
        return [float(max(spd_th.max(), spd_ix.max()))]

    elif motion_type == "pinching":
        d_ti = np.linalg.norm(p_th - p_ix, axis=1) / hand_len
        return [float(np.abs(np.diff(d_ti)).sum()) if len(d_ti) > 2 else 0.0]

    else:
        return [float(spd_ix.mean())]


def extract_all_features(raw, item_id, downsample=15):
    """
    Extract all 46 scale-free kinematic features from a 100 Hz trial array.

    Parameters
    ----------
    raw : ndarray, shape (n_frames_100hz, 48)
        Time-stripped 100 Hz motion capture data (24 position + 24 velocity cols).
    item_id : int
        Gesture item ID (1-39).
    downsample : int
        Temporal decimation factor (default 15: 100 Hz -> 6.67 Hz).

    Returns
    -------
    list of float
        46 features: 19 hand posture + 13 arm posture + 13 amplitude/timing + 1 motion type.
    """
    pos, zp, zv, hand_len = _prepare_streams(raw, downsample=downsample)
    motion_type = get_motion_type(item_id)
    return (
        compute_hand_posture_features(pos, zp, zv, hand_len)
        + compute_arm_posture_features(zv)
        + compute_amplitude_timing_features(zp, zv, downsample=downsample)
        + compute_motion_type_features(pos, zv, motion_type, hand_len)
    )


# =============================================================================
# Data Loading
# =============================================================================

def parse_component_scores(score_entry):
    """
    Parse trial manual score entry `[[content, HP, AP, AT]]`.

    Returns
    -------
    (content, hp, ap, at, total) : tuple of (int or None)
        `content` is 1 if the correct tool was pantomimed (0 otherwise, None if NaN).
        When `content == 1`, `hp`, `ap`, `at` are in {0, 1} and `total = hp + ap + at` (0-3).
    """
    if isinstance(score_entry, list) and len(score_entry) > 0:
        inner = score_entry[0] if isinstance(score_entry[0], list) else score_entry

        def is_nan(x):
            return x is None or (isinstance(x, float) and np.isnan(x))

        if len(inner) >= 4:
            content, hp, ap, at = inner[0], inner[1], inner[2], inner[3]
            if is_nan(content):
                return None, None, None, None, None
            content = int(content)
            if content == 0:
                return content, None, None, None, None
            hp = int(hp) if not is_nan(hp) else None
            ap = int(ap) if not is_nan(ap) else None
            at = int(at) if not is_nan(at) else None
            total = (hp + ap + at) if (hp is not None and ap is not None and at is not None) else None
            return content, hp, ap, at, total

    return None, None, None, None, None


@lru_cache(maxsize=4)
def _load_and_extract_cached(abs_data_dir, downsample=15):
    """
    Load segmented 100 Hz trial recordings from `data_dir` and extract the
    46 kinematic features for every valid trial.

    Returns
    -------
    X : ndarray, shape (n_trials, 46)
    scores_dict : dict of lists ('content', 'hp', 'ap', 'at', 'total')
    items : ndarray, shape (n_trials,)
    participants : list of str
    """
    with open(os.path.join(abs_data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    meta_dict = {entry["trial_name"]: entry for entry in metadata}
    features, items, participants = [], [], []
    scores_dict = {"content": [], "hp": [], "ap": [], "at": [], "total": []}

    for trial_name, meta in meta_dict.items():
        content, hp, ap, at, total = parse_component_scores(meta["score"])
        if content is None and total is None:
            continue

        raw_with_time = np.load(os.path.join(abs_data_dir, f"{trial_name}.npy")).astype(np.float32)
        raw = raw_with_time[:, 1:]

        features.append(extract_all_features(raw, meta["item"], downsample=downsample))
        items.append(meta["item"])
        participants.append(meta["participant"])

        scores_dict["content"].append(content)
        scores_dict["hp"].append(hp)
        scores_dict["ap"].append(ap)
        scores_dict["at"].append(at)
        scores_dict["total"].append(total)

    return np.array(features), scores_dict, np.array(items), tuple(participants)


def load_and_extract(data_dir, downsample=15):
    """Load segmented 100 Hz trial recordings from `data_dir` and extract all 46 features (cached in memory)."""
    X, scores_dict, items, participants = _load_and_extract_cached(os.path.abspath(data_dir), downsample)
    return X.copy(), {k: list(v) for k, v in scores_dict.items()}, items.copy(), list(participants)


def get_train_test_masks(participants):
    """Return boolean masks `(train_mask, test_mask)` splitting n=41 train and n=12 test."""
    test_set = set(TEST_PARTICIPANTS)
    train_mask = np.array([p not in test_set for p in participants])
    return train_mask, ~train_mask


def load_train_data(data_dir, downsample=15, content_filter=None):
    """Load training cohort data (41 participants: 7 control, 34 stroke)."""
    X, scores_dict, items, participants = load_and_extract(data_dir, downsample=downsample)
    train_mask, _ = get_train_test_masks(participants)
    if content_filter is not None:
        content_mask = np.array([c == content_filter for c in scores_dict["content"]])
        combined_mask = train_mask & content_mask
    else:
        combined_mask = train_mask

    filtered_scores = {
        k: [v for v, m in zip(vals, combined_mask) if m]
        for k, vals in scores_dict.items()
    }
    return (
        X[combined_mask],
        filtered_scores,
        items[combined_mask],
        [p for p, m in zip(participants, combined_mask) if m],
    )


def load_test_data(data_dir, downsample=15, content_filter=None):
    """Load held-out test cohort data (12 participants: 3 control, 9 stroke)."""
    X, scores_dict, items, participants = load_and_extract(data_dir, downsample=downsample)
    _, test_mask = get_train_test_masks(participants)
    if content_filter is not None:
        content_mask = np.array([c == content_filter for c in scores_dict["content"]])
        combined_mask = test_mask & content_mask
    else:
        combined_mask = test_mask

    filtered_scores = {
        k: [v for v, m in zip(vals, combined_mask) if m]
        for k, vals in scores_dict.items()
    }
    return (
        X[combined_mask],
        filtered_scores,
        items[combined_mask],
        [p for p, m in zip(participants, combined_mask) if m],
    )


def load_all_data(data_dir, downsample=15):
    """Load full-cohort data (all 53 participants: 10 control, 43 stroke)."""
    return load_and_extract(data_dir, downsample=downsample)


# =============================================================================
# BlendModel: Item-Level + Participant-Level Ridge Blend
# =============================================================================

class BlendModel:
    """
    Two-pathway Ridge regression blend for participant-level praxis scoring.

    Pathway 1 — Item-level models:
        39 item-specific Ridge regressions (`alpha_item = 10.0`) predicting
        trial-level gesture scores (0-3). Item predictions are aggregated per
        participant using inverse 5-fold cross-validated MSE reliability weights
        and rescaled around the global mean (`bias_factor`).

    Pathway 2 — Participant-level model:
        Single Ridge regression (`alpha_direct = 10.0`) fit to participant-level
        summary kinematics (concatenated mean and SD across trials; 92 inputs).

    Blended prediction:
        Equal-weight combination (`blend_weight = 0.5`) followed by linear
        variance-matching calibration fit on the training fold.
    """

    def __init__(self, alpha_item=10.0, alpha_direct=10.0, blend_weight=0.5, calibrate=True):
        self.alpha_item = alpha_item
        self.alpha_direct = alpha_direct
        self.blend_weight = blend_weight
        self.calibrate = calibrate

        self.item_models = {}
        self.item_scalers = {}
        self.item_means = {}
        self.item_weights = {}
        self.global_mean = 2.5
        self.bias_factor = 0.5
        self.direct_model = None
        self.direct_scaler = None
        self.calib_slope = 1.0
        self.calib_intercept = 0.0

    def fit(self, X, y, items, participants):
        """Fit item-level, participant-level, and calibration parameters on training data."""
        self.global_mean = y.mean()

        part_scores = defaultdict(list)
        for score, pid in zip(y, participants):
            part_scores[pid].append(score)
        part_true = {p: np.mean(s) for p, s in part_scores.items()}

        # 1. Item-level Ridge models + inverse-CV-MSE reliability weights
        for item_id in set(items):
            mask = items == item_id
            Xi, yi = X[mask], y[mask]
            self.item_means[item_id] = yi.mean()

            if len(Xi) < 10 or len(set(yi)) < 2:
                self.item_weights[item_id] = 0.1
                continue

            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xi)
            self.item_scalers[item_id] = scaler

            model = Ridge(alpha=self.alpha_item)
            model.fit(Xs, yi)
            self.item_models[item_id] = model

            try:
                cv_preds = cross_val_predict(model, Xs, yi, cv=min(5, len(Xi)))
                self.item_weights[item_id] = 1.0 / (np.mean((cv_preds - yi) ** 2) + 0.01)
            except Exception:
                self.item_weights[item_id] = 0.1

        total_weight = sum(self.item_weights.values())
        for k in self.item_weights:
            self.item_weights[k] /= total_weight

        # 2. Rescaling factor for aggregated item-level predictions
        trial_preds = self._predict_trials(X, items)
        loio_data = defaultdict(lambda: {"preds": [], "weights": [], "true": []})
        for pred, item_id, true_score, pid in zip(trial_preds, items, y, participants):
            loio_data[pid]["preds"].append(pred)
            loio_data[pid]["weights"].append(self.item_weights.get(item_id, 0.1))
            loio_data[pid]["true"].append(true_score)

        def bias_objective(bias):
            errors = []
            for pid, d in loio_data.items():
                w = np.array(d["weights"])
                w = w / w.sum()
                weighted_pred = np.sum(np.array(d["preds"]) * w)
                pred = np.clip(self.global_mean + (weighted_pred - self.global_mean) * (1 + bias), 0, 3)
                errors.append(pred - np.mean(d["true"]))
            return np.mean(np.array(errors) ** 2)

        result = minimize_scalar(bias_objective, bounds=(-1, 3), method="bounded")
        self.bias_factor = result.x

        # 3. Participant-level Ridge regression on [mean, SD] across trials
        part_features = defaultdict(list)
        for i, pid in enumerate(participants):
            part_features[pid].append(X[i])

        pids = sorted(part_features.keys())
        X_parts, y_parts = [], []
        for pid in pids:
            feats = np.array(part_features[pid])
            X_parts.append(np.concatenate([feats.mean(axis=0), feats.std(axis=0)]))
            y_parts.append(part_true[pid])

        X_parts = np.nan_to_num(np.array(X_parts))
        y_parts = np.array(y_parts)

        self.direct_scaler = StandardScaler()
        X_scaled = self.direct_scaler.fit_transform(X_parts)
        self.direct_model = Ridge(alpha=self.alpha_direct)
        self.direct_model.fit(X_scaled, y_parts)

        # 4. Linear variance-matching calibration
        if self.calibrate:
            loio_preds = {}
            for pid, d in loio_data.items():
                w = np.array(d["weights"])
                w = w / w.sum()
                raw = np.sum(np.array(d["preds"]) * w)
                loio_preds[pid] = np.clip(
                    self.global_mean + (raw - self.global_mean) * (1 + self.bias_factor), 0, 3
                )

            direct_preds = {}
            for pid in pids:
                feats = np.array(part_features[pid])
                agg = np.concatenate([feats.mean(axis=0), feats.std(axis=0)])
                scaled = self.direct_scaler.transform(np.nan_to_num(agg.reshape(1, -1)))
                direct_preds[pid] = np.clip(self.direct_model.predict(scaled)[0], 0, 3)

            bw = self.blend_weight
            blended = np.array([bw * loio_preds[p] + (1 - bw) * direct_preds[p] for p in pids])
            true_arr = np.array([part_true[p] for p in pids])

            if np.std(blended) > 1e-8:
                self.calib_slope = np.std(true_arr) / np.std(blended)
                self.calib_intercept = np.mean(true_arr) - self.calib_slope * np.mean(blended)

    def _predict_trials(self, X, items):
        """Predict clipped trial-level scores (0-3) using item-specific Ridge models."""
        preds = np.zeros(len(X))
        for i in range(len(X)):
            item_id = items[i]
            if item_id not in self.item_models:
                preds[i] = self.item_means.get(item_id, self.global_mean)
                continue
            preds[i] = self.item_models[item_id].predict(
                self.item_scalers[item_id].transform(X[i : i + 1])
            )[0]
        return np.clip(preds, 0, 3)

    def predict_pathways(self, X, items, participants):
        """
        Predict participant-level scores for all three model variants:
        `('direct', 'loio', 'blend')`, each a `{participant_id: score}` dict.
        """
        trial_preds = self._predict_trials(X, items)
        loio_data = defaultdict(lambda: {"preds": [], "weights": []})
        for pred, item_id, pid in zip(trial_preds, items, participants):
            loio_data[pid]["preds"].append(pred)
            loio_data[pid]["weights"].append(self.item_weights.get(item_id, 0.1))

        loio_preds = {}
        for pid, d in loio_data.items():
            w = np.array(d["weights"])
            w = w / w.sum()
            raw = np.sum(np.array(d["preds"]) * w)
            loio_preds[pid] = float(
                np.clip(self.global_mean + (raw - self.global_mean) * (1 + self.bias_factor), 0, 3)
            )

        part_features = defaultdict(list)
        for i, pid in enumerate(participants):
            part_features[pid].append(X[i])

        direct_preds = {}
        for pid, feats_list in part_features.items():
            feats = np.array(feats_list)
            agg = np.concatenate([feats.mean(axis=0), feats.std(axis=0)])
            scaled = self.direct_scaler.transform(np.nan_to_num(agg.reshape(1, -1)))
            direct_preds[pid] = float(np.clip(self.direct_model.predict(scaled)[0], 0, 3))

        bw = self.blend_weight
        blend_preds = {}
        for pid in loio_preds:
            blended = bw * loio_preds[pid] + (1 - bw) * direct_preds.get(pid, self.global_mean)
            if self.calibrate:
                blend_preds[pid] = float(np.clip(self.calib_slope * blended + self.calib_intercept, 0, 3))
            else:
                blend_preds[pid] = float(np.clip(blended, 0, 3))

        return direct_preds, loio_preds, blend_preds

    def predict_participant(self, X, items, participants):
        """Return `{participant_id: predicted_blended_score}`."""
        _, _, blend_preds = self.predict_pathways(X, items, participants)
        return blend_preds


# =============================================================================
# Cross-Validation & Evaluation Helpers
# =============================================================================

def run_lopo(X, y, items, participants):
    """
    Run Leave-One-Participant-Out (LOPO) cross-validation.

    Returns
    -------
    predictions : dict `{participant_id: predicted_score}`
    true_means : dict `{participant_id: true_mean_score}`
    """
    unique_pids = sorted(set(participants))
    part_scores = defaultdict(list)
    for score, pid in zip(y, participants):
        part_scores[pid].append(score)
    true_means = {p: float(np.mean(s)) for p, s in part_scores.items()}

    predictions = {}
    for test_pid in unique_pids:
        train_mask = np.array([p != test_pid for p in participants])
        test_mask = ~train_mask

        model = BlendModel(alpha_item=10, alpha_direct=10, blend_weight=0.5, calibrate=True)
        model.fit(
            X[train_mask],
            y[train_mask],
            items[train_mask],
            [p for p, m in zip(participants, train_mask) if m],
        )
        pred = model.predict_participant(
            X[test_mask],
            items[test_mask],
            [p for p, m in zip(participants, test_mask) if m],
        )
        predictions[test_pid] = float(pred.get(test_pid, y[train_mask].mean()))

    return predictions, true_means


def evaluate_predictions(predictions, true_means, cutoff=APRAXIA_CUTOFF):
    """Compute regression and stroke-only apraxia classification metrics."""
    pids = sorted(predictions.keys())
    pred_arr = np.array([predictions[p] for p in pids])
    true_arr = np.array([true_means[p] for p in pids])
    errors = pred_arr - true_arr

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    r, r_p = pearsonr(pred_arr, true_arr)
    rho, rho_p = spearmanr(pred_arr, true_arr)

    stroke_pids = [p for p in pids if p.startswith("ST")]
    t_aprax = np.array([true_means[p] < cutoff for p in stroke_pids])
    p_aprax = np.array([predictions[p] < cutoff for p in stroke_pids])
    s_pred = np.array([predictions[p] for p in stroke_pids])

    auc = float(roc_auc_score(t_aprax.astype(int), -s_pred)) if len(set(t_aprax)) > 1 else float("nan")
    tp = int((t_aprax & p_aprax).sum())
    fn = int((t_aprax & ~p_aprax).sum())
    tn = int((~t_aprax & ~p_aprax).sum())
    fp = int((~t_aprax & p_aprax).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "n_participants": len(pids),
        "n_controls": sum(p.startswith("CL") for p in pids),
        "n_stroke": len(stroke_pids),
        "n_apraxic": int(t_aprax.sum()),
        "r": float(r),
        "r_p": float(r_p),
        "rho": float(rho),
        "rho_p": float(rho_p),
        "mae": mae,
        "rmse": rmse,
        "auc": auc,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": float(sens),
        "specificity": float(spec),
    }


def _filter_valid(X, scores_dict, items, participants):
    y = np.array([t if t is not None else np.nan for t in scores_dict["total"]])
    valid = ~np.isnan(y)
    return (
        X[valid],
        y[valid],
        items[valid],
        [p for p, m in zip(participants, valid) if m],
    )


def _print_report(title, predictions, true_means):
    m = evaluate_predictions(predictions, true_means)
    print("=" * 68)
    print(title)
    print("=" * 68)
    print(
        f"Cohort: {m['n_participants']} participants "
        f"({m['n_controls']} control, {m['n_stroke']} stroke [{m['n_apraxic']} with apraxia])"
    )
    print(f"  Pearson r    = {m['r']:.3f} (p = {m['r_p']:.2e})")
    print(f"  Spearman rho = {m['rho']:.3f} (p = {m['rho_p']:.2e})")
    print(f"  MAE          = {m['mae']:.3f}")
    print(f"  RMSE         = {m['rmse']:.3f}")
    print(f"  Stroke AUC   = {m['auc']:.3f}")
    print(f"  Sensitivity  = {m['sensitivity']:.2f} ({m['tp']}/{m['tp'] + m['fn']})")
    print(f"  Specificity  = {m['specificity']:.2f} ({m['tn']}/{m['tn'] + m['fp']})")
    print(f"\n{'Participant':<12} {'Group':<24} {'True':>7} {'Predicted':>10} {'Error':>8}")
    print("-" * 65)
    for pid in sorted(predictions.keys(), key=lambda p: true_means[p]):
        if pid.startswith("CL"):
            grp = "Control"
        elif true_means[pid] < APRAXIA_CUTOFF:
            grp = "Stroke (with apraxia)"
        else:
            grp = "Stroke (without apraxia)"
        err = predictions[pid] - true_means[pid]
        print(f"{pid:<12} {grp:<24} {true_means[pid]:>7.3f} {predictions[pid]:>10.3f} {err:>+8.3f}")


def main():
    parser = argparse.ArgumentParser(description="Automated kinematic praxis score prediction")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test", action="store_true", help="Evaluate on the 12 held-out test participants")
    group.add_argument("--all", action="store_true", help="Run LOPO cross-validation across all 53 participants")
    group.add_argument("--items", action="store_true", help="List all 39 tool-use pantomime items")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "modelinputs")

    if args.items:
        print("TOOL-USE PANTOMIME GESTURE ITEMS (39)")
        print("=" * 72)
        print_all_items()
        return

    if args.test:
        X_tr, y_tr, it_tr, pt_tr = _filter_valid(*load_train_data(data_dir, downsample=15))
        X_te, y_te, it_te, pt_te = _filter_valid(*load_test_data(data_dir, downsample=15))
        mdl = BlendModel(alpha_item=10, alpha_direct=10, blend_weight=0.5, calibrate=True)
        mdl.fit(X_tr, y_tr, it_tr, pt_tr)
        preds = mdl.predict_participant(X_te, it_te, pt_te)
        part_scores = defaultdict(list)
        for score, pid in zip(y_te, pt_te):
            part_scores[pid].append(score)
        true_means = {p: float(np.mean(s)) for p, s in part_scores.items()}
        _print_report("HELD-OUT TEST SET EVALUATION (n = 12, trained on n = 41)", preds, true_means)
    elif args.all:
        X, y, items, parts = _filter_valid(*load_all_data(data_dir, downsample=15))
        preds, true_means = run_lopo(X, y, items, parts)
        _print_report("FULL-COHORT LOPO CROSS-VALIDATION (N = 53)", preds, true_means)
    else:
        X, y, items, parts = _filter_valid(*load_train_data(data_dir, downsample=15))
        preds, true_means = run_lopo(X, y, items, parts)
        _print_report("TRAINING SET LOPO CROSS-VALIDATION (n = 41)", preds, true_means)


if __name__ == "__main__":
    main()
