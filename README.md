# Automated Kinematic Scoring of Tool-Use Pantomime Gestures in Limb Apraxia

This repository contains the de-identified electromagnetic motion-capture dataset, kinematic feature-extraction and modeling code, and vector schematic templates needed to reproduce all statistical results and all four figures in the accompanying manuscript.

---

## Overview

Limb apraxia is commonly assessed by having patients pantomime the use of everyday tools (e.g., *hammer*, *corkscrew*, *toothbrush*) while trained raters score each gesture across hand posture, arm posture/trajectory, and amplitude/timing. Here we predict continuous participant-level praxis scores ($0\text{--}3$ scale; clinical apraxia cutoff $< 2.52$) directly from 8-sensor $100\text{ Hz}$ upper-limb motion-capture recordings across **39 tool-use gestures** and **53 participants** ($10$ healthy controls, $43$ left-hemisphere stroke survivors; $1{,}950$ content-valid trials):

1. **Scale-Free Kinematic Representation ($46$ features per trial):** Within-trial normalized and hand-length-scaled features spanning four motor domains:
   - *Hand Posture & Distal Individuation* ($19$ features; indices `0–18`)
   - *Arm Posture & Inter-Joint Coordination* ($13$ features; indices `19–31`)
   - *Amplitude, Smoothness & Timing* ($13$ features; indices `32–44`)
   - *Motion-Type Specific Kinematics* ($1$ feature; index `45`)
2. **Two-Pathway Ridge Blend (`BlendModel`):**
   - **Item-level pathway:** $39$ item-specific $L_2$-regularized Ridge regressions ($\alpha = 10$) predicting trial-level gesture scores, aggregated across items per participant via inverse cross-validated MSE reliability weights and rescaled around the training-fold global mean.
   - **Participant-level pathway:** A single $L_2$-regularized Ridge regression ($\alpha = 10$) predicting overall praxis score from participant-level summary kinematics (concatenated mean and SD across trials; $92$ inputs).
   - **Blended prediction:** Equal-weight ($0.5 / 0.5$) average of the two pathways followed by linear variance-matching calibration fit strictly within each training fold.

---

## Quick Start (Reproduce Everything in One Command)

### 1. Python Environment

Requires **Python 3.10+** and standard scientific Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Optional — Vector Schematic Headers)*: Each figure combines vector schematic diagrams (`figures/templates/figure{1..4}.svg`) with Matplotlib data panels to produce both editable `.svg` files (`figures/figure{1..4}.svg`) and $350\text{ DPI}$ publication `.png` files (`figures/fig{1..4}.png`). To rasterize the composite `.svg` files to `.png`, install **`librsvg`** (`rsvg-convert`) or **Inkscape**:

- **macOS (Homebrew):** `brew install librsvg`
- **Ubuntu / Debian:** `sudo apt-get install librsvg2-bin`

*(If neither `rsvg-convert` nor `inkscape` is installed, the pipeline still writes the complete composite `.svg` files and automatically exports standalone Matplotlib `.png` data panels, or you can set `NO_HEADER=1` explicitly.)*

### 2. Run Everything

Run the unified reproduction script from the repository root:

```bash
python reproduce_all.py
```

`reproduce_all.py` clears any cached intermediates, extracts all $46$ kinematic features from raw $100\text{ Hz}$ sensor recordings in `modelinputs/`, prints the complete statistical report for every section of the paper, and generates all four manuscript figures:

| Output File | Description |
| :--- | :--- |
| `figures/fig1.png` / `figures/figure1.svg` | **Figure 1:** Task setup, manual coding scheme, group praxis/component distributions ($N = 53$), item score heatmaps, and item-level apraxia ROC AUCs |
| `figures/fig2.png` / `figures/figure2.svg` | **Figure 2:** Model architecture, sample 3D kinematic traces (*Corkscrew*), LOPO feature & item weights, and training cross-validation ($n = 41$) |
| `figures/fig3.png` / `figures/figure3.svg` | **Figure 3:** Out-of-sample held-out test evaluation ($n = 12$) and feature/item decompositions of the three test patients with apraxia |
| `figures/fig4.png` / `figures/figure4.svg` | **Figure 4:** Full-cohort LOPO cross-validation ($N = 53$) and stability of learned feature and item weights ($n = 41$ vs. $N = 53$) |

Useful flags for `reproduce_all.py`:

```bash
python reproduce_all.py --stats-only   # Print all manuscript statistics without rendering figures
python reproduce_all.py --use-cache    # Re-render figures quickly from cached fold outputs in figures/
python reproduce_all.py --no-header    # Export Matplotlib subplot PNGs without SVG schematic headers
```

---

## Summary of Reproduced Manuscript Results

| Evaluation Split | Pathway | Pearson $r$ | MAE | Stroke ROC AUC | Sensitivity | Specificity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Training LOPO ($n = 41$)**<br>*(7 control, 34 stroke; 14 with apraxia)* | Participant<br>Item<br>**Blended** | $0.709$<br>$0.779$<br>**$0.833$** | $0.176$<br>$0.155$<br>**$0.146$** | $0.850$<br>$0.864$<br>**$0.914$** | $0.71$ ($10/14$)<br>$0.57$ ($8/14$)<br>**$0.71$ ($10/14$)** | $0.80$ ($16/20$)<br>$0.95$ ($19/20$)<br>**$0.95$ ($19/20$)** |
| **Held-Out Test ($n = 12$)**<br>*(3 control, 9 stroke; 3 with apraxia)* | Participant<br>Item<br>**Blended** | $0.760$<br>$0.484$<br>**$0.776$** | $0.127$<br>$0.159$<br>**$0.123$** | $1.000$<br>$0.722$<br>**$0.889$** | $1.00$ ($3/3$)<br>$0.33$ ($1/3$)<br>**$0.67$ ($2/3$)** | $1.00$ ($6/6$)<br>$1.00$ ($6/6$)<br>**$1.00$ ($6/6$)** |
| **Full-Cohort LOPO ($N = 53$)**<br>*(10 control, 43 stroke; 17 with apraxia)* | Participant<br>Item<br>**Blended** | $0.757$<br>$0.766$<br>**$0.870$** | $0.149$<br>$0.158$<br>**$0.126$** | $0.898$<br>$0.885$<br>**$0.946$** | $0.82$ ($14/17$)<br>$0.59$ ($10/17$)<br>**$0.82$ ($14/17$)** | $0.85$ ($22/26$)<br>$0.96$ ($25/26$)<br>**$0.92$ ($24/26$)** |

---

## Repository Structure

```text
.
├── reproduce_all.py          # Single-script runner: reproduces all statistics and Figures 1-4
├── model.py                  # 46-feature extraction, BlendModel, LOPO CV, and CLI evaluation
├── item_info.py              # Metadata and kinematic motion categories for the 39 tool items
├── plot_fig1.py              # Generates Figure 1 (manual praxis coding & item discriminability)
├── plot_fig2.py              # Generates Figure 2 (training LOPO CV, traces, and model weights)
├── plot_fig3.py              # Generates Figure 3 (held-out test evaluation & patient profiles)
├── plot_fig4.py              # Generates Figure 4 (full-cohort LOPO CV & weight stability)
├── svg_composer.py           # Composes Matplotlib panels into vector schematic SVG templates
├── requirements.txt          # Python package dependencies
├── LICENSE                   # MIT License
├── figures/
│   ├── fig1.png - fig4.png       # High-resolution manuscript figures (350 DPI)
│   ├── figure1.svg - figure4.svg # Editable composite vector figures
│   └── templates/                # Vector schematic templates (figure1.svg - figure4.svg)
└── modelinputs/              # De-identified 100 Hz trial kinematics & manual component scores
    ├── metadata.json         # Trial metadata and manual scores for all 1,991 recorded trials
    └── trial*.npy            # Movement-segmented 100 Hz sensor arrays (1,991 files)
```

---

## Running Individual Scripts

Each analysis and figure script can also be run independently:

```bash
# Evaluate BlendModel on individual splits (prints metrics + per-participant table)
python model.py          # Training set LOPO cross-validation (n = 41)
python model.py --test   # Out-of-sample held-out test set evaluation (n = 12)
python model.py --all    # Full-cohort LOPO cross-validation (N = 53)

# List all 39 tool-use pantomime items and their kinematic motion categories
python item_info.py

# Render individual manuscript figures
python plot_fig1.py
python plot_fig2.py
python plot_fig3.py
python plot_fig4.py
```

---

## Dataset Format (`modelinputs/`)

### `modelinputs/metadata.json`

A JSON list of $1{,}991$ trial records with fields:
- `trial_name`: Filename stem corresponding to `modelinputs/<trial_name>.npy` (e.g., `"trial0"`).
- `participant`: De-identified participant code (`CL*` = healthy control, $n = 10$; `ST*` = left-hemisphere stroke survivor, $n = 43$). The $12$ held-out test participants are listed in `TEST_PARTICIPANTS` in [`model.py`](model.py).
- `item`: Integer tool-use item ID (`1`–`39`; mapped to names and motion categories in [`item_info.py`](item_info.py)).
- `experiment`: Sub-study protocol code.
- `score`: Nested manual rating `[[content, HP, AP, AT]]`:
  - `content` $\in \{0, 1\}$: Whether the participant pantomimed the recognizable target tool (`0` and unscorable `NaN` trials are excluded from kinematic quality scoring, leaving $1{,}950$ valid `content = 1` trials).
  - `HP`, `AP`, `AT` $\in \{0, 1\}$: Binary accuracy ratings for **Hand Posture**, **Arm Posture & Trajectory**, and **Amplitude & Timing**. Their sum (`0`–`3`) is the trial praxis score, and a participant's mean across `content = 1` items is their overall praxis score.

### `modelinputs/trial*.npy`

Each `.npy` file stores a movement-segmented $100\text{ Hz}$ electromagnetic motion-capture array (Ascension 3D Guidance trakSTAR; ipsilateral left upper limb) of shape `(T, 49)`:
- **Column `0`:** Time in seconds ($100\text{ Hz}$ sampling rate).
- **Columns `1:25`:** 3D sensor positions $(X, Y, Z)$ in meters for the $8$ landmarks in order:
  1. `Thumb` (distal phalanx) — cols `1:4`
  2. `Index` (distal phalanx) — cols `4:7`
  3. `Middle` (distal phalanx) — cols `7:10`
  4. `Knuckle` (index MCP2 joint) — cols `10:13`
  5. `Wrist` (dorsal radiocarpal joint) — cols `13:16`
  6. `Elbow` (lateral epicondyle) — cols `16:19`
  7. `Left Shoulder` (ipsilateral acromion) — cols `19:22`
  8. `Right Shoulder` (contralateral acromion) — cols `22:25`
- **Columns `25:49`:** Corresponding 3D sensor velocities $(v_x, v_y, v_z)$ in $\text{m/s}$ for the same $8$ landmarks ($Z$ = vertical axis; $(X, Y)$ = horizontal plane).

---

## License

Released under the [MIT License](LICENSE).
