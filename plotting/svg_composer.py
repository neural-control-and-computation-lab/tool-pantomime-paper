"""
Compose complete manuscript figures (editable `.svg` + high-resolution `.png`)
by embedding rendered Matplotlib data panels into the vector schematic templates
in `figures/templates/figure{1..4}.svg` and aligning headers and family brackets.

Set `NO_HEADER=1` in the environment to export standalone Matplotlib subplot PNGs
(`figures/fig{1..4}.png`) without vector header composition.
"""

import base64
import copy
import io
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SVG_NS = "http://www.w3.org/2000/svg"
INK_NS = "http://www.inkscape.org/namespaces/inkscape"
SODI_NS = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
XLINK_NS = "http://www.w3.org/1999/xlink"

for _prefix, _uri in (
    ("", SVG_NS),
    ("inkscape", INK_NS),
    ("sodipodi", SODI_NS),
    ("xlink", XLINK_NS),
):
    ET.register_namespace(_prefix, _uri)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
TEMPLATES_DIR = os.path.join(FIGURES_DIR, "templates")

# Default rasterization resolution (38 cm @ 350 DPI = 5236 px wide)
EXPORT_DPI = int(os.environ.get("SVG_EXPORT_DPI", "350"))


class SubplotToSvgMapper:
    """Linear coordinate map from Matplotlib display/data space to SVG viewBox units."""

    def __init__(self, fig, rend, tb_in, pad_in, svg_x, svg_y, svg_w, svg_h):
        self.fig = fig
        self.rend = rend
        self.x0_in = tb_in.x0 - pad_in
        self.y1_in = tb_in.y1 + pad_in
        self.w_in = tb_in.width + 2.0 * pad_in
        self.h_in = tb_in.height + 2.0 * pad_in
        self.svg_x = svg_x
        self.svg_y = svg_y
        self.svg_w = svg_w
        self.svg_h = svg_h

    def disp_to_svg(self, dx, dy):
        x_in = dx / self.fig.dpi
        y_in = dy / self.fig.dpi
        u = (x_in - self.x0_in) / self.w_in
        v = (self.y1_in - y_in) / self.h_in
        return self.svg_x + u * self.svg_w, self.svg_y + v * self.svg_h

    def data_to_svg(self, ax, x_data, y_data):
        dx, dy = ax.transData.transform((x_data, y_data))
        return self.disp_to_svg(dx, dy)

    def ax_box_svg(self, ax, tight=False):
        bb = ax.get_tightbbox(self.rend) if tight else ax.get_window_extent(self.rend)
        x0, y1 = self.disp_to_svg(bb.x0, bb.y0)
        x1, y0 = self.disp_to_svg(bb.x1, bb.y1)
        return x0, y0, x1, y1


def _render_and_inject_pixel_matched(fig, root, img_id, export_dpi=EXPORT_DPI, x_override=None, w_override=None):
    """
    Render `fig` using 2x supersampled Lanczos downsampling matched 1-to-1 to the
    exact output pixel grid at `export_dpi`, eliminating non-integer resampling blur
    between the Matplotlib panel and vector schematic elements.
    """
    el = root.find(f'.//*[@id="{img_id}"]')
    if el is None:
        raise KeyError(f'Missing <image id="{img_id}"> in SVG template')

    px_mm = 25.4 / float(export_dpi)
    raw_x = float(el.attrib["x"]) if x_override is None else float(x_override)
    raw_y = float(el.attrib["y"])
    raw_w = float(el.attrib["width"]) if w_override is None else float(w_override)

    # Snap image placement & width to exact output pixel boundaries
    svg_x = round(raw_x / px_mm) * px_mm
    svg_y = round(raw_y / px_mm) * px_mm
    target_w = max(1, round(raw_w / px_mm))
    svg_w = target_w * px_mm

    # Measure tight bounding box on a reference draw pass
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(rend)
    pad = float(plt.rcParams.get("savefig.pad_inches", 0.1))
    full_w_in = tb.width + 2.0 * pad

    # Render at 2x target pixel density, then Lanczos downsample to exact 1x target_w
    dpi_2x = max(300.0, (2.0 * target_w) / full_w_in)
    buf_hi = io.BytesIO()
    fig.savefig(buf_hi, format="png", dpi=dpi_2x, bbox_inches="tight", facecolor="white")
    im_hi = Image.open(io.BytesIO(buf_hi.getvalue())).convert("RGB")
    target_h = max(1, round(target_w * (im_hi.height / im_hi.width)))
    im_exact = im_hi.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
    svg_h = target_h * px_mm

    buf_out = io.BytesIO()
    im_exact.save(buf_out, format="PNG", optimize=True)
    png_bytes = buf_out.getvalue()

    # Re-grab renderer after savefig (savefig temporarily alters figure dpi)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(rend)

    el.attrib["x"] = f"{svg_x:.6f}"
    el.attrib["y"] = f"{svg_y:.6f}"
    el.attrib["width"] = f"{svg_w:.6f}"
    el.attrib["height"] = f"{svg_h:.6f}"
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    href_keys = [k for k in el.attrib if k.endswith("href")]
    if href_keys:
        for k in href_keys:
            el.attrib[k] = data_uri
    else:
        el.attrib[f"{{{XLINK_NS}}}href"] = data_uri

    return png_bytes, rend, tb, pad, svg_x, svg_y, svg_w, svg_h


def _set_tspan_text(root, tspan_id, new_text):
    el = root.find(f'.//*[@id="{tspan_id}"]')
    if el is None:
        raise KeyError(f'Missing <tspan id="{tspan_id}"> in SVG template')
    el.text = new_text


def _set_two_line_label_y(text_el, y_mid, line_gap=6.17361):
    """Center a two-line SVG `<text>` block vertically on `y_mid`."""
    y1 = y_mid - 1.55
    y2 = y1 + line_gap
    text_el.attrib["y"] = f"{y1:.5f}"
    tspans = [c for c in text_el if c.tag.endswith("tspan")]
    if len(tspans) >= 1:
        tspans[0].attrib["y"] = f"{y1:.5f}"
    if len(tspans) >= 2:
        tspans[1].attrib["y"] = f"{y2:.5f}"


def _update_canvas_height(root, bg_rect_id, view_w, new_h, unit="mm", export_dpi=EXPORT_DPI):
    """Adjust SVG height, viewBox, and background rectangle snapped to integer output pixels."""
    px_mm = 25.4 / float(export_dpi)
    snapped_w = round(view_w / px_mm) * px_mm
    snapped_h = round(new_h / px_mm) * px_mm
    if unit == "cm":
        root.attrib["width"] = f"{snapped_w / 10.0:.6f}cm"
        root.attrib["height"] = f"{snapped_h / 10.0:.6f}cm"
    else:
        root.attrib["width"] = f"{snapped_w:.6f}mm"
        root.attrib["height"] = f"{snapped_h:.6f}mm"
    root.attrib["viewBox"] = f"0 0 {view_w:g} {snapped_h:.4f}"
    bg = root.find(f'.//*[@id="{bg_rect_id}"]')
    if bg is not None:
        bg.attrib["x"] = "0"
        bg.attrib["y"] = "0"
        bg.attrib["width"] = f"{view_w:g}"
        bg.attrib["height"] = f"{snapped_h:.4f}"
        bg.attrib["style"] = "fill:#ffffff;fill-opacity:1;stroke:none"


def _normalize_svg_markers(root):
    """Ensure Inkscape marker strokes use unitless user units so arrowheads do not scale with export DPI."""
    for mk in root.iter(f"{{{SVG_NS}}}marker"):
        for p in mk.iter(f"{{{SVG_NS}}}path"):
            st = p.attrib.get("style", "")
            if "stroke-width:1pt" in st:
                p.attrib["style"] = st.replace("stroke-width:1pt", "stroke-width:1.33333")


def _flatten_png_to_opaque_rgb(png_path):
    """Composite any semi-transparent edge pixels over pure white (#ffffff) and save as 24-bit RGB."""
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(png_path)
    if im.mode == "RGB":
        return
    im = im.convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    alpha = arr[..., 3:4] / 255.0
    rgb = np.clip(arr[..., :3] * alpha + 255.0 * (1.0 - alpha), 0, 255).round().astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(png_path, format="PNG")


def _write_svg_and_png(tree, svg_out, png_out, fallback_png_bytes=None, dpi=EXPORT_DPI):
    _normalize_svg_markers(tree.getroot())
    tree.write(svg_out, encoding="utf-8", xml_declaration=True)
    if shutil.which("rsvg-convert"):
        subprocess.run(
            [
                "rsvg-convert",
                "--background-color=white",
                "-u",
                "-d",
                str(dpi),
                "-p",
                str(dpi),
                svg_out,
                "-o",
                png_out,
            ],
            check=True,
        )
        _flatten_png_to_opaque_rgb(png_out)
    elif shutil.which("inkscape"):
        subprocess.run(
            [
                "inkscape",
                svg_out,
                "--export-background=#ffffff",
                "--export-background-opacity=1.0",
                f"--export-dpi={dpi}",
                f"--export-filename={png_out}",
            ],
            check=True,
        )
        _flatten_png_to_opaque_rgb(png_out)
    else:
        try:
            import resvg_py

            with open(svg_out, "r", encoding="utf-8") as f_svg:
                svg_str = f_svg.read()
            png_bytes = resvg_py.svg_to_bytes(svg_string=svg_str, dpi=dpi, background="#ffffff")
            with open(png_out, "wb") as f_png:
                f_png.write(png_bytes)
            _flatten_png_to_opaque_rgb(png_out)
        except Exception:
            if fallback_png_bytes is not None:
                with open(png_out, "wb") as f:
                    f.write(fallback_png_bytes)
                _flatten_png_to_opaque_rgb(png_out)
            else:
                raise


def _move_svg_text_y(root, tid, y_val, x_val=None):
    el = root.find(f'.//*[@id="{tid}"]')
    if el is None:
        return
    if x_val is not None:
        el.attrib["x"] = f"{x_val:.5f}"
    el.attrib["y"] = f"{y_val:.5f}"
    for ts in el:
        if x_val is not None and "x" in ts.attrib:
            ts.attrib["x"] = f"{x_val:.5f}"
        if "y" in ts.attrib:
            ts.attrib["y"] = f"{y_val:.5f}"


def _scale_family_label(tel, fs_px=5.35):
    old = "4.93889px"
    new = f"{fs_px:.5f}px"
    if "style" in tel.attrib:
        tel.attrib["style"] = tel.attrib["style"].replace(old, new)
    for ts in tel:
        if "style" in ts.attrib:
            ts.attrib["style"] = ts.attrib["style"].replace(old, new)


# ============================================================================
# FIGURE 1
# ============================================================================
def compose_fig1(fig, out_png, dpi=EXPORT_DPI):
    tree = ET.parse(os.path.join(TEMPLATES_DIR, "figure1.svg"))
    root = tree.getroot()
    png_bytes, _, _, _, _, sy, _, sh = _render_and_inject_pixel_matched(fig, root, "image1-7", export_dpi=dpi)
    _update_canvas_height(root, "rect10", 380, max(380.0, sy + sh + 3.0), unit="cm", export_dpi=dpi)
    for tid in ("text1-7", "text1-7-5", "text1-7-5-1"):
        _move_svg_text_y(root, tid, 110.2)
    out_svg = os.path.join(FIGURES_DIR, "figure1.svg")
    _write_svg_and_png(tree, out_svg, out_png, fallback_png_bytes=png_bytes, dpi=dpi)
    print(f"Saved {out_svg} and {out_png}")


# ============================================================================
# FIGURE 2
# ============================================================================
def compose_fig2(fig, sfL, axl, axr, axD, axE, axF, out_png, dpi=EXPORT_DPI):
    tree = ET.parse(os.path.join(TEMPLATES_DIR, "figure2.svg"))
    root = tree.getroot()
    png_bytes, rend, tb, pad, sx, sy, sw, sh = _render_and_inject_pixel_matched(
        fig, root, "image1-3", export_dpi=dpi, x_override=0.0, w_override=380.0
    )
    new_total_h = max(420.0, sy + sh + 3.0)
    _update_canvas_height(root, "rect1", 380, new_total_h, unit="cm", export_dpi=dpi)
    mp = SubplotToSvgMapper(fig, rend, tb, pad, sx, sy, sw, sh)

    # 1. Ensure Panel 2A pipeline labels match the 46-feature equal-weight blend
    _set_tspan_text(root, "tspan87-7-8-2-5", "Kinematic Features")
    _set_tspan_text(root, "tspan18", "All 46 scale-free ")
    _set_tspan_text(root, "tspan20", "features per trial")
    _set_tspan_text(root, "tspan29", "Equal-weight mean ")
    _set_tspan_text(root, "tspan20-5-6-5", "Leave-one-participant-out cross validation (41 folds)")
    _set_tspan_text(root, "tspan87-7-8-2-7", "7 control + 34 stroke")
    _set_tspan_text(root, "tspan6-1", "Without Apraxia")
    _set_tspan_text(root, "tspan6-1-2", "With Apraxia")

    # 1b. Align vector 3-line trace headers cleanly below 'Item: Corkscrew' and above 3D axes
    axes_3d = [a for a in sfL.axes if getattr(a, "name", "") == "3d"]
    if len(axes_3d) == 3:
        t_boxes = []
        for a3 in axes_3d:
            bb = a3.title.get_window_extent(rend)
            tx0, ty1 = mp.disp_to_svg(bb.x0, bb.y0)
            tx1, ty0 = mp.disp_to_svg(bb.x1, bb.y1)
            t_boxes.append((tx0, ty0, tx1, ty1))
        base_y1 = 143.0
        r8 = root.find('.//*[@id="rect8"]')
        if r8 is not None:
            mx0 = min(b[0] for b in t_boxes) - 3.0
            my0 = min(base_y1 - 5.0, min(b[1] for b in t_boxes) - 1.5)
            mx1 = max(b[2] for b in t_boxes) + 3.0
            my1 = max(b[3] for b in t_boxes) + 1.2
            r8.attrib["x"] = f"{mx0:.5f}"
            r8.attrib["y"] = f"{my0:.5f}"
            r8.attrib["width"] = f"{mx1 - mx0:.5f}"
            r8.attrib["height"] = f"{my1 - my0:.5f}"
            r8.attrib["style"] = "fill:#ffffff;fill-opacity:1;stroke:none"
        for tid, a3 in zip(("text6", "text6-6", "text6-6-1"), axes_3d):
            b3 = mp.ax_box_svg(a3, tight=False)
            xc = 0.5 * (b3[0] + b3[2])
            tel = root.find(f'.//*[@id="{tid}"]')
            if tel is not None:
                tel.attrib["x"] = f"{xc:.5f}"
                tel.attrib["y"] = f"{base_y1:.5f}"
                tsp = [c for c in tel if c.tag.endswith("tspan")]
                for idx_t, ts in enumerate(tsp):
                    ts.attrib["x"] = f"{xc:.5f}"
                    ts.attrib["y"] = f"{base_y1 + idx_t * 7.6:.5f}"
                    if idx_t == 0 and "5.64444px" in ts.attrib.get("style", ""):
                        ts.attrib["style"] = ts.attrib["style"].replace("5.64444px", "5.25000px")

    # 2. Align Mean / SD headers cleanly over axl and axr centers
    axl_x0, axl_y0, axl_x1, _ = mp.ax_box_svg(axl, tight=False)
    axr_x0, axr_y0, axr_x1, _ = mp.ax_box_svg(axr, tight=False)
    r86 = root.find('.//*[@id="rect8-6"]')
    if r86 is not None:
        r86.attrib["x"] = f"{axl_x0 - 4.0:.5f}"
        r86.attrib["y"] = f"{axl_y0 - 12.0:.5f}"
        r86.attrib["width"] = f"{(axr_x1 - axl_x0) + 8.0:.5f}"
        r86.attrib["height"] = "11.40000"
        r86.attrib["style"] = "fill:#ffffff;fill-opacity:1;stroke:none"
    for rid, x0, y0, x1 in (
        ("rect8-5", axl_x0, axl_y0, axl_x1),
        ("rect8-5-0", axr_x0, axr_y0, axr_x1),
    ):
        rm = root.find(f'.//*[@id="{rid}"]')
        if rm is not None:
            rm.attrib["x"] = f"{x0 - 2.0:.5f}"
            rm.attrib["y"] = f"{y0 - 9.0:.5f}"
            rm.attrib["width"] = f"{(x1 - x0) + 4.0:.5f}"
            rm.attrib["height"] = "8.50000"
            rm.attrib["style"] = "fill:#ffffff;fill-opacity:1;stroke:none"

    def _center_svg_header(tid, xc, yc):
        tel = root.find(f'.//*[@id="{tid}"]')
        if tel is None:
            return
        tel.attrib["x"] = f"{xc:.5f}"
        tel.attrib["y"] = f"{yc:.5f}"
        for k in ("end", "start"):
            tel.attrib["style"] = (
                tel.attrib.get("style", "")
                .replace(f"text-align:{k}", "text-align:center")
                .replace(f"text-anchor:{k}", "text-anchor:middle")
            )
        if "text-anchor:middle" not in tel.attrib.get("style", ""):
            tel.attrib["style"] = tel.attrib.get("style", "") + ";text-align:center;text-anchor:middle"
        for ts in tel:
            ts.attrib["x"] = f"{xc:.5f}"
            ts.attrib["y"] = f"{yc:.5f}"
            ts.attrib["style"] = (
                ts.attrib.get("style", "")
                .replace("text-align:end", "text-align:center")
                .replace("text-anchor:end", "text-anchor:middle")
                .replace("text-align:start", "text-align:center")
                .replace("text-anchor:start", "text-anchor:middle")
            )

    _center_svg_header("text6-6-1-5", 0.5 * (axl_x0 + axl_x1), axl_y0 - 3.0)
    _center_svg_header("text6-6-1-5-2", 0.5 * (axr_x0 + axr_x1), axr_y0 - 3.0)

    # 3. Position right-hand kinematic family brackets across the 46 features
    x_bar = axr_x1 + 1.8
    x_txt = x_bar + 1.9

    _, hp_y0 = mp.data_to_svg(axr, 0, -0.35)
    _, hp_y1 = mp.data_to_svg(axr, 0, 18.35)
    _, ap_y0 = mp.data_to_svg(axr, 0, 18.65)
    _, ap_y1 = mp.data_to_svg(axr, 0, 31.35)
    _, at_y0 = mp.data_to_svg(axr, 0, 31.65)
    _, at_y1 = mp.data_to_svg(axr, 0, 44.35)
    _, mt_y_mid = mp.data_to_svg(axr, 0, 45.0)

    p_hp = root.find('.//*[@id="path17-3-1-4"]')
    p_ap = root.find('.//*[@id="path17-3-1-4-0"]')
    p_at = root.find('.//*[@id="path17-3-1"]')
    p_hp.attrib["d"] = f"m {x_bar:.5f},{hp_y0:.5f} v {hp_y1 - hp_y0:.5f}"
    p_ap.attrib["d"] = f"m {x_bar:.5f},{ap_y0:.5f} v {ap_y1 - ap_y0:.5f}"
    p_at.attrib["d"] = f"m {x_bar:.5f},{at_y0:.5f} v {at_y1 - at_y0:.5f}"

    for tid, y_mid in (
        ("text9-9", 0.5 * (hp_y0 + hp_y1)),
        ("text9-8", 0.5 * (ap_y0 + ap_y1)),
        ("text9", 0.5 * (at_y0 + at_y1) - 1.2),
    ):
        tel = root.find(f'.//*[@id="{tid}"]')
        _scale_family_label(tel, 4.94)
        tel.attrib["x"] = f"{x_txt:.5f}"
        for ts in tel:
            ts.attrib["x"] = f"{x_txt:.5f}"
        _set_two_line_label_y(tel, y_mid, line_gap=6.17)

    # Add 4th kinematic family bracket ('Motion-specific', row 45) symmetrically centered
    layer = [c for c in root if c.tag.endswith("g")][0]
    ET.SubElement(
        layer,
        f"{{{SVG_NS}}}path",
        {
            "id": "path17-motion",
            "style": "fill:#7a7a84;fill-opacity:1;stroke:#7a7a84;stroke-width:0.8;stroke-miterlimit:6;stroke-dasharray:none;stroke-opacity:1",
            "d": f"m {x_bar:.5f},{mt_y_mid - 1.0:.5f} v 2.00000",
        },
    )
    t_mt = copy.deepcopy(root.find('.//*[@id="text9"]'))
    t_mt.attrib["id"] = "text9-motion"
    t_mt.attrib["style"] = t_mt.attrib["style"].replace("#69bbad", "#7a7a84")
    mt_tspans = [c for c in t_mt if c.tag.endswith("tspan")]
    mt_tspans[0].attrib["id"] = "tspan_mt_1"
    mt_tspans[0].attrib["style"] = mt_tspans[0].attrib["style"].replace("#69bbad", "#7a7a84")
    mt_tspans[0].text = "Motion-"
    mt_tspans[1].attrib["id"] = "tspan_mt_2"
    mt_tspans[1].attrib["style"] = mt_tspans[1].attrib["style"].replace("#69bbad", "#7a7a84")
    mt_tspans[1].text = "specific"
    t_mt.attrib["x"] = f"{x_txt:.5f}"
    for ts in mt_tspans:
        ts.attrib["x"] = f"{x_txt:.5f}"
    _set_two_line_label_y(t_mt, mt_y_mid + 0.5, line_gap=5.4)
    layer.append(t_mt)

    # 4. Align Panel E, F, G headers and r / AUC legend cleanly to bottom-row axes
    bD_w = mp.ax_box_svg(axD, tight=False)
    bF_w = mp.ax_box_svg(axF, tight=False)
    _move_svg_text_y(root, "text1-7-2-1", bD_w[1] - 6.0)
    _move_svg_text_y(root, "text1-7-2-1-2", bD_w[1] - 6.0)
    _move_svg_text_y(root, "text1-7-2-1-2-4", bF_w[1] - 6.2)

    eE_w = mp.ax_box_svg(axE, tight=False)
    eE_t = mp.ax_box_svg(axE, tight=True)
    g16 = root.find('.//*[@id="g16"]')
    if g16 is not None:
        cx = 0.5 * (eE_w[0] + eE_w[2])
        top_y = eE_t[3] + 2.0
        g16.attrib["transform"] = f"translate({cx - 334.7:.5f},{top_y - 308.93869:.5f})"

    out_svg = os.path.join(FIGURES_DIR, "figure2.svg")
    _write_svg_and_png(tree, out_svg, out_png, fallback_png_bytes=png_bytes, dpi=dpi)
    print(f"Saved {out_svg} and {out_png}")


# ============================================================================
# FIGURE 3
# ============================================================================
def compose_fig3(fig, axA, axP, axM, axB, axC, out_png, dpi=EXPORT_DPI):
    tree = ET.parse(os.path.join(TEMPLATES_DIR, "figure3.svg"))
    root = tree.getroot()
    png_bytes, rend, tb, pad, sx, sy, sw, sh = _render_and_inject_pixel_matched(
        fig, root, "image1-5", export_dpi=dpi, x_override=2.5, w_override=380.0
    )
    new_total_h = sy + sh + 3.5
    _update_canvas_height(root, "rect1", 385, new_total_h, unit="mm", export_dpi=dpi)
    mp = SubplotToSvgMapper(fig, rend, tb, pad, sx, sy, sw, sh)

    # 1. Align Panel 3A pipeline text with dashed container left edge
    _set_tspan_text(root, "tspan87-7-8-2-5-3", "Kinematic Features")
    _set_tspan_text(root, "tspan18-3", "All 46 scale-free ")
    _set_tspan_text(root, "tspan20-90", "features per trial")
    _set_tspan_text(root, "tspan29-04", "Equal-weight mean ")
    _set_tspan_text(root, "tspan20-5-6-5-8", "Model trained on full training set (41 participants)")
    _set_tspan_text(root, "tspan87-7-8-2-7-6", "7 control + 34 stroke")
    t_sub = root.find('.//*[@id="text11-9-8-8-5-1-8-7-4-6-4"]')
    if t_sub is not None:
        x_left = f"{284.4 / 0.96302773:.5f}"
        t_sub.attrib["x"] = x_left
        t_sub.attrib["style"] = t_sub.attrib.get("style", "") + ";text-align:start;text-anchor:start"
        for ts in t_sub:
            ts.attrib["x"] = x_left
            ts.attrib["style"] = (
                ts.attrib.get("style", "")
                .replace("text-align:center", "text-align:start")
                .replace("text-anchor:middle", "text-anchor:start")
            )

    # 2. Align Panel B, C, D, E, F headers and r / AUC legend to subplot boxes
    bA_t = mp.ax_box_svg(axA, tight=True)
    bP_t = mp.ax_box_svg(axP, tight=True)
    bP_w = mp.ax_box_svg(axP, tight=False)
    bM_w = mp.ax_box_svg(axM, tight=False)
    bB_t = mp.ax_box_svg(axB, tight=True)
    bC_t = mp.ax_box_svg(axC, tight=True)

    _move_svg_text_y(root, "text1-7-0", 121.20, bA_t[0])
    _move_svg_text_y(root, "text1-7-0-2", 121.20, bP_t[0])
    _move_svg_text_y(root, "text1-7-0-2-4", 121.20, min(bC_t[0], 251.24417))
    _move_svg_text_y(root, "text1-7-0-2-9", bM_w[1] - 6.2, bP_t[0])
    _move_svg_text_y(root, "text1-7-0-2-4-3", bB_t[1] - 5.2, bA_t[0])

    g16 = root.find('.//*[@id="g16"]')
    if g16 is not None:
        cx = 0.5 * (bP_w[0] + bP_w[2])
        top_y = bP_t[3] + 2.4
        g16.attrib["transform"] = f"translate({cx - 334.7:.5f},{top_y - 308.93869:.5f})"

    out_svg = os.path.join(FIGURES_DIR, "figure3.svg")
    _write_svg_and_png(tree, out_svg, out_png, fallback_png_bytes=png_bytes, dpi=dpi)
    print(f"Saved {out_svg} and {out_png}")


# ============================================================================
# FIGURE 4
# ============================================================================
def compose_fig4(fig, axG, axH, axD, axE, axF, out_png, dpi=EXPORT_DPI):
    tree = ET.parse(os.path.join(TEMPLATES_DIR, "figure4.svg"))
    root = tree.getroot()
    png_bytes, rend, tb, pad, sx, sy, sw, sh = _render_and_inject_pixel_matched(
        fig, root, "image1-2", export_dpi=dpi, x_override=2.5, w_override=300.0
    )
    new_total_h = max(450.0, sy + sh + 2.5)
    _update_canvas_height(root, "rect4", 305, new_total_h, unit="cm", export_dpi=dpi)
    mp = SubplotToSvgMapper(fig, rend, tb, pad, sx, sy, sw, sh)

    # 1. Ensure Panel 4A pipeline text and Panel 4B/4C legends match the 46-feature models
    _set_tspan_text(root, "tspan87-7-8-2-5-3-7", "Kinematic Features")
    _set_tspan_text(root, "tspan18-3-0", "All 46 scale-free ")
    _set_tspan_text(root, "tspan20-90-9", "features per trial")
    _set_tspan_text(root, "tspan29-04-0", "Equal-weight mean ")
    _set_tspan_text(root, "tspan20-5-6-5-8-9", "Leave-one-participant-out cross validation (53 folds)")
    _set_tspan_text(root, "tspan87-7-8-2-7-6-3", "10 control + 43 stroke")

    _set_tspan_text(root, "tspan1-6", "46 features (N = 41)")
    _set_tspan_text(root, "tspan1-6-9", "46 features (N = 53)")
    _set_tspan_text(root, "tspan1-6-8", "46 features (N = 41)")
    _set_tspan_text(root, "tspan1-6-9-0", "46 features (N = 53)")

    # Position vector legend groups over Panel 4B and 4C legend boxes
    for rid, gid, orig_dx, ax_leg in (
        ("rect2", "g3", 2.5, axG),
        ("rect2-3", "g4", 2.8, axH),
    ):
        leg = ax_leg.get_legend()
        rmask = root.find(f'.//*[@id="{rid}"]')
        g_leg = root.find(f'.//*[@id="{gid}"]')
        if leg is not None and rmask is not None:
            bb = leg.get_window_extent(rend)
            lx0, ly1 = mp.disp_to_svg(bb.x0, bb.y0)
            lx1, ly0 = mp.disp_to_svg(bb.x1, bb.y1)
            rmask.attrib["x"] = f"{lx0 - 3.0:.5f}"
            rmask.attrib["y"] = f"{ly0 - 1.5:.5f}"
            rmask.attrib["width"] = f"{(lx1 - lx0) + 6.0:.5f}"
            rmask.attrib["height"] = f"{(ly1 - ly0) + 3.0:.5f}"
            if g_leg is not None:
                yc = 0.5 * (ly0 + ly1)
                g_leg.attrib["transform"] = f"translate({orig_dx:.5f},{yc - 263.05893:.5f})"

    # 2. Position right-hand kinematic family brackets on Panel 4B
    bG_w = mp.ax_box_svg(axG, tight=False)
    x_bar = bG_w[2] + 2.0
    x_txt = x_bar + 2.8

    _, hp_y0 = mp.data_to_svg(axG, 0, 45.35)
    _, hp_y1 = mp.data_to_svg(axG, 0, 26.65)
    _, ap_y0 = mp.data_to_svg(axG, 0, 26.35)
    _, ap_y1 = mp.data_to_svg(axG, 0, 13.65)
    _, at_y0 = mp.data_to_svg(axG, 0, 13.35)
    _, at_y1 = mp.data_to_svg(axG, 0, 0.65)
    _, mt_y_mid = mp.data_to_svg(axG, 0, 0.0)

    root.find('.//*[@id="path17-3-1-4-6"]').attrib["d"] = f"M {x_bar:.5f},{hp_y0:.5f} V {hp_y1:.5f}"
    root.find('.//*[@id="path17-3-1-4-0-1"]').attrib["d"] = f"m {x_bar:.5f},{ap_y0:.5f} v {ap_y1 - ap_y0:.5f}"
    root.find('.//*[@id="path17-3-1-0"]').attrib["d"] = f"m {x_bar:.5f},{at_y0:.5f} v {at_y1 - at_y0:.5f}"
    root.find('.//*[@id="path17-3-1-0-7"]').attrib["d"] = f"m {x_bar:.5f},{mt_y_mid - 1.0:.5f} v 2.00000"

    for tid, y_mid, gap in (
        ("text9-9-1", 0.5 * (hp_y0 + hp_y1), 6.4),
        ("text9-8-9", 0.5 * (ap_y0 + ap_y1), 6.4),
        ("text9-95", 0.5 * (at_y0 + at_y1) - 1.0, 6.4),
        ("text9-95-2", mt_y_mid + 0.4, 5.4),
    ):
        tel = root.find(f'.//*[@id="{tid}"]')
        _scale_family_label(tel, 5.2)
        tel.attrib["x"] = f"{x_txt:.5f}"
        for ts in tel:
            ts.attrib["x"] = f"{x_txt:.5f}"
        _set_two_line_label_y(tel, y_mid, line_gap=gap)

    # 3. Align Panel D, E, F headers and r / AUC legend cleanly to bottom-row axes
    bD_w = mp.ax_box_svg(axD, tight=False)
    bE_w = mp.ax_box_svg(axE, tight=False)
    bE_t = mp.ax_box_svg(axE, tight=True)
    bF_w = mp.ax_box_svg(axF, tight=False)
    _move_svg_text_y(root, "text1-7-2-1", bD_w[1] - 6.0)
    _move_svg_text_y(root, "text1-7-2-1-2", bD_w[1] - 6.0)
    _move_svg_text_y(root, "text1-7-2-1-2-3", bF_w[1] - 6.2)

    g16 = root.find('.//*[@id="g16"]')
    if g16 is not None:
        cx = 0.5 * (bE_w[0] + bE_w[2])
        top_y = bE_t[3] + 2.2
        g16.attrib["transform"] = f"translate({cx - 334.7:.5f},{top_y - 308.93869:.5f})"

    out_svg = os.path.join(FIGURES_DIR, "figure4.svg")
    _write_svg_and_png(tree, out_svg, out_png, fallback_png_bytes=png_bytes, dpi=dpi)
    print(f"Saved {out_svg} and {out_png}")
