# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Copyright (c) [2025] [Roman Tenger]

import re, sys, logging, os, argparse, math, time
import numpy as np
import matplotlib.pyplot as plt

# ---------- logging ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "nonplanar_ironing_log.txt")
logging.basicConfig(filename=log_file_path, filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")

def dbg(msg, echo=False):
    logging.info(msg)
    if echo:
        print(msg)
# ---------- visualization helpers ----------
def _ensure_dir(d):
    try:
        os.makedirs(d, exist_ok=True)
    except Exception as e:
        dbg(f"[viz] could not create dir {d}: {e}")

def _viz_grid(grid, mask, xmin, ymin, cell_size, title, outpath):
    try:
        if not outpath: return
        plt.figure()
        arr = np.array(grid, dtype=float)
        plt.title(title)
        plt.imshow(arr, origin="lower", extent=[xmin, xmin+arr.shape[1]*cell_size, ymin, ymin+arr.shape[0]*cell_size], aspect="equal")
        plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
        plt.tight_layout()
        plt.savefig(outpath, dpi=400)
        plt.close()
    except Exception as e:
        dbg(f"[viz] grid failed {title}: {e}")

def _viz_mask(mask, xmin, ymin, cell_size, title, outpath):
    try:
        if not outpath: return
        plt.figure()
        arr = np.array(mask, dtype=float)
        plt.title(title)
        plt.imshow(arr, origin="lower", extent=[xmin, xmin+arr.shape[1]*cell_size, ymin, ymin+arr.shape[0]*cell_size], aspect="equal")
        plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
        plt.tight_layout()
        plt.savefig(outpath, dpi=400)
        plt.close()
    except Exception as e:
        dbg(f"[viz] mask failed {title}: {e}")

def _viz_slope(Gx, Gy, xmin, ymin, cell_size, title, outpath):
    try:
        if not outpath: return
        plt.figure()
        mag = np.sqrt(np.array(Gx, dtype=float)**2 + np.array(Gy, dtype=float)**2)
        plt.title(title)
        plt.imshow(mag, origin="lower", extent=[xmin, xmin+mag.shape[1]*cell_size, ymin, ymin+mag.shape[0]*cell_size], aspect="equal")
        plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
        plt.tight_layout()
        plt.savefig(outpath, dpi=400)
        plt.close()
    except Exception as e:
        dbg(f"[viz] slope failed {title}: {e}")

def _viz_polylines(polys, xmin, ymin, xmax, ymax, title, outpath):
    try:
        if not outpath: return
        plt.figure()
        plt.title(title)
        for pl in polys or []:
            xs = [p[0] for p in pl]; ys = [p[1] for p in pl]
            plt.plot(xs, ys, linewidth=0.6)
        plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
        ax = plt.gca(); ax.set_aspect("equal", adjustable="box")
        plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
        plt.tight_layout()
        plt.savefig(outpath, dpi=400)
        plt.close()
    except Exception as e:
        dbg(f"[viz] polylines failed {title}: {e}")


# ---------- regex ----------
_float = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

RE_MOVE_LINE  = re.compile(r"^\s*(?:N\d+\s*)?(?:G0|G00|G1|G01|G2|G02|G3|G03)\b", re.I)
RE_G01        = re.compile(r"^\s*(?:N\d+\s*)?(?:G0|G00|G1|G01)\b", re.I)
RE_G2         = re.compile(r"^\s*(?:N\d+\s*)?(?:G2|G02)\b", re.I)
RE_X = re.compile(r"\bX(" + _float + r")\b", re.I)
RE_Y = re.compile(r"\bY(" + _float + r")\b", re.I)
RE_Z = re.compile(r"\bZ(" + _float + r")\b", re.I)
RE_I = re.compile(r"\bI(" + _float + r")\b", re.I)
RE_J = re.compile(r"\bJ(" + _float + r")\b", re.I)
RE_R = re.compile(r"\bR(" + _float + r")\b", re.I)
RE_E = re.compile(r"\bE(" + _float + r")\b", re.I)
RE_TOOL = re.compile(r"^\s*T(\d+)\s*$", re.I)
RE_TYPE_LINE = re.compile(r"^\s*;TYPE:\s*(.+)$", re.I)

RE_WIPE_END  = re.compile(r";\s*WIPE_END\b", re.I)   #-->Hotfix besser weg finde
RE_FIL_END   = re.compile(r";\s*Filament[- ]specific end gcode", re.I)


RE_END_COMMENT = re.compile(r";\s*end\s*(?:g[- ]?code|code)", re.I)
RE_END_CMD     = re.compile(
    r"^\s*(?:M104|M109|M140|M190)\s+[^;]*S0\b|^\s*M84\b|^\s*M106\s+[^;]*S0\b", re.I
)

# ---------- helpers ----------
def _viz_z_fields(xmin, ymin, nx_cells, ny_cells, cell_size, band_top, band_min,
                  H_top_s, H_min_s, H_ero, MASK, IRON_MASK, H_side, H_ball, out_dir, prefix):
    try:
        if not out_dir: return
        import numpy as _np
        Ztop = _np.full((ny_cells, nx_cells), _np.nan)
        Zbase = _np.full_like(Ztop, _np.nan)
        Zlb = _np.full_like(Ztop, _np.nan)
        Zfinal = _np.full_like(Ztop, _np.nan)
        Valid = _np.zeros((ny_cells, nx_cells), dtype=float)
        for yi in range(ny_cells):
            py = ymin + (yi + 0.5) * cell_size
            for xi in range(nx_cells):
                px = xmin + (xi + 0.5) * cell_size
                top = _bilinear(H_top_s, MASK, px, py)
                base = _bilinear(H_ero, IRON_MASK, px, py)
                zmin_material = _bilinear(H_min_s, MASK, px, py)
                if top is None or base is None: continue
                lower_by_valley = top - max_depth
                lb = max(lower_by_valley, band_min)
                if zmin_material is not None: lb = max(lb, zmin_material)
                if H_side is not None:
                    sb = _bilinear(H_side, MASK, px, py)
                    if sb is not None: lb = max(lb, sb)
                if H_ball is not None:
                    bb = _bilinear(H_ball, MASK, px, py)
                    if bb is not None: lb = max(lb, bb)
                zt = max(base + z_offset, lb)
                zf = min(zt, top)
                Ztop[yi,xi] = top; Zbase[yi,xi] = base + z_offset; Zlb[yi,xi] = lb; Zfinal[yi,xi] = zf
                if band_min - 1e-6 <= top <= band_top + 1e-6 and zf is not None:
                    Valid[yi,xi] = 1.0
        _viz_grid(Ztop, MASK, xmin, ymin, cell_size, f"{prefix} top", os.path.join(out_dir, f"{prefix}_top.png"))
        _viz_grid(Zbase, MASK, xmin, ymin, cell_size, f"{prefix} base_plus_offset", os.path.join(out_dir, f"{prefix}_base_plus_offset.png"))
        _viz_grid(Zlb, MASK, xmin, ymin, cell_size, f"{prefix} lower_bound", os.path.join(out_dir, f"{prefix}_lower_bound.png"))
        _viz_grid(Zfinal, MASK, xmin, ymin, cell_size, f"{prefix} finalZ", os.path.join(out_dir, f"{prefix}_finalZ.png"))
        _viz_mask(Valid, xmin, ymin, cell_size, f"{prefix} valid cells", os.path.join(out_dir, f"{prefix}_valid.png"))
    except Exception as e:
        dbg(f"[viz] z_fields failed: {e}")
def _viz_composite(img_paths, title, outpath):
    """Create a 2x2 composite board from up to 4 image paths."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        imgs = [Image.open(pth).convert("RGB") for pth in img_paths if pth and os.path.exists(pth)]
        if not imgs: return
       
        w = min(im.size[0] for im in imgs)
        h = min(im.size[1] for im in imgs)
        imgs = [im.resize((w,h)) for im in imgs[:4]]

        while len(imgs) < 4:
            imgs.append(Image.new("RGB", (w,h), (240,240,240)))
        board = Image.new("RGB", (2*w, 2*h), (255,255,255))
        board.paste(imgs[0], (0,0))
        board.paste(imgs[1], (w,0))
        board.paste(imgs[2], (0,h))
        board.paste(imgs[3], (w,h))
        # Title bar
        draw = ImageDraw.Draw(board)
        draw.rectangle([0,0,2*w,30], fill=(255,255,255))
        draw.text((10,8), title, fill=(0,0,0))
        board.save(outpath)
    except Exception as e:
        dbg(f"[viz] composite failed {title}: {e}")


def _viz_diff_mask(A_mask, B_mask, xmin, ymin, cell_size, title, outpath):
    """Visualize A and B and A\B difference (A only)."""
    try:
        import numpy as _np
        H, W = len(A_mask), len(A_mask[0]) if A_mask else 0
        A = _np.array(A_mask, dtype=float)
        B = _np.array(B_mask, dtype=float)
        Diff = ((A > 0.5) & ~(B > 0.5)).astype(float)
        _viz_mask(A, xmin, ymin, cell_size, title+" (A)", outpath.replace(".png", "_A.png"))
        _viz_mask(B, xmin, ymin, cell_size, title+" (B)", outpath.replace(".png", "_B.png"))
        _viz_mask(Diff, xmin, ymin, cell_size, title+" (A minus B)", outpath.replace(".png", "_DIFF.png"))
    except Exception as e:
        dbg(f"[viz] diff mask failed {title}: {e}")


def _viz_legend(outpath):
    """Create a simple vertical colorbar legend image."""
    try:
        import numpy as _np
        import matplotlib.pyplot as _plt
        grad = _np.linspace(0,1,256).reshape(-1,1)
        _plt.figure(figsize=(1.8,3), dpi=400)
        _plt.imshow(grad, aspect='auto', origin='lower')
        _plt.axis('off')
        _plt.tight_layout(pad=0.1)
        _plt.savefig(outpath)
        _plt.close()
    except Exception as e:
        dbg(f"[viz] legend failed: {e}")


def classify_type(tag_text: str) -> str | None:
    t = tag_text.upper()
    # outer/top
    if ("EXTERNAL" in t and "PERIMETER" in t) or ("WALL-OUTER" in t) or ("OUTER WALL" in t) or ("EXTERNAL WALL" in t):
        return "external"
    if ("TOP" in t and "SKIN" in t) or ("TOP SOLID INFILL" in t) or ("TOP INFILL" in t) or ("SKIN TOP" in t):
        return "top"
    # donors 
    if ("PERIMETER" in t and "EXTERNAL" not in t) or ("INNER" in t and "WALL" in t):
        return "internal"
    # solid infill fix
    if "SOLID INFILL" in t and "TOP" not in t and "BOTTOM" not in t:
        return "top"
    return None

def strip_comment(line: str) -> str:
    i = line.find(";")
    return (line if i < 0 else line[:i]).strip()

def parse_val(rx, text, default=None):
    m = rx.search(text);  return float(m.group(1)) if m else default

def lerp(a, b, t): return a + (b - a) * t

def arc_to_lines(x0, y0, x1, y1, cw, i, j, r, max_seg_len=0.6):
    pts = []
    if (i is None or j is None) and r is None:
        return [(x1, y1)]
    if i is not None and j is not None:
        cx, cy = x0 + i, y0 + j
        dx0, dy0 = x0 - cx, y0 - cy
        dx1, dy1 = x1 - cx, y1 - cy
        r0 = math.hypot(dx0, dy0)
        if r0 == 0: return [(x1, y1)]
        a0 = math.atan2(dy0, dx0); a1 = math.atan2(dy1, dx1)
        d = a1 - a0
        if cw:
            if d > 0: d -= 2*math.pi
        else:
            if d < 0: d += 2*math.pi
        sweep = abs(d); arc_len = r0 * sweep
        nseg = max(1, int(math.ceil(arc_len / max_seg_len)))
        for k in range(1, nseg + 1):
            ak = a0 + (d * k / nseg)
            pts.append((cx + r0 * math.cos(ak), cy + r0 * math.sin(ak)))
        return pts
    
    mx, my = (x0 + x1)/2.0, (y0 + y1)/2.0
    dx, dy = x1 - x0, y1 - y0
    q = math.hypot(dx, dy)
    if q == 0: return [(x1, y1)]
    rr = abs(r); rr = max(rr, q/2)
    h = math.sqrt(max(rr*rr - (q*q)/4.0, 0.0))
    ux, uy = -dy/q, dx/q
    if (r >= 0) ^ cw:
        cx, cy = mx + ux*h, my + uy*h
    else:
        cx, cy = mx - ux*h, my - uy*h
    return arc_to_lines(x0, y0, x1, y1, cw, cx - x0, cy - y0, None, max_seg_len)

# ---- grid & contours ----
def _ed_dist_transform(mask, cell_size):
    ny, nx = len(mask), len(mask[0])
    INF = 10**9
    d = [[0 if not mask[y][x] else INF for x in range(nx)] for y in range(ny)]
    w1, w2 = 1.0, 1.41421356237
    for y in range(ny):
        for x in range(nx):
            if d[y][x] == 0: continue
            v = d[y][x]
            if x > 0:                 v = min(v, d[y][x-1] + w1)
            if y > 0:                 v = min(v, d[y-1][x] + w1)
            if x > 0 and y > 0:       v = min(v, d[y-1][x-1] + w2)
            if x+1 < nx and y > 0:    v = min(v, d[y-1][x+1] + w2)
            d[y][x] = v
    for y in range(ny-1, -1, -1):
        for x in range(nx-1, -1, -1):
            v = d[y][x]
            if x+1 < nx:               v = min(v, d[y][x+1] + w1)
            if y+1 < ny:               v = min(v, d[y+1][x] + w1)
            if x+1 < nx and y+1 < ny:  v = min(v, d[y+1][x+1] + w2)
            if x > 0 and y+1 < ny:     v = min(v, d[y+1][x-1] + w2)
            d[y][x] = v
    maxd = 0.0
    for y in range(ny):
        for x in range(nx):
            if mask[y][x] and d[y][x] < 10**9:
                d[y][x] *= cell_size
                if d[y][x] > maxd: maxd = d[y][x]
            else:
                d[y][x] = 0.0
    return d, maxd

def _marching_squares_binary(mask):
    ny, nx = len(mask), len(mask[0])
    segs = []
    def v(x,y): return 1 if mask[y][x] else 0
    for y in range(ny-1):
        for x in range(nx-1):
            a = v(x,   y); b = v(x+1, y)
            c = v(x+1, y+1); d = v(x, y+1)
            case = (a<<3) | (b<<2) | (c<<1) | d
            if case in (0, 15): continue
            top    = (x+0.5, y)
            right  = (x+1.0, y+0.5)
            bottom = (x+0.5, y+1.0)
            left   = (x,     y+0.5)
            table = {
                1:[(left,bottom)],2:[(bottom,right)],3:[(left,right)],
                4:[(top,right)],5:[(left,top),(bottom,right)],6:[(top,bottom)],
                7:[(left,bottom)],8:[(left,top)],9:[(top,right)],
                10:[(left,right),(top,bottom)],11:[(bottom,right)],
                12:[(left,right)],13:[(bottom,right)],14:[(left,top)]
            }
            for (p0,p1) in table.get(case, []):
                segs.append((p0[0],p0[1],p1[0],p1[1]))
    return segs

def _segments_to_polylines(segs, xmin, ymin, cell_size, min_len_mm=0.0):
    def snap(p): return (round(p[0], 4), round(p[1], 4))
    from collections import defaultdict
    pts = []
    for x0,y0,x1,y1 in segs:
        p0 = snap((xmin + x0*cell_size, ymin + y0*cell_size))
        p1 = snap((xmin + x1*cell_size, ymin + y1*cell_size))
        pts.append((p0,p1))
    adj = defaultdict(list)
    for p0,p1 in pts:
        adj[p0].append(p1); adj[p1].append(p0)
    used = set(); polylines = []
    for p0,p1 in pts:
        if (p0,p1) in used or (p1,p0) in used: continue
        line = [p0,p1]
        used.add((p0,p1)); used.add((p1,p0))
        cur = p1
        while True:
            nxts = [q for q in adj[cur] if (cur,q) not in used]
            if not nxts: break
            q = nxts[0]; used.add((cur,q)); used.add((q,cur))
            line.append(q); cur = q
        cur = p0
        while True:
            nxts = [q for q in adj[cur] if (cur,q) not in used]
            if not nxts: break
            q = nxts[0]; used.add((cur,q)); used.add((q,cur))
            line.insert(0,q); cur = q
        length = 0.0
        for i in range(1, len(line)):
            xA,yA = line[i-1]; xB,yB = line[i]
            length += math.hypot(xB-xA, yB-yA)
        if length >= min_len_mm:
            polylines.append(line)
    return polylines

def generate_adaptive_polylines(mask, xmin, ymin, cell_size, stepover_mm, min_loop_mm):
    dist_mm, maxd = _ed_dist_transform(mask, cell_size)
    if maxd < stepover_mm: return []
    levels = [i*stepover_mm for i in range(1, int(maxd/stepover_mm)+1)]
    all_polys = []
    for lvl in levels:
        maskL = [[mask[y][x] and (dist_mm[y][x] >= lvl)
                 for x in range(len(mask[0]))] for y in range(len(mask))]
        segs = _marching_squares_binary(maskL)
        all_polys.extend(_segments_to_polylines(segs, xmin, ymin, cell_size, min_len_mm=min_loop_mm))
    return all_polys

def generate_concentric_polylines(mask, xmin, ymin, cell_size, stepover_mm, min_loop_mm):
    dist_mm, maxd = _ed_dist_transform(mask, cell_size)
    if maxd <= 0.0: return []
    eps = max(0.25*cell_size, 0.01)
    levels = [eps] + [i*stepover_mm for i in range(1, int(maxd/stepover_mm)+1)]
    all_polys = []
    for lvl in levels:
        maskL = [[mask[y][x] and (dist_mm[y][x] >= lvl)
                 for x in range(len(mask[0]))] for y in range(len(mask))]
        segs = _marching_squares_binary(maskL)
        all_polys.extend(_segments_to_polylines(segs, xmin, ymin, cell_size, min_len_mm=min_loop_mm))
    return all_polys

def generate_scanline_polylines(mask, xmin, ymin, xmax, ymax, cell_size, pitch, angle_deg, sample_step):
    ang = math.radians(angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    cx0 = (xmin + xmax) * 0.5; cy0 = (ymin + ymax) * 0.5
    def fwd(px, py):
        dx = px - cx0; dy = py - cy0
        return dx * ca + dy * sa, -dx * sa + dy * ca
    def inv(rx, ry):
        dx =  rx * ca - ry * sa
        dy =  rx * sa + ry * ca
        return cx0 + dx, cy0 + dy
    corners = [fwd(*p) for p in [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]]
    rxmin = min(p[0] for p in corners); rxmax = max(p[0] for p in corners)
    rymin = min(p[1] for p in corners); rymax = max(p[1] for p in corners)
    nx = len(mask[0]); ny = len(mask)
    def mask_at(px, py):
        cx = int((px - xmin) / cell_size); cy = int((py - ymin) / cell_size)
        return (0 <= cx < nx and 0 <= cy < ny and mask[cy][cx])
    polylines = []
    y_scan = rymin; serp = False
    step_x = max(sample_step, 0.05); step_y = max(pitch, step_x)
    while y_scan <= rymax + 1e-9:
        x_cur = rxmin; span = []; spans = []
        while x_cur <= rxmax + 1e-9:
            wx, wy = inv(x_cur, y_scan)
            if mask_at(wx, wy): span.append((wx, wy))
            else:
                if len(span) > 1: spans.append(span)
                span = []
            x_cur += step_x
        if len(span) > 1: spans.append(span)
        for s in spans:
            polylines.append(s if not serp else list(reversed(s))); serp = not serp
        y_scan += step_y
    return polylines


def load_snippet(val: str) -> list[str]:
    """
    Accepts:
      - "" (returns empty list)
      - inline string (supports '\n' sequences for newlines)
      - '@path/to/file.gcode' to load lines from a file
    Ensures every line ends with '\n'.
    """
    if not val:
        return []
    val = str(val)
    if val.startswith("@"):
        path = val[1:]
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            return [l if l.endswith("\n") else (l + "\n") for l in lines]
        except Exception as e:
            dbg(f"[warn] Could not read snippet file '{path}': {e}")
            return []
   
    val = val.replace("\\n", "\n")
    if not val.endswith("\n"):
        val += "\n"
    return val.splitlines(keepends=True)

# ---------- main ----------
def process_gcode(
    input_file,
    cell_size,
    nozzle_diam,
    ironing_pitch,
    z_offset,
    feedrate,
    flow_percent,
    filament_diam,
    ironing_width,
    ironing_height,
    smooth_box_radius,
    erode_radius,
    angle_deg,
    sample_step,
    max_angle_deg,
    max_depth,
    reach_depth,
    shaft_radius,
    shaft_check,
    # Debug
    debug=False,
    debug_reach=False,
    fill_radius=0.8,
    strategy="scanline",
    stepover=0.60,
    min_loop=5.0,
    split_segments=False,
    iron_tool=None,
    band_order="topdown",
    ball_radius=0.0,
    collision_scope="global",   
    adaptive_split=False,
    band_step=None,
    min_new_frac=0.02,
    min_new_abs=25,
    travel_clearance=0.6,
    regen_per_band_paths=True,
    iron_offset_x=0.0,
    iron_offset_y=0.0,
    iron_offset_z=0.0,
    pre_iron_snippet="",
    post_iron_snippet="",
    viz_out_dir=None):
    log = lambda s: dbg(s, echo=debug)
    t_start = time.time()
    if viz_out_dir:
        _ensure_dir(viz_out_dir)
        log(f"[viz] saving visuals to: {viz_out_dir}")
        _viz_legend(os.path.join(viz_out_dir, "legend.png"))
    log("== Nonplanar Ironing: start ==")


    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    log(f"Read {len(lines)} lines")
    log(f"Tool offsets: dX={iron_offset_x:.3f} dY={iron_offset_y:.3f} dZ={iron_offset_z:.3f}")


    pre_lines  = load_snippet(pre_iron_snippet)
    post_lines = load_snippet(post_iron_snippet)
    if pre_lines:
        log(f"Loaded pre-iron snippet ({len(pre_lines)} lines)")
    if post_lines:
        log(f"Loaded post-iron snippet ({len(post_lines)} lines)")


    if iron_tool is not None:
        s = str(iron_tool).strip().upper()
        if s == "" or s == "NONE":
            iron_tool = None
        else:
            s = s if s.startswith("T") else ("T" + s)
            if not re.fullmatch(r"T\d+", s):
                log(f"[warn] -ironTool '{iron_tool}' not recognized → ignoring")
                iron_tool = None
            else:
                iron_tool = s

    if band_order not in ("topdown", "bottomup"):
        band_order = "topdown"
    if collision_scope not in ("global", "printed"):
        collision_scope = "global"
    if adaptive_split and collision_scope == "global":
        log("[hint] adaptiveSplit works best with collisionScope=printed; switching to printed for this run.")
        collision_scope = "printed"

    # sampling defaults
    if cell_size < 0.02:
        log(f"[warn] cellSize {cell_size} too small; clamping to 0.02")
        cell_size = 0.02
    scan_dx = sample_step if sample_step and sample_step > 0 else 0.05
    scan_dy = max(ironing_pitch if ironing_pitch and ironing_pitch > 0 else 0.10, scan_dx)
    if band_step is None or band_step <= 0:
        band_step = max(0.5, min(reach_depth/6.0 if reach_depth>0 else 1.0, 2.0))

    # detect extrusion mode
    absolute_e = True
    for raw in lines:
        w = raw.lstrip().upper()
        if w.startswith("M82"): absolute_e = True; break
        if w.startswith("M83"): absolute_e = False; break
    log(f"E mode: {'absolute (M82)' if absolute_e else 'relative (M83)'}")

    try:
        if viz_out_dir and segments_top:
            _viz_polylines([[ (x,y) for it in seg for (_,x,y,_,_,_,*rest) in [it] ] for seg in segments_top], xmin, ymin, xmax, ymax,
                           "00a Input segments (surface candidates)", os.path.join(viz_out_dir, "00a_input_segments_top.png"))
        if viz_out_dir and segments_donor:
            _viz_polylines([[ (x,y) for it in seg for (_,x,y,_,_,_,*rest) in [it] ] for seg in segments_donor], xmin, ymin, xmax, ymax,
                           "00b Input segments (donors)", os.path.join(viz_out_dir, "00b_input_segments_donor.png"))
    except Exception as e:
        dbg(f"[viz] input segments viz failed: {e}")



    current_kind = None
    x = y = z = None
    last_e_for_abs = None
    segments_top = []      # external + top solid infill (surface candidates)
    segments_donor = []    # internal perimeters (donor for fill/slope)
    xmin = ymin = 1e9; xmax = ymax = -1e9

    def update_bounds(px, py):
        nonlocal xmin, ymin, xmax, ymax
        xmin = min(xmin, px); ymin = min(ymin, py)
        xmax = max(xmax, px); ymax = max(ymax, py)

    for raw in lines:
        ls = raw.rstrip("\n")
        mtype = RE_TYPE_LINE.match(ls)
        if mtype:
            current_kind = classify_type(mtype.group(1))
            continue
        cl = strip_comment(ls)
        if not cl: continue
        if not RE_MOVE_LINE.match(cl): continue

        u = cl.upper()
        nx = parse_val(RE_X, u, x)
        ny = parse_val(RE_Y, u, y)
        nz = parse_val(RE_Z, u, z)
        ne = parse_val(RE_E, u, None)

        extruding = False
        if ne is not None:
            if absolute_e:
                if last_e_for_abs is None: last_e_for_abs = ne
                if ne > last_e_for_abs + 1e-9: extruding = True
                last_e_for_abs = ne
            else:
                extruding = (ne > 0.0 + 1e-9)

        have_prev_xy = (x is not None and y is not None)
        have_new_xy  = (nx is not None and ny is not None)

        def add_seg(container):
            if not have_prev_xy or not have_new_xy: return
            is_arc = (RE_I.search(u) or RE_J.search(u) or RE_R.search(u)) is not None and not RE_G01.match(u)
            if not is_arc:
                container.append((x, y, nx, ny, nz if nz is not None else z))
                update_bounds(x, y); update_bounds(nx, ny)
            else:
                cw = bool(RE_G2.match(u))
                i = parse_val(RE_I, u, None); j = parse_val(RE_J, u, None); r = parse_val(RE_R, u, None)
                px, py = x, y
                for qx, qy in arc_to_lines(x, y, nx, ny, cw, i, j, r, max_seg_len=max(scan_dx, 0.4)):
                    container.append((px, py, qx, qy, nz if nz is not None else z))
                    update_bounds(px, py); update_bounds(qx, qy); px, py = qx, qy

        if extruding:
            if current_kind in ("external", "top"):
                add_seg(segments_top)
            elif current_kind == "internal":
                add_seg(segments_donor)

        x, y, z = nx, ny, nz

    if not segments_top and not segments_donor:
        log("No eligible segments found. Exiting.")
        return

    # -------- global grid envelope  --------
    padding = max(cell_size * 2.0, scan_dy * 2.0)
    xmin -= padding; ymin -= padding; xmax += padding; ymax += padding
    width = xmax - xmin; height = ymax - ymin
    nx_cells = max(1, int(math.ceil(width / cell_size)))
    ny_cells = max(1, int(math.ceil(height / cell_size)))
    log(f"Grid: {nx_cells} x {ny_cells} (cellSize={cell_size}) over {width:.2f}x{height:.2f} mm")

    max_cells = 8_000_000
    if nx_cells * ny_cells > max_cells:
        scale = math.sqrt((nx_cells*ny_cells)/max_cells)
        cell_size *= scale
        nx_cells = max(1, int(math.ceil(width / cell_size)))
        ny_cells = max(1, int(math.ceil(height / cell_size)))
        log(f"[warn] grid too large; raised cellSize to {cell_size:.3f} → {nx_cells}x{ny_cells}")

    def blank_H():
        return [[-1e9 for _ in range(nx_cells)] for __ in range(ny_cells)]
    def blank_M():
        return [[False for _ in range(nx_cells)] for __ in range(ny_cells)]

    
    def to_cell(px, py):
        cx = int((px - xmin) / cell_size); cy = int((py - ymin) / cell_size)
        return cx, cy

    def rasterize_lines_into(fields, segs, footprint_cells, z_limit=None):
        H_top, H_min, MASK = fields
        for (x0, y0, x1, y1, zline) in segs:
            if z_limit is not None and zline is not None and zline > z_limit + 1e-9:
                continue
            dx = x1 - x0; dy = y1 - y0
            dist = math.hypot(dx, dy)
            steps = max(1, int(dist / (0.5 * cell_size)))
            for i in range(steps + 1):
                t = i / steps
                px = lerp(x0, x1, t); py = lerp(y0, y1, t)
                cx, cy = to_cell(px, py)
                if 0 <= cx < nx_cells and 0 <= cy < ny_cells:
                    zv = zline if zline is not None else 0.0
                    if zv > H_top[cy][cx]: H_top[cy][cx] = zv
                    if zv < H_min[cy][cx]: H_min[cy][cx] = zv
                    MASK[cy][cx] = True
                for oy in range(-footprint_cells, footprint_cells + 1):
                    for ox in range(-footprint_cells, footprint_cells + 1):
                        cxx, cyy = cx + ox, cy + oy
                        if 0 <= cxx < nx_cells and 0 <= cyy < ny_cells:
                            MASK[cyy][cxx] = True
                            H_top[cyy][cxx] = max(H_top[cyy][cxx], H_top[cy][cx])
                            H_min[cyy][cxx] = min(H_min[cyy][cxx], H_min[cy][cx])

    def box_blur(H, M, r_cells, label=""):
        if r_cells <= 0: return H
        ny = len(H); nx = len(H[0])
        out = [[H[j][i] for i in range(nx)] for j in range(ny)]
        t = time.time()
        for yy in range(ny):
            y0 = max(0, yy - r_cells); y1 = min(ny - 1, yy + r_cells)
            for xx in range(nx):
                if not M[yy][xx]: continue
                x0 = max(0, xx - r_cells); x1 = min(nx - 1, xx + r_cells)
                s = 0.0; c = 0
                for y2 in range(y0, y1 + 1):
                    for x2 in range(x0, x1 + 1):
                        if M[y2][x2]:
                            v = H[y2][x2]
                            if -1e8 < v < 1e8:
                                s += v; c += 1
                if c > 0: out[yy][xx] = s / c
        dbg(f"{label} blur radius={r_cells} took {time.time()-t:.2f}s")
        return out

    def gray_dilate_max(H, M, r_cells):
        if r_cells <= 0: return H
        ny = len(H); nx = len(H[0])
        out = [[H[j][i] for i in range(nx)] for j in range(ny)]
        kernel = [(dx, dy) for dy in range(-r_cells, r_cells + 1)
                            for dx in range(-r_cells, r_cells + 1)
                            if dx*dx + dy*dy <= r_cells*r_cells]
        for y in range(ny):
            for x in range(nx):
                if not M[y][x]: continue
                mx = -1e9
                for dx, dy in kernel:
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < nx and 0 <= yy < ny and M[yy][xx]:
                        hv = H[yy][xx]
                        if hv > -1e8:
                            mx = max(mx, hv)
                if mx > -1e9 and mx > out[y][x]:
                    out[y][x] = mx
        return out

    def gradient(H, M):
        ny, nx = len(H), len(H[0])
        Gx = [[0.0 for _ in range(nx)] for __ in range(ny)]
        Gy = [[0.0 for _ in range(nx)] for __ in range(ny)]
        inv2h = 1.0 / (2.0 * cell_size)
        for y in range(ny):
            for x in range(nx):
                if not M[y][x] or H[y][x] <= -1e8: continue
                xm = max(0, x - 1); xp = min(nx - 1, x + 1)
                ym = max(0, y - 1); yp = min(ny - 1, y + 1)
                hx = hy = 0.0
                if M[y][xp] and M[y][xm] and -1e8 < H[y][xp] < 1e8 and -1e8 < H[y][xm] < 1e8:
                    hx = (H[y][xp] - H[y][xm]) * inv2h
                if M[yp][x] and M[ym][x] and -1e8 < H[yp][x] < 1e8 and -1e8 < H[ym][x] < 1e8:
                    hy = (H[yp][x] - H[ym][x]) * inv2h
                Gx[y][x] = hx; Gy[y][x] = hy
        return Gx, Gy

    # Build final/global once
    FOOTPRINT = max(1, int(round((ironing_width * 0.5) / cell_size)))
    H_top_g, H_min_g, MASK_g = blank_H(), [[ 1e9 for _ in range(nx_cells)] for __ in range(ny_cells)], blank_M()
    H_don_g, MASK_don_g = blank_H(), blank_M()

    rasterize_lines_into((H_top_g, H_min_g, MASK_g), segments_top, FOOTPRINT, z_limit=None)
    rasterize_lines_into((H_don_g, H_min_g, MASK_don_g), segments_donor, FOOTPRINT, z_limit=None)
    H_top_s_g   = box_blur(H_top_g, MASK_g, smooth_box_radius, "Top(g)")
    H_min_s_g   = box_blur(H_min_g, MASK_g, smooth_box_radius, "Min(g)")
    H_don_s_g   = box_blur(H_don_g, MASK_don_g, smooth_box_radius, "Donor(g)")

    # fill hidden tops via donor union
    fill_cells = max(0, int(round(fill_radius / cell_size)))
    UNION_M = [[MASK_g[y][x] or MASK_don_g[y][x] for x in range(nx_cells)] for y in range(ny_cells)]
    H_union = [[max(H_top_s_g[y][x], H_don_s_g[y][x]) for x in range(nx_cells)] for y in range(ny_cells)]
    H_filled = gray_dilate_max(H_union, UNION_M, fill_cells)
    for y in range(ny_cells):
        for x in range(nx_cells):
            if MASK_g[y][x] and H_filled[y][x] > H_top_s_g[y][x] > -1e8:
                H_top_s_g[y][x] = H_filled[y][x]

    # erosion
    er_cells = max(0, int(round(erode_radius / cell_size)))
    
    try:

        if viz_out_dir:
            _viz_grid(H_don_s_g, MASK_don_g, xmin, ymin, cell_size, "Donor smoothed (global)", os.path.join(viz_out_dir, "02b_global_H_donor_smoothed.png"))
            _viz_mask(MASK_don_g, xmin, ymin, cell_size, "Donor MASK", os.path.join(viz_out_dir, "02c_global_MASK_donor.png"))
            _viz_mask(UNION_M, xmin, ymin, cell_size, "UNION_M (top ∪ donor)", os.path.join(viz_out_dir, "02d_global_UNION.png"))
            _viz_grid(H_union, UNION_M, xmin, ymin, cell_size, "H_union (max of top & donor)", os.path.join(viz_out_dir, "02e_global_H_union.png"))
            _viz_grid(H_filled, UNION_M, xmin, ymin, cell_size, "H_filled (gray dilated)", os.path.join(viz_out_dir, "02f_global_H_filled.png"))

        H_ero_g = [[H_top_s_g[j][i] for i in range(nx_cells)] for j in range(ny_cells)]
        if viz_out_dir:
            _viz_grid(H_ero_g, MASK_g, xmin, ymin, cell_size, "H_ero (global, post-erosion)", os.path.join(viz_out_dir, "02g_global_H_ero.png"))
        if er_cells > 0:
            kernel_er = [(dx, dy) for dy in range(-er_cells, er_cells + 1)
                                  for dx in range(-er_cells, er_cells + 1)
                                  if dx*dx + dy*dy <= er_cells*er_cells]
            for y in range(ny_cells):
                for x in range(nx_cells):
                    if not MASK_g[y][x]: continue
                    mn = 1e9; got = False
                    for dx, dy in kernel_er:
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < nx_cells and 0 <= yy < ny_cells and MASK_g[yy][xx]:
                            h = H_top_s_g[yy][xx]
                            if h > -1e8:
                                mn = min(mn, h); got = True
                    if got: H_ero_g[y][x] = mn

    except Exception as e:
        dbg(f"[viz] block failed (donor/union): {e}")

    # global slope & mask
    Gx_g, Gy_g = gradient(H_top_s_g, MASK_g)
    max_ang_rad = math.radians(max_angle_deg)
    IRON_MASK_G = [[MASK_g[j][i] and (math.atan(math.hypot(Gx_g[j][i], Gy_g[j][i])) <= max_ang_rad)
                   for i in range(nx_cells)] for j in range(ny_cells)]

    # global top
    Z_GLOBAL_TOP = max(H_top_s_g[y][x] for y in range(ny_cells) for x in range(nx_cells) if MASK_g[y][x])
    log(f"Global top Z: {Z_GLOBAL_TOP:.3f} mm")

    # global dilations
    def build_dilate_from(H_src, MASK_src):
        H_side = H_ball = None
        if shaft_check and shaft_radius > 0:
            shaft_cells = int(round(shaft_radius / cell_size))
            H_side = [[-1e9 for _ in range(nx_cells)] for __ in range(ny_cells)]
            for y in range(ny_cells):
                for x in range(nx_cells):
                    if not MASK_src[y][x]: continue
                    mx = -1e9
                    for dy in range(-shaft_cells, shaft_cells+1):
                        for dx in range(-shaft_cells, shaft_cells+1):
                            if dx*dx + dy*dy > shaft_cells*shaft_cells: continue
                            xx, yy = x+dx, y+dy
                            if 0 <= xx < nx_cells and 0 <= yy < ny_cells and MASK_src[yy][xx]:
                                mx = max(mx, H_src[yy][xx])
                    if mx > -1e9: H_side[y][x] = mx
        if ball_radius and ball_radius > 0:
            rad_cells = int(max(1, round(ball_radius / cell_size)))
            R = float(ball_radius)
            H_ball = [[-1e9 for _ in range(nx_cells)] for __ in range(ny_cells)]
            for y in range(ny_cells):
                for x in range(nx_cells):
                    if not MASK_src[y][x]: continue
                    best = -1e9
                    for dy in range(-rad_cells, rad_cells+1):
                        for dx in range(-rad_cells, rad_cells+1):
                            dd_mm = math.hypot(dx*cell_size, dy*cell_size)
                            if dd_mm > R + 1e-9: continue
                            xx, yy = x+dx, y+dy
                            if 0 <= xx < nx_cells and 0 <= yy < ny_cells and MASK_src[yy][xx]:
                                cand = H_src[yy][xx] + math.sqrt(max(R*R - dd_mm*dd_mm, 0.0))
                                if cand > best: best = cand
                    if best > -1e9: H_ball[y][x] = best
        return H_side, H_ball

    H_side_g, H_ball_g = build_dilate_from(H_top_s_g, MASK_g)

    try:
 
        if viz_out_dir:
            _viz_z_fields(xmin, ymin, nx_cells, ny_cells, cell_size, Z_GLOBAL_TOP, -1e-9,
                          H_top_s_g, H_min_s_g, H_ero_g, MASK_g, IRON_MASK_G, H_side_g, H_ball_g,
                          viz_out_dir, "08_global_zbounds")

    except Exception as e:
        dbg(f"[viz] block failed (zbounds): {e}")

    try:
 
        if viz_out_dir:
            _viz_grid(H_top_g, MASK_g, xmin, ymin, cell_size, "H_top_raw (global)", os.path.join(viz_out_dir, "01_global_H_top_raw.png"))
            _viz_grid(H_top_s_g, MASK_g, xmin, ymin, cell_size, "H_top_smoothed (global)", os.path.join(viz_out_dir, "02_global_H_top_smoothed.png"))
            _viz_grid(H_min_s_g, MASK_g, xmin, ymin, cell_size, "H_min_smoothed (global)", os.path.join(viz_out_dir, "03_global_H_min_smoothed.png"))
            _viz_mask(MASK_g, xmin, ymin, cell_size, "MASK_g (top coverage)", os.path.join(viz_out_dir, "04_global_MASK_g.png"))
            _viz_slope(Gx_g, Gy_g, xmin, ymin, cell_size, "Slope magnitude (global)", os.path.join(viz_out_dir, "05_global_slope.png"))
            _viz_mask(IRON_MASK_G, xmin, ymin, cell_size, "IRON_MASK_G (angle OK)", os.path.join(viz_out_dir, "06_global_ironmask.png"))


    except Exception as e:
        dbg(f"[viz] block failed (global): {e}")

    # Global polylines 
    if strategy == "scanline":
        polylines_global = generate_scanline_polylines(IRON_MASK_G, xmin, ymin, xmax, ymax, cell_size, ironing_pitch, angle_deg, sample_step)
    elif strategy == "adaptive":
        polylines_global = generate_adaptive_polylines(IRON_MASK_G, xmin, ymin, cell_size, stepover_mm=stepover, min_loop_mm=min_loop)
    else:
        polylines_global = generate_concentric_polylines(IRON_MASK_G, xmin, ymin, cell_size, stepover_mm=stepover, min_loop_mm=min_loop)
    if not polylines_global:
        log("No polylines after global masking. Exiting.");  return
    if viz_out_dir:
        _viz_polylines(polylines_global, xmin, ymin, xmax, ymax, f"Global polylines ({strategy})", os.path.join(viz_out_dir, "07_global_polylines.png"))


    if flow_percent > 0 and ironing_width > 0 and ironing_height > 0 and filament_diam > 0:
        filament_area = math.pi * (filament_diam * 0.5) ** 2
        e_per_mm = (ironing_width * ironing_height / filament_area) * (flow_percent / 100.0)
        log(f"Ironing e/mm = {e_per_mm:.6f}")
    else:
        e_per_mm = 0.0; log("Wipe-only ironing (flowPercent=0 or missing dims)")


    def _bilinear(H, M, px, py):
        fx = (px - xmin)/cell_size; fy = (py - ymin)/cell_size
        x0 = int(math.floor(fx)); y0 = int(math.floor(fy))
        tx = fx - x0; ty = fy - y0
        def h(ix, iy):
            if 0 <= ix < nx_cells and 0 <= iy < ny_cells and M[iy][ix] and -1e8 < H[iy][ix] < 1e8:
                return H[iy][ix]
            return None
        h00 = h(x0,y0); h10 = h(x0+1,y0); h01 = h(x0,y0+1); h11 = h(x0+1,y0+1)
        vals = [v for v in (h00,h10,h01,h11) if v is not None]
        if not vals: return None
        if None in (h00,h10,h01,h11):
            return sum(vals)/len(vals)
        a = lerp(h00,h10,tx); b = lerp(h01,h11,tx); return lerp(a,b,ty)

    def z_path(px, py, band_top, band_min, H_top_s, H_min_s, H_ero, MASK, IRON_MASK, H_side=None, H_ball=None):
        top = _bilinear(H_top_s, MASK, px, py)
        if top is None or top < band_min - 1e-6 or top > band_top + 1e-6: return None
        base = _bilinear(H_ero, IRON_MASK, px, py)
        if base is None: return None
        zmin_material = _bilinear(H_min_s, MASK, px, py)
        lower_by_valley = top - max_depth
        lb = max(lower_by_valley, band_min)
        if zmin_material is not None: lb = max(lb, zmin_material)
        if H_side is not None:
            sb = _bilinear(H_side, MASK, px, py)
            if sb is not None: lb = max(lb, sb)
        if H_ball is not None:
            bb = _bilinear(H_ball, MASK, px, py)
            if bb is not None: lb = max(lb, bb)
        zt = max(base + z_offset, lb)
        return min(zt, top)

    
    def e_mode_scan(orig):
        cur_abs = absolute_e; cur_e = 0.0; snaps = []
        for raw in orig:
            u = raw.lstrip().upper()
            if u.startswith("M82"): cur_abs = True
            elif u.startswith("M83"): cur_abs = False
            cl = strip_comment(raw)
            if cl and RE_MOVE_LINE.match(cl):
                ne = parse_val(RE_E, cl.upper(), None)
                if ne is not None:
                    if cur_abs: cur_e = ne
                    else:       cur_e += ne
            snaps.append((cur_e, cur_abs))
        return snaps
    e_snaps = e_mode_scan(lines)

    def build_layers(orig):
        idxs = []
        for i, raw in enumerate(orig):
            u = raw.upper()
            if ";LAYER_CHANGE" in u or ";LAYER:" in u or "; BEFORE_LAYER_CHANGE" in u:
                idxs.append(i)
        entries = []
        for idx in idxs:
            z_val = None
            for k in range(idx, min(idx+40, len(orig))):
                cl = strip_comment(orig[k])
                if not cl: continue
                if RE_G01.match(cl) and RE_Z.search(cl) and not (RE_X.search(cl) or RE_Y.search(cl)):
                    z_val = parse_val(RE_Z, cl.upper(), None); break
            if z_val is not None:
                entries.append({'start_idx': idx, 'z': z_val})
        layers = []
        for i, ent in enumerate(entries):
            s = ent['start_idx']; z = ent['z']
            e = (entries[i+1]['start_idx'] - 1) if (i+1 < len(entries)) else (len(orig) - 1)
            layers.append({'start_idx': s, 'end_idx': e, 'z': z})
        return layers
    layers = build_layers(lines)

    def prev_tool_at(i):
        j = min(max(i, 0), len(lines)-1)
        while j >= 0:
            cl = strip_comment(lines[j])
            if cl:
                m = RE_TOOL.match(cl)
                if m: return "T" + m.group(1)
            j -= 1
        return "T0"

    def single_pass_insertion_idx():
        
        last_wipe = -1
        for i, raw in enumerate(lines):
            if RE_WIPE_END.search(raw):
                last_wipe = i
        if last_wipe >= 0:
            log(f"SINGLE-PASS insertion after last ;WIPE_END at line {last_wipe}")
            return last_wipe  
        
        for i, raw in enumerate(lines):
            if RE_FIL_END.search(raw):
                log(f"SINGLE-PASS insertion before filament-specific end gcode at line {i}")
                return max(0, i-1)

      
        if layers:
            ins = layers[-1]['end_idx']
            log(f"SINGLE-PASS insertion after last layer end at line {ins}")
            return ins

        #funzt nid -> fix
        for i, raw in enumerate(lines):
            text = raw.strip()
            if RE_END_COMMENT.search(text) or RE_END_CMD.search(text):
                log(f"SINGLE-PASS insertion before end code at line {i}")
                return max(0, i-1)
        log("SINGLE-PASS insertion fallback at EOF-1")
        return max(0, len(lines)-2)



    if (not split_segments) or reach_depth <= 0:
        ins_idx = single_pass_insertion_idx()
        E_seed, e_abs_here = e_snaps[ins_idx if ins_idx < len(e_snaps) else len(e_snaps)-1]
        active_tool = prev_tool_at(ins_idx)
  
        fp = None
        for poly in polylines_global:
            for (px, py) in poly:
                pz = z_path(px, py, Z_GLOBAL_TOP, -1e-9, H_top_s_g, H_min_s_g, H_ero_g, MASK_g, IRON_MASK_G, H_side_g, H_ball_g)
                if pz is not None: fp = (px,py,pz); break
            if fp: break
        if not fp:
            log("No valid ironing points for single-pass. Leaving file untouched."); return
        sx, sy, sz = fp; safe_z = Z_GLOBAL_TOP + travel_clearance

        blk = []
        blk += ["\n; -----------------------------------------\n",
                "; BEGIN NONPLANAR IRONING (single-pass)\n",
                ";TYPE:CustomIroning\n",
                f"; strategy={strategy} angle={angle_deg} pitch={ironing_pitch} maxAngle={max_angle_deg} "
                f"valleyClamp={max_depth} zOffset={z_offset} shaftCheck={'on' if shaft_check else 'off'}\n",
                f"; toolOffset: dX={iron_offset_x:.3f} dY={iron_offset_y:.3f} dZ={iron_offset_z:.3f}\n",
                "; -----------------------------------------\n"]
        use_tool = (iron_tool is not None and iron_tool != active_tool)
        E_acc = 0.0 if (use_tool and e_abs_here) else E_seed
        curF = None

    
        def emit(wx=None,wy=None,wz=None,dist=0.0,force_f=False):
            nonlocal curF,E_acc
            ox = (wx + iron_offset_x) if wx is not None else None
            oy = (wy + iron_offset_y) if wy is not None else None
            oz = (wz + iron_offset_z) if wz is not None else None
            parts=["G1"]
            if ox is not None: parts.append(f"X{ox:.3f}")
            if oy is not None: parts.append(f"Y{oy:.3f}")
            if oz is not None: parts.append(f"Z{oz:.3f}")
            if dist>0.0 and e_per_mm>0.0:
                de = dist*e_per_mm
                if e_abs_here: E_acc += de; parts.append(f"E{E_acc:.5f}")
                else: parts.append(f"E{de:.5f}")
            if force_f or curF!=feedrate: parts.append(f"F{feedrate:.0f}"); curF=feedrate
            blk.append(" ".join(parts)+"\n")

        if use_tool:
            blk.append(f"{iron_tool}\n")
            if e_abs_here: blk.append("G92 E0\n")

       
        if pre_lines:
            blk.append("; -- PRE-IRON CUSTOM GCODE --\n")
            blk.extend(pre_lines)

        blk.append("; travel\n"); emit(wz=safe_z, force_f=True); emit(wx=sx,wy=sy,wz=safe_z); emit(wx=sx,wy=sy,wz=sz)
        blk.append("; paths\n")
        for poly in polylines_global:
            started=False; lx=ly=None
            for (px,py) in poly:
                pz = z_path(px,py,Z_GLOBAL_TOP,-1e-9,H_top_s_g,H_min_s_g,H_ero_g,MASK_g,IRON_MASK_G,H_side_g,H_ball_g)
                if pz is None: lx,ly=px,py; continue
                if not started:
                    emit(wz=max(pz,safe_z)); emit(wx=px,wy=py,wz=max(pz,safe_z)); emit(wx=px,wy=py,wz=pz)
                    started=True; lx,ly=px,py; continue
                d=math.hypot(px-lx,py-ly); emit(wx=px,wy=py,wz=pz,dist=d); lx,ly=px,py

       
        if post_lines:
            blk.append("; -- POST-IRON CUSTOM GCODE --\n")
            blk.extend(post_lines)

        if use_tool:
            blk.append(f"{active_tool}\n")
            if e_abs_here: blk.append(f"G92 E{E_seed:.5f}\n")
        else:
            if e_abs_here and abs(E_acc-E_seed)>1e-9: blk.append(f"G92 E{E_seed:.5f}\n")
            if e_abs_here and abs(E_acc-E_seed)>1e-9: blk.append(f"G92 E{E_seed:.5f}\n")
        blk.append("; END NONPLANAR IRONING\n")

        out=[];  [out.append(line) or (out.extend(blk) if i==ins_idx else None) for i,line in enumerate(lines)]
        with open(input_file,"w",encoding="utf-8") as wf: wf.writelines(out)
        log(f"Inserted single-pass ironing before end G-code at line {ins_idx}.")
        log(f"== Done in {time.time()-t_start:.2f}s. Log: {log_file_path} ==");  return

    # ---------- per-band ----------
    band_cache = {}
    def get_band_fields(band_top):
        key = round(band_top, 3)
        if key in band_cache: return band_cache[key]

        Ht, Hm, Mk = blank_H(), [[ 1e9 for _ in range(nx_cells)] for __ in range(ny_cells)], blank_M()
        Hd, Md = blank_H(), blank_M()
        rasterize_lines_into((Ht,Hm,Mk), segments_top, FOOTPRINT, z_limit=band_top)
        rasterize_lines_into((Hd,Hm,Md), segments_donor, FOOTPRINT, z_limit=band_top)
        Ht_s = box_blur(Ht, Mk, smooth_box_radius, "Top(band)")
        Hm_s = box_blur(Hm, Mk, smooth_box_radius, "Min(band)")
        Hd_s = box_blur(Hd, Md, smooth_box_radius, "Donor(band)")
        # donor lift
        UNION = [[Mk[y][x] or Md[y][x] for x in range(nx_cells)] for y in range(ny_cells)]
        Huni = [[max(Ht_s[y][x], Hd_s[y][x]) for x in range(nx_cells)] for y in range(ny_cells)]
        Hfill = gray_dilate_max(Huni, UNION, max(0, int(round(fill_radius / cell_size))))
        for y in range(ny_cells):
            for x in range(nx_cells):
                if Mk[y][x] and Hfill[y][x] > Ht_s[y][x] > -1e8:
                    Ht_s[y][x] = Hfill[y][x]
        # erosion
        H_ero = [[Ht_s[j][i] for i in range(nx_cells)] for j in range(ny_cells)]
        if er_cells > 0:
            kernel_er = [(dx, dy) for dy in range(-er_cells, er_cells + 1)
                                  for dx in range(-er_cells, er_cells + 1)
                                  if dx*dx + dy*dy <= er_cells*er_cells]
            for y in range(ny_cells):
                for x in range(nx_cells):
                    if not Mk[y][x]: continue
                    mn = 1e9; got=False
                    for dx,dy in kernel_er:
                        xx,yy=x+dx,y+dy
                        if 0<=xx<nx_cells and 0<=yy<ny_cells and Mk[yy][xx]:
                            h=Ht_s[yy][xx]
                            if h>-1e8: mn=min(mn,h); got=True
                    if got: H_ero[y][x]=mn
        # slope & angle mask
        Gx_b, Gy_b = gradient(Ht_s, Mk)
        IRON_M = [[Mk[j][i] and (math.atan(math.hypot(Gx_b[j][i], Gy_b[j][i])) <= max_ang_rad)
                  for i in range(nx_cells)] for j in range(ny_cells)]
        # collisions 
        H_side_b, H_ball_b = build_dilate_from(Ht_s, Mk)
        
        if viz_out_dir:
            tag = f"{key:.3f}".replace(".","p")
            _viz_grid(Ht_s, Mk, xmin, ymin, cell_size, f"Band {key:.3f} H_top_smoothed", os.path.join(viz_out_dir, f"band_{tag}_H_top.png"))
            _viz_grid(H_ero, Mk, xmin, ymin, cell_size, f"Band {key:.3f} H_ero (eroded)", os.path.join(viz_out_dir, f"band_{tag}_H_ero.png"))
            _viz_mask(IRON_M, xmin, ymin, cell_size, f"Band {key:.3f} IRON_MASK", os.path.join(viz_out_dir, f"band_{tag}_IRON_MASK.png"))

        pack = (Ht_s, Hm_s, H_ero, Mk, IRON_M, H_side_b, H_ball_b)
        band_cache[key] = pack
        return pack

    
    zmin_iron = min(H_top_s_g[y][x] for y in range(ny_cells) for x in range(nx_cells) if MASK_g[y][x] and H_top_s_g[y][x] > -1e8)
    span = max(0.0, Z_GLOBAL_TOP - zmin_iron)
    if reach_depth <= 0:
        log("reachDepth <= 0; nothing to split."); return

    bands=[]
    if not adaptive_split:
        nseg = max(1, int(math.ceil(span / reach_depth)))
        if band_order == "topdown":
            for k in range(nseg):
                bt = Z_GLOBAL_TOP - k*reach_depth; bm = bt - reach_depth; bands.append((bm, bt))
        else:
            for k in range(nseg):
                bm = zmin_iron + k*reach_depth; bt = min(zmin_iron + (k+1)*reach_depth, Z_GLOBAL_TOP); bands.append((bm, bt))
    else:
        total_candidates = sum(1 for y in range(ny_cells) for x in range(nx_cells) if IRON_MASK_G[y][x])
        thr_rel = int(math.ceil(min_new_frac * max(1, total_candidates)))
        thr_abs = max(0, int(min_new_abs))
        thr = max(1, max(thr_rel, thr_abs))
        log(f"Adaptive planning (printed): candidates={total_candidates}, threshold={thr} (rel={thr_rel}, abs={thr_abs})")

        last_commit_top = Z_GLOBAL_TOP
        current_top = Z_GLOBAL_TOP
        while current_top > zmin_iron + 1e-6:
            prev_top = current_top
            next_top = max(zmin_iron, current_top - band_step)
            def reach_mask(btop, pack):
                Ht_s,Hm_s,H_ero,MK,IM,HS,HB = pack
                REACH = [[False for _ in range(nx_cells)] for __ in range(ny_cells)]
                for yi in range(ny_cells):
                    for xi in range(nx_cells):
                        if not IM[yi][xi]: continue
                        px = xmin + (xi + 0.5) * cell_size; py = ymin + (yi + 0.5) * cell_size
                        if z_path(px,py,btop,-1e-9,Ht_s,Hm_s,H_ero,MK,IM,HS,HB) is not None:
                            REACH[yi][xi] = True
                return REACH
            reach_prev = reach_mask(prev_top, get_band_fields(prev_top))
            reach_next = reach_mask(next_top, get_band_fields(next_top))
            new_cells = 0
            for y in range(ny_cells):
                for x in range(nx_cells):
                    if reach_next[y][x] and not reach_prev[y][x]:
                        new_cells += 1
            thickness = last_commit_top - next_top
            reason = None
            if thickness >= reach_depth - 1e-9: reason="thickness"
            elif new_cells >= thr: reason="newCells"
            dbg(f"  sweep: prev_top={prev_top:.3f} next_top={next_top:.3f} "
                f"thickness={thickness:.3f} newlyReachable={new_cells} "
                f"{'-- COMMIT: '+reason if reason else ''}", echo=debug)
            if reason:
                bands.append((next_top, last_commit_top))
                last_commit_top = next_top
            current_top = next_top

        if last_commit_top > zmin_iron + 1e-6:
            cur = last_commit_top
            while cur > zmin_iron + 1e-6:
                bt = cur; bm = max(zmin_iron, bt - reach_depth); bands.append((bm, bt)); cur = bm

        bands = list(dict.fromkeys(bands))
        bands.sort(key=(lambda p: p[1]), reverse=(band_order=="topdown"))

    if bands:
        log(f"Planned {len(bands)} ironing band(s) ({'adaptive' if adaptive_split else 'uniform'}, order={band_order}):")
        for i,(bm,bt) in enumerate(bands,1): log(f"  Band {i}: Z [{bm:.3f}, {bt:.3f}]  thickness={bt-bm:.3f} mm")
    else:
        log("No bands planned."); return

    
    def insertion_after_band_top(bt):
        if layers:
            chosen=None
            for L in layers:
                if L['z'] <= bt + 1e-6: chosen=L
            idx = (chosen['end_idx'] if chosen else -1)
            idx=max(-1,min(idx,len(lines)-1))
            E_seed, mode_abs = e_snaps[idx if idx>=0 else 0]
            return idx, E_seed, mode_abs
        last_z=None; cur_abs=absolute_e; cur_e=0.0
        for i,raw in enumerate(lines):
            u=raw.lstrip().upper()
            if u.startswith("M82"): cur_abs=True
            elif u.startswith("M83"): cur_abs=False
            cl=strip_comment(raw)
            if not cl or not RE_MOVE_LINE.match(cl): continue
            ne=parse_val(RE_E, cl.upper(), None)
            if ne is not None: cur_e = ne if cur_abs else (cur_e+ne)
            nz=parse_val(RE_Z, cl.upper(), None)
            if nz is None: continue
            if last_z is not None and last_z <= bt + 1e-6 and nz > bt + 1e-6:
                return i-1, cur_e, cur_abs
            last_z = nz
        return len(lines)-1, cur_e, cur_abs

    
    inject_map = {}; bands_built=0

    def generate_polylines_from_mask(mask_band):
        if strategy == "scanline":
            return generate_scanline_polylines(mask_band, xmin, ymin, xmax, ymax, cell_size, ironing_pitch, angle_deg, sample_step)
        elif strategy == "adaptive":
            return generate_adaptive_polylines(mask_band, xmin, ymin, cell_size, stepover_mm=stepover, min_loop_mm=min_loop)
        else:
            return generate_concentric_polylines(mask_band, xmin, ymin, cell_size, stepover_mm=stepover, min_loop_mm=min_loop)

    def build_band_block(idx, bm, bt, E_seed, e_abs_here, active_tool):
        if collision_scope == "printed":
            Ht_s,Hm_s,H_ero,MK,IM,HS,HB = get_band_fields(bt)
        else:
            Ht_s,Hm_s,H_ero,MK,IM,HS,HB = H_top_s_g,H_min_s_g,H_ero_g,MASK_g,IRON_MASK_G,H_side_g,H_ball_g

     
        if regen_per_band_paths or collision_scope == "printed":
            REACH = [[False for _ in range(nx_cells)] for __ in range(ny_cells)]
            for yi in range(ny_cells):
                for xi in range(nx_cells):
                    if not IM[yi][xi]: continue
                    px = xmin + (xi + 0.5)*cell_size; py = ymin + (yi + 0.5)*cell_size
                    if z_path(px,py,bt,bm,Ht_s,Hm_s,H_ero,MK,IM,HS,HB) is not None:
                        REACH[yi][xi]=True
            
            if viz_out_dir:
                tag = f"{bt:.3f}".replace(".","p")
                _viz_mask(REACH, xmin, ymin, cell_size, f"Band reach [{bm:.3f},{bt:.3f}]", os.path.join(viz_out_dir, f"band_{tag}_REACH.png"))

            polylines = generate_polylines_from_mask(REACH)
        else:
            polylines = polylines_global

        
        if viz_out_dir and polylines:
            tag = f"{bt:.3f}".replace(".","p")
            _viz_polylines(polylines, xmin, ymin, xmax, ymax, f"Band polylines {idx+1} [{bm:.3f},{bt:.3f}]", os.path.join(viz_out_dir, f"band_{tag}_polylines.png"))

        if not polylines:
            return [";TYPE:CustomIroning\n", f"; Band {idx+1}: no reachable paths.\n"], False

        def z_local(px,py): return z_path(px,py,bt,bm,Ht_s,Hm_s,H_ero,MK,IM,HS,HB)

        blk=[]; curF=None
        use_tool = (iron_tool is not None and iron_tool != active_tool)
        E_acc = 0.0 if (use_tool and e_abs_here) else E_seed

        def emit(wx=None,wy=None,wz=None,dist=0.0,force_f=False):
            nonlocal curF,E_acc
            ox = (wx + iron_offset_x) if wx is not None else None
            oy = (wy + iron_offset_y) if wy is not None else None
            oz = (wz + iron_offset_z) if wz is not None else None
            parts=["G1"]
            if ox is not None: parts.append(f"X{ox:.3f}")
            if oy is not None: parts.append(f"Y{oy:.3f}")
            if oz is not None: parts.append(f"Z{oz:.3f}")
            if dist>0.0 and e_per_mm>0.0:
                de=dist*e_per_mm
                if e_abs_here: E_acc+=de; parts.append(f"E{E_acc:.5f}")
                else: parts.append(f"E{de:.5f}")
            if force_f or curF!=feedrate: parts.append(f"F{feedrate:.0f}"); curF=feedrate
            blk.append(" ".join(parts)+"\n")

        blk += ["\n; -----------------------------------------\n",
                f"; SEGMENT {idx+1} NONPLANAR IRONING\n",
                ";TYPE:CustomIroning\n",
                f"; Z band [{max(0.0,bm):.3f}, {bt:.3f}] scope={collision_scope}\n",
                f"; toolOffset: dX={iron_offset_x:.3f} dY={iron_offset_y:.3f} dZ={iron_offset_z:.3f}\n"]

        if use_tool:
            blk.append(f"{iron_tool}\n")
        if e_abs_here: blk.append("G92 E0\n")

        
        if pre_lines:
            blk.append("; -- PRE-IRON CUSTOM GCODE --\n")
            blk.extend(pre_lines)

        safe_z = bt + travel_clearance
        
        first=None
        for pl in polylines:
            for (px,py) in pl:
                pz=z_local(px,py)
                if pz is not None: first=(px,py,pz); break
            if first: break
        if not first:
            blk.append("; (no points in this band)\n")
            if use_tool:
                blk.append(f"{active_tool}\n")
                if e_abs_here: blk.append(f"G92 E{E_seed:.5f}\n")
            blk.append("; END CustomIroning segment\n")
            return blk, False
        sx,sy,sz=first
        blk.append("; travel to first point\n")
        emit(wz=safe_z, force_f=True); emit(wx=sx,wy=sy,wz=safe_z); emit(wx=sx,wy=sy,wz=sz)

        for poly in polylines:
            started=False; lx=ly=None
            for (px,py) in poly:
                pz=z_local(px,py)
                if pz is None: lx,ly=px,py; continue
                if not started:
                    emit(wz=max(pz,safe_z)); emit(wx=px,wy=py,wz=max(pz,safe_z)); emit(wx=px,wy=py,wz=pz)
                    started=True; lx,ly=px,py; continue
                d=math.hypot(px-lx,py-ly); emit(wx=px,wy=py,wz=pz,dist=d); lx,ly=px,py

        
        if post_lines:
            blk.append("; -- POST-IRON CUSTOM GCODE --\n")
            blk.extend(post_lines)

        if use_tool:
            blk.append(f"{active_tool}\n")
            if e_abs_here: blk.append(f"G92 E{E_seed:.5f}\n")
        else:
            if e_abs_here and abs(E_acc-E_seed)>1e-9: blk.append(f"G92 E{E_seed:.5f}\n")
        blk.append("; END CustomIroning segment\n")
        return blk, True

    
    for idx,(bm,bt) in enumerate(bands):
        after_idx,E_seed,e_abs_here = insertion_after_band_top(bt)
        log(f"Injecting band {idx+1} after line {after_idx} (band_top={bt:.3f} mm)")
        active_tool = prev_tool_at(after_idx)
        band_blk, ok = build_band_block(idx,bm,bt,E_seed,e_abs_here,active_tool)
        if ok: inject_map.setdefault(after_idx,[]).append(band_blk); bands_built+=1

    final_out=[]
    if -1 in inject_map:
        for blk in inject_map[-1]: final_out.extend(blk)
    for i,line in enumerate(lines):
        final_out.append(line)
        if i in inject_map:
            for blk in inject_map[i]: final_out.extend(blk)

    with open(input_file,"w",encoding="utf-8") as wf: wf.writelines(final_out)
    log(f"Inserted {bands_built} ironing pass(es).")
    log(f"== Done in {time.time()-t_start:.2f}s. Log: {log_file_path} ==")

# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Insert nonplanar ironing passes (single or segmented) into G-code.")
    # General
    p.add_argument("input_file")
    p.add_argument("-debug", action="store_true")
    p.add_argument("-debugReach", action="store_true")
    # Geometry & Sampling
    p.add_argument("-cellSize", type=float, default=0.20)
    p.add_argument("-sampleStep", type=float, default=0.15)
    p.add_argument("-smoothBoxRadius", type=int, default=1)
    p.add_argument("-fillRadius", type=float, default=0.0)
    # Ironing geometry
    p.add_argument("-nozzleDiam", type=float, default=0.40)
    p.add_argument("-ironingWidth", type=float, default=0.40)
    p.add_argument("-ironingHeight", type=float, default=0.05)
    p.add_argument("-ironingPitch", type=float, default=0.40)
    p.add_argument("-angle", type=float, default=0.0)
    p.add_argument("-maxAngleDeg", type=float, default=60.0)
    # Z envelope & reach
    p.add_argument("-zOffset", type=float, default=-0.05)
    p.add_argument("-maxDepth", type=float, default=0.30)
    p.add_argument("-reachDepth", type=float, default=1.80)
    p.add_argument("-erodeRadius", type=float, default=0.0)
    # Side/Ball
    p.add_argument("-shaftCheck", action="store_true")
    p.add_argument("-shaftRadius", type=float, default=4.0)
    p.add_argument("-ballRadius", type=float, default=0.0)
    # Extrusion & Feed
    p.add_argument("-feedrate", type=float, default=1200.0)   # mm/min
    p.add_argument("-flowPercent", type=float, default=5.0)
    p.add_argument("-filamentDiam", type=float, default=1.75)
    # Strategy
    p.add_argument("-strategy", choices=["scanline","adaptive","concentric"], default="scanline")
    p.add_argument("-stepover", type=float, default=0.60)
    p.add_argument("-minLoop", type=float, default=0.3)
    # Emission / control
    p.add_argument("-splitSegments", action="store_true")
    p.add_argument("-ironTool", type=str, default=None)
    p.add_argument("-bandOrder", choices=["topdown","bottomup"], default="topdown")
    p.add_argument("-collisionScope", choices=["global","printed"], default="global")
    # Adaptive banding
    p.add_argument("-adaptiveSplit", action="store_true")
    p.add_argument("-bandStep", type=float, default=0.0)
    p.add_argument("-minNewFrac", type=float, default=0.02)
    p.add_argument("-minNewAbs", type=int, default=25)
    # Travel 
    p.add_argument("-travelClearance", type=float, default=0.2)
    # Per-band path regeneration
    p.add_argument("-regenPerBandPaths", action="store_true")
    # Tool offsets
    p.add_argument("-ironOffsetX", type=float, default=0)
    p.add_argument("-ironOffsetY", type=float, default=0)
    p.add_argument("-ironOffsetZ", type=float, default=0)
    #p.add_argument("-ironOffsetX", type=float, default=53.0)
    #p.add_argument("-ironOffsetY", type=float, default=34.0)
    #p.add_argument("-ironOffsetZ", type=float, default=30.0)
    # NEW: custom snippets
    p.add_argument("-preIron", type=str, default='', help="Inline string or @file.gcode to emit before ironing")
    p.add_argument("-postIron", type=str, default='', help="Inline string or @file.gcode to emit after ironing")
    #p.add_argument("-preIron", type=str, default='M118 P2 "G1 Z-32"', help="Inline string or @file.gcode to emit before ironing")
    #p.add_argument("-postIron", type=str, default='M118 P2 "G1 Z32"', help="Inline string or @file.gcode to emit after ironing")
    p.add_argument("-vizOut", type=str, default="D:/Roman/Gitrepos/NonPlanarIroning/newday/newdayFr/viz")

    args = p.parse_args()

    process_gcode(
        input_file=args.input_file,
        viz_out_dir=(args.vizOut if args.vizOut else None),
        cell_size=args.cellSize,
        nozzle_diam=args.nozzleDiam,
        ironing_pitch=args.ironingPitch,
        z_offset=args.zOffset,
        feedrate=args.feedrate,
        flow_percent=args.flowPercent,
        filament_diam=args.filamentDiam,
        ironing_width=args.ironingWidth,
        ironing_height=args.ironingHeight,
        smooth_box_radius=args.smoothBoxRadius,
        erode_radius=args.erodeRadius if args.erodeRadius > 0 else args.nozzleDiam*0.5,
        angle_deg=args.angle,
        sample_step=args.sampleStep,
        max_angle_deg=args.maxAngleDeg,
        max_depth=args.maxDepth,
        reach_depth=args.reachDepth,
        shaft_radius=args.shaftRadius,
        shaft_check=args.shaftCheck,
        debug=args.debug,
        debug_reach=args.debugReach,
        fill_radius=args.fillRadius,
        strategy=args.strategy,
        stepover=args.stepover,
        min_loop=args.minLoop,
        split_segments=args.splitSegments,
        iron_tool=args.ironTool,
        band_order=args.bandOrder,
        ball_radius=args.ballRadius,
        collision_scope=args.collisionScope,
        adaptive_split=args.adaptiveSplit,
        band_step=args.bandStep,
        min_new_frac=args.minNewFrac,
        min_new_abs=args.minNewAbs,
        travel_clearance=args.travelClearance,
        regen_per_band_paths=args.regenPerBandPaths,
        iron_offset_x=args.ironOffsetX,
        iron_offset_y=args.ironOffsetY,
        iron_offset_z=args.ironOffsetZ,
        pre_iron_snippet=args.preIron,
        post_iron_snippet=args.postIron,
    )
