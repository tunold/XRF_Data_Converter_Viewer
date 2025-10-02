# app.py
import io
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend for Streamlit Cloud
import matplotlib.pyplot as plt
import streamlit as st

# ---------- optional element data via 'periodictable' ----------
try:
    import periodictable as pt
except Exception:
    pt = None  # fallback dicts will be used

# Fallbacks (extend as needed)
FALLBACK_ATOMIC_WEIGHTS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "Si": 28.085, "Mo": 95.95,
    "Cu": 63.546, "Zn": 65.38, "Sn": 118.71, "Se": 78.971,
    "Pb": 207.2, "I": 126.90447, "Br": 79.904, "Cs": 132.90545,
}
FALLBACK_DENSITIES = {  # g/cm³
    "Si": 2.329, "Mo": 10.28, "Cu": 8.96, "Zn": 7.14, "Sn": 7.31, "Se": 4.82,
    "Pb": 11.34, "I": 4.93, "Br": 3.12, "Cs": 1.93,
}

ELEMENTS_HINT = ["Cu", "Zn", "Sn", "Se", "Mo", "Si", "Pb", "I", "Br", "Cs"]

# --------------------- header utils (robust loader) ---------------------
def _clean_token(s: object) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ").strip()
    s = re.sub(r"^Unnamed:.*", "", s, flags=re.IGNORECASE).strip()
    if s.lower() == "nan":
        s = ""
    s = s.replace("[", " ").replace("]", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _make_unique(names: List[str]) -> List[str]:
    out = []
    counts = defaultdict(int)
    for n in names:
        base = n if n != "" else "col"
        counts[base] += 1
        out.append(base if counts[base] == 1 else f"{base}__{counts[base]}")
    return out

def _read_raw(uploaded_file, sep=None) -> pd.DataFrame:
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, header=None, dtype=str, engine="python", sep=sep)

def read_csv_with_possible_two_row_header_streamlit(uploaded_file) -> pd.DataFrame:
    """Robust CSV reader:
       - tries autodetect, ';', ',' separators
       - detects 1–2 row Excel-style header
       - forward-fills top header (group names)
       - ensures unique column names
       - renames % → %mass
    """
    last_err = None
    for sep in (None, ";", ","):
        try:
            raw = _read_raw(uploaded_file, sep=sep)
            break
        except Exception as e:
            last_err = e
            raw = None
    if raw is None:
        raise RuntimeError(f"Failed to read CSV (autodetect, ';', ','). Last error: {last_err}")

    # Detect header rows
    hdr0 = hdr1 = None
    max_scan = min(10, len(raw))
    for i in range(max_scan):
        row = [_clean_token(x) for x in raw.iloc[i].tolist()]
        if any(re.search(r"\bSpectrum\b", v, re.IGNORECASE) for v in row):
            hdr0 = i
            if i + 1 < len(raw):
                nxt = [_clean_token(x) for x in raw.iloc[i+1].tolist()]
                if sum("%" in v for v in nxt) >= 2 or sum(any(sym in v for sym in ELEMENTS_HINT) for v in nxt) >= 2:
                    hdr1 = i + 1
            break
    if hdr0 is None:
        for i in range(max_scan):
            row = [_clean_token(x) for x in raw.iloc[i].tolist()]
            if any(v != "" for v in row):
                hdr0 = i
                if i + 1 < len(raw):
                    nxt = [_clean_token(x) for x in raw.iloc[i+1].tolist()]
                    if sum("%" in v for v in nxt) >= 2:
                        hdr1 = i + 1
                break

    # Build column names with forward-fill on the top row
    ncols = raw.shape[1]
    top = [_clean_token(raw.iat[hdr0, j]) if hdr0 is not None else "" for j in range(ncols)]
    ff_top, last = [], ""
    for t in top:
        if t == "":
            ff_top.append(last)
        else:
            ff_top.append(t)
            last = t
    bot = [_clean_token(raw.iat[hdr1, j]) for j in range(ncols)] if hdr1 is not None else [""] * ncols

    cols = []
    for j in range(ncols):
        t, b = ff_top[j], bot[j]
        name = f"{t} {b}".strip() if (t and b) else (t or b)
        name = _clean_token(name) or f"col{j+1}"
        cols.append(name)
    cols = _make_unique(cols)

    data_start = (hdr1 if hdr1 is not None else hdr0) + 1
    df = raw.iloc[data_start:].reset_index(drop=True).copy()
    df.columns = cols
    df = df.dropna(axis=1, how="all")

    # Mass%: rename "... % " -> "... %mass"
    mass_ren = {c: c.replace("%", "%mass")
                for c in df.columns
                if re.search(r"\b%\b", c) and "%mass" not in c}
    if mass_ren:
        df = df.rename(columns=mass_ren)

    return df

# --------------------- detection & conversion ---------------------
def is_thickness_col(name: str) -> bool:
    return bool(re.search(r"thick|thickn|thickness|nm|µm|um", str(name), re.IGNORECASE))

def detect_mass_percent_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Return {element_symbol -> column_name} for columns that look like mass-%."""
    mapping = {}
    cols = list(df.columns)
    if pt is not None:
        valid_syms = {el.symbol for el in pt.elements if getattr(el, "symbol", None)}
    else:
        valid_syms = set(FALLBACK_ATOMIC_WEIGHTS.keys())
    for c in cols:
        if is_thickness_col(c) or "%mass" not in str(c):
            continue
        tokens = re.findall(r"\b([A-Z][a-z]?)\b", str(c))
        for t in tokens:
            if t in valid_syms:
                mapping.setdefault(t, c)
                break
    return mapping

def get_element_data(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    if pt is not None:
        try:
            el = getattr(pt, symbol)
            am = float(el.mass) if getattr(el, "mass", None) else None
            den = float(el.density) if getattr(el, "density", None) else None
            return am, den
        except Exception:
            pass
    return FALLBACK_ATOMIC_WEIGHTS.get(symbol), FALLBACK_DENSITIES.get(symbol)

def add_layer2_atomic_percent(df: pd.DataFrame,
                              layer2_masspct: Dict[str, str],
                              corr_factors: Dict[str, float]) -> Tuple[pd.DataFrame, Dict[str, Tuple[Optional[float], Optional[float]]]]:
    """Compute Layer 2 at.% only from Cu,Zn,Sn,Se mass-% columns belonging to Layer 2.
       Apply correction factors to at.% (Cu,Zn,Sn) and renormalize over {Cu,Zn,Sn,Se}.
       Creates: 'Layer 2 Cu [at%]', ..., and ratios.
    """
    target = [el for el in ["Cu", "Zn", "Sn", "Se"] if el in layer2_masspct]
    if not target:
        return df, {}

    props = {el: get_element_data(el) for el in target}

    # Convert to numeric mass%
    for el in target:
        df[layer2_masspct[el]] = pd.to_numeric(df[layer2_masspct[el]], errors="coerce")

    # m_i / A_i -> n_i
    parts = {}
    for el in target:
        am, _ = props[el]
        if not am:
            st.warning(f"Atomic mass not found for {el}; using 1.0 u (will distort at.%).")
            am = 1.0
        parts[el] = df[layer2_masspct[el]] / float(am)

    # Normalize within the set (Cu,Zn,Sn,Se)
    denom = None
    for el in target:
        denom = parts[el] if denom is None else denom + parts[el]
    at = {el: 100.0 * (parts[el] / denom) for el in target}

    # Apply correction to at.% for Cu,Zn,Sn, then renormalize again within the set
    for el in ["Cu", "Zn", "Sn"]:
        if el in at:
            at[el] = at[el] * float(corr_factors.get(el, 1.0))

    denom2 = None
    for el in target:
        denom2 = at[el] if denom2 is None else denom2 + at[el]
    valid = denom2 > 0
    for el in target:
        colname = f"Layer 2 {el} [at%]"
        df[colname] = np.nan
        df.loc[valid, colname] = 100.0 * (at[el][valid] / denom2[valid])

    # Ratios based on corrected Layer 2 at.%
    def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        r = pd.Series(np.nan, index=a.index)
        m = (b != 0) & a.notna() & b.notna()
        r.loc[m] = (a[m] / b[m]).astype(float)
        return r

    a = {el: f"Layer 2 {el} [at%]" for el in target}
    if all(k in a for k in ("Cu", "Zn", "Sn")):
        df["Layer 2 Cu/(Zn+Sn)"] = safe_ratio(df[a["Cu"]], df[a["Zn"]] + df[a["Sn"]])
        df["Layer 2 Zn/Sn"] = safe_ratio(df[a["Zn"]], df[a["Sn"]])
        df["Layer 2 Cu/Sn"] = safe_ratio(df[a["Cu"]], df[a["Sn"]])
    if all(k in a for k in ("Cu", "Zn", "Sn", "Se")):
        df["Layer 2 Se/(Cu+Zn+Sn)"] = safe_ratio(df[a["Se"]], df[a["Cu"]] + df[a["Zn"]] + df[a["Sn"]])

    return df, props

# --------------------- thickness candidates + canonicalization ---------------------
def thickness_candidates(df: pd.DataFrame) -> List[str]:
    """List all columns that look thickness-like (very permissive)."""
    cands = []
    for c in df.columns:
        if (re.search(r"thick|thickn|thickness", c, re.IGNORECASE)
            and re.search(r"(nm|µm|um)", c, re.IGNORECASE)):
            cands.append(c)
    return cands

def preselect_thickness(cands: List[str], token: str) -> Optional[str]:
    """Pick a candidate containing the token (Layer 1 / Layer 2 / Substrate / Base)."""
    pri = [x for x in cands if re.search(token, x, re.IGNORECASE)]
    if pri:
        # prefer the shortest name
        return sorted(pri, key=len)[0]
    return None

def add_canonical_thickness(df: pd.DataFrame,
                            sub_col: Optional[str],
                            l1_col: Optional[str],
                            l2_col: Optional[str],
                            l1_has_mo: bool) -> pd.DataFrame:
    out = df.copy()
    if sub_col:
        out["Substrate Thickness [nm]"] = pd.to_numeric(out[sub_col], errors="coerce")
    if l1_col:
        name_l1 = "Layer 1 Mo Thickness [nm]" if l1_has_mo else "Layer 1 Thickness [nm]"
        out[name_l1] = pd.to_numeric(out[l1_col], errors="coerce")
    if l2_col:
        out["Layer 2 Thickness [nm]"] = pd.to_numeric(out[l2_col], errors="coerce")
    return out

# --------------------- grid & plotting ---------------------
def to_grid(vals: np.ndarray, nx: int, ny: int, snake: bool, flip_x: bool, flip_y: bool) -> np.ndarray:
    n = min(vals.size, nx * ny)
    vals = vals[:n]
    k = np.arange(n)
    y = k // nx
    x = k % nx
    if snake:
        x = np.where(y % 2 == 0, x, (nx - 1 - x))
    if flip_x:
        x = (nx - 1) - x
    if flip_y:
        y = (ny - 1) - y
    Z = np.full((ny, nx), np.nan, dtype=float)
    Z[y, x] = vals
    return Z

def robust_minmax(a: np.ndarray, p_lo=2, p_hi=98):
    good = a[np.isfinite(a)]
    if good.size == 0:
        return None, None
    lo, hi = np.nanpercentile(good, [p_lo, p_hi])
    return float(lo), float(hi)

# ====================== UI ======================
st.set_page_config(page_title="XRF mass% → Layer 2 at.% & Thickness", layout="wide")
st.title("XRF mass% → Layer 2 atomic% (Cu, Zn, Sn, Se) & thickness heatmaps")

with st.sidebar:
    st.header("1) Load data")
    up = st.file_uploader("Upload CSV", type=["csv"])
    drop_summary = st.checkbox("Drop non-numeric Spectrum rows (e.g., 'Mean value')", True)
    st.caption("Two-row Excel headers handled; duplicate headers de-duplicated.\nMass-% renamed to '%mass' automatically.")

    st.header("2) Grid mapping")
    NX = st.number_input("NX (columns, x)", 1, 1000, 24)
    NY = st.number_input("NY (rows, y)", 1, 1000, 22)
    snake = st.checkbox("Snake (zig-zag by row)?", False)
    flip_x = st.checkbox("Flip X", False)
    flip_y = st.checkbox("Flip Y", False)

    st.header("3) Color scaling")
    use_robust = st.checkbox("Use robust 2–98% limits", True)
    cmap = st.selectbox("Colormap", ["viridis", "plasma", "inferno", "magma", "cividis"], index=0)

    st.header("4) Layer 2 at.% correction factors")
    cu_corr = st.number_input("Cu factor", min_value=0.0, value=0.96, step=0.01)
    zn_corr = st.number_input("Zn factor", min_value=0.0, value=0.97, step=0.01)
    sn_corr = st.number_input("Sn factor", min_value=0.0, value=1.15, step=0.01)

# ---- load & guard -------------------------------------------------
if up is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    df = read_csv_with_possible_two_row_header_streamlit(up)
except Exception as e:
    st.error("Failed to read/parse the CSV.")
    st.exception(e)
    st.stop()

if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    st.error("Parsed dataframe is empty/None. Check the file or header rows.")
    st.stop()

with st.expander("Preview parsed data", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# ---- drop summary rows (only if Spectrum exists) ------------------
spec_cols = [c for c in df.columns if re.search(r"\bSpectrum\b", str(c), re.IGNORECASE)]
if drop_summary and spec_cols:
    sc = spec_cols[0]
    mask = df[sc].astype(str).str.fullmatch(r"\d+")
    df = df[mask].reset_index(drop=True)

# ---- detect mass-% columns & restrict to Layer 2 for Cu/Zn/Sn/Se ----
auto_map = detect_mass_percent_columns(df)  # e.g. {'Cu': 'Layer 2 Cu %mass', ...}

def pick_layer2(mapping: Dict[str, str], el: str) -> Optional[str]:
    c = mapping.get(el)
    if c and re.search(r"\bLayer 2\b", c, re.IGNORECASE):
        return c
    candidates = [col for col in df.columns
                  if re.search(r"\bLayer 2\b", col, re.IGNORECASE)
                  and re.search(rf"\b{el}\b", col)]
    return candidates[0] if candidates else c

layer2_map = {el: pick_layer2(auto_map, el) for el in ["Cu", "Zn", "Sn", "Se"]}

#st.subheader("Layer 2 mass-% columns (%mass)")
#cols_list = list(df.columns)
#sel_map = {}
#for el in ["Cu", "Zn", "Sn", "Se"]:
#    guess = layer2_map.get(el)
#    sel_map[el] = st.selectbox(
#        f"Layer 2 {el} %mass column",
#        options=["<ignore>"] + cols_list,
#        index=(cols_list.index(guess) + 1) if (guess in cols_list) else 0,
#    )

with st.sidebar:
    st.header("Layer 2 mass-% columns (%mass)")
    cols_list = list(df.columns)
    sel_map = {}
    for el in ["Cu", "Zn", "Sn", "Se"]:
        guess = layer2_map.get(el)
        sel_map[el] = st.selectbox(
            f"Layer 2 {el} %mass column",
            options=["<ignore>"] + cols_list,
            index=(cols_list.index(guess) + 1) if (guess in cols_list) else 0,
        )


# Keep chosen
layer2_masspct = {el: col for el, col in sel_map.items() if col and col != "<ignore>"}

# ---- compute Layer 2 at.% (only Cu,Zn,Sn,Se) with correction & renorm ----
df, element_props = add_layer2_atomic_percent(
    df,
    layer2_masspct=layer2_masspct,
    corr_factors={"Cu": cu_corr, "Zn": zn_corr, "Sn": sn_corr}
)

# ---- THICKNESS: find candidates + allow manual mapping ----
cands = thickness_candidates(df)
with st.sidebar:
    st.header("5) Thickness columns")
    st.caption("Pick thickness columns (if your headers are unusual).")
    # preselect heuristics
    pre_sub = preselect_thickness(cands, r"\b(Substrate|Base)\b")
    pre_l1  = preselect_thickness(cands, r"\bLayer\s*1\b")
    pre_l2  = preselect_thickness(cands, r"\bLayer\s*2\b")
    sub_col = st.selectbox("Substrate thickness", ["<none>"] + cands,
                           index=(cands.index(pre_sub) + 1) if (pre_sub in cands) else 0)
    l1_col  = st.selectbox("Layer 1 thickness", ["<none>"] + cands,
                           index=(cands.index(pre_l1) + 1) if (pre_l1 in cands) else 0)
    l2_col  = st.selectbox("Layer 2 thickness", ["<none>"] + cands,
                           index=(cands.index(pre_l2) + 1) if (pre_l2 in cands) else 0)

# Decide Layer-1 Mo naming based on presence of Layer-1 Mo %mass
l1_has_mo = any(re.search(r"^Layer 1 .*Mo .*%mass$", c) for c in df.columns)
df = add_canonical_thickness(
    df,
    sub_col=None if sub_col == "<none>" else sub_col,
    l1_col=None if l1_col == "<none>" else l1_col,
    l2_col=None if l2_col == "<none>" else l2_col,
    l1_has_mo=l1_has_mo
)

# ---- choose column to plot (Layer 2 at.%, ratios, and thickness) ----
st.subheader("Heatmap")
plot_candidates = []

# Layer 2 at.% & ratios
plot_candidates += [c for c in df.columns if re.match(r"^Layer 2 .*?\[at%\]$", c)]
for c in ["Layer 2 Cu/(Zn+Sn)", "Layer 2 Zn/Sn", "Layer 2 Cu/Sn", "Layer 2 Se/(Cu+Zn+Sn)"]:
    if c in df.columns:
        plot_candidates.append(c)

# Thickness (canonical; appear if selected above)
for c in ["Substrate Thickness [nm]", "Layer 1 Mo Thickness [nm]", "Layer 1 Thickness [nm]", "Layer 2 Thickness [nm]"]:
    if c in df.columns and c not in plot_candidates:
        plot_candidates.append(c)

# If none yet, also allow any numeric column
if not plot_candidates:
    plot_candidates = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]

plot_col = st.selectbox("Column to plot", options=sorted(set(plot_candidates)), index=0)

# ---- grid + plot (compact figure) ----
vals = pd.to_numeric(df[plot_col], errors="coerce").to_numpy()
Z = to_grid(vals, nx=NX, ny=NY, snake=snake, flip_x=flip_x, flip_y=flip_y)
vmin_, vmax_ = (robust_minmax(Z) if use_robust else (None, None))

fig = plt.figure(figsize=(5, 4))  # smaller plot
ax = fig.add_subplot(111)
im = ax.imshow(Z, origin="lower", extent=[0, NX, 0, NY],
               aspect="equal", interpolation="nearest", cmap=cmap,
               vmin=vmin_, vmax=vmax_)
cb = fig.colorbar(im, ax=ax)
cb.set_label(plot_col)
ax.set_xlabel("x (index)")
ax.set_ylabel("y (index)")
ax.set_title(f"{plot_col} (row-major{' snake' if snake else ''})")
st.pyplot(fig, clear_figure=True)

# ---- save augmented CSV ----
st.subheader("Save augmented CSV")
fname = st.text_input("Output filename", value="xrf_with_layer2_at_percent_and_thickness.csv")
if st.button("Prepare download"):
    out_bytes = io.BytesIO()
    df.to_csv(out_bytes, index=False)
    st.download_button("Download CSV", data=out_bytes.getvalue(), file_name=fname, mime="text/csv")

