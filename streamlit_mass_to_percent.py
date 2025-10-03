# app.py
import io
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend (Streamlit Cloud safe)
import matplotlib.pyplot as plt
import streamlit as st

# ---------- optional element data via 'periodictable' ----------
try:
    import periodictable as pt
except Exception:
    pt = None  # fallback dicts will be used

# Fallback atomic weights (u)
FALLBACK_ATOMIC_WEIGHTS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "Si": 28.085, "Mo": 95.95, "S": 32.06, "Cl": 35.45,
    "Cu": 63.546, "Zn": 65.38, "Sn": 118.71, "Se": 78.971,
    "Ag": 107.8682, "Au": 196.96657, "Pb": 207.2, "I": 126.90447, "Br": 79.904, "Cs": 132.90545,
}

# ================================================================
#                         FILE LOADERS
# ================================================================
def _convert_numeric_inplace(df: pd.DataFrame) -> None:
    """Convert strings (comma decimal) to floats wherever sensible."""
    for c in list(df.columns):
        s = df[c]
        if s.dtype == object:
            ser = s.astype(str).str.replace(",", ".", regex=False)\
                                .str.replace("\u00A0", "", regex=False)\
                                .str.strip()
            nums = pd.to_numeric(ser, errors="coerce")
            if nums.notna().sum() >= max(3, int(0.5 * len(nums))):
                df[c] = nums

def _drop_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "Spectrum" not in df.columns:
        return df
    mask = df["Spectrum"].astype(str).str.match(r"\d+(\.spx)?$", na=False)
    return df[mask].reset_index(drop=True) if mask.any() else df

# ----------------- TXT reader -----------------
def _read_txt_table(src: Union[str, os.PathLike, io.IOBase]) -> pd.DataFrame:
    # read text
    if hasattr(src, "read"):
        src.seek(0)
        text = src.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
    else:
        text = Path(src).read_text(encoding="utf-8", errors="ignore")

    lines = [ln.rstrip("\r\n") for ln in text.splitlines() if ln.strip() != ""]
    if len(lines) < 3:
        raise ValueError("TXT file too short / malformed.")

    header_cols = lines[1]
    data_lines = []
    for ln in lines[2:]:
        if re.match(r"\s*(Mean value|Std dev)", ln):
            break
        data_lines.append(ln.strip())
    if not data_lines:
        raise ValueError("TXT contains no data rows.")

    # Data rows: split by >= 2 spaces
    rows = [re.split(r"\s{2,}", ln) for ln in data_lines]
    ncols = max(len(r) for r in rows)
    rows = [(r + [""] * (ncols - len(r)))[:ncols] for r in rows]

    # Header tokens
    tokens = re.findall(r"(Spectrum|Thickn\.\s*\[[^\]]+\]|[A-Z][a-z]?\s*\[%\])", header_cols)
    # unlabeled substrate column after Spectrum?
    if tokens and tokens[0] == "Spectrum" and len(tokens) == ncols - 1:
        tokens = [tokens[0], "Substrate"] + tokens[1:]

    # Normalize tokens
    norm = []
    for t in tokens:
        if t in ("Spectrum", "Substrate"):
            norm.append(t)
        elif t.startswith("Thickn"):
            norm.append("Thickn. [nm]")
        else:
            el = re.findall(r"[A-Z][a-z]?", t)[0]
            norm.append(f"{el} %mass")

    # Build canonical columns while walking thickness markers
    cols, current, layer_idx = [], "Substrate", 0
    for tok in norm:
        if tok == "Spectrum":
            cols.append("Spectrum"); current = "Substrate"
        elif tok == "Substrate":
            cols.append("Substrate"); current = "Substrate"
        elif tok == "Thickn. [nm]":
            layer_idx += 1
            cols.append(f"Layer {layer_idx} Thickness [nm]")
            current = f"Layer {layer_idx}"
        elif tok.endswith(" %mass"):
            el = tok.split()[0]
            cols.append(f"{current} {el} %mass" if current != "Substrate" else f"Substrate {el} %mass")
        else:
            cols.append(tok or "col")

    if len(cols) != ncols:
        cols = (cols + [f"col{i}" for i in range(len(cols) + 1, ncols + 1)])[:ncols]

    df = pd.DataFrame(rows, columns=cols)
    _convert_numeric_inplace(df)
    return df

# ----------------- Excel reader (robust: xlsx/xls/html) -----------------
def _sniff_excel_kind_from_bytes(b: bytes) -> Optional[str]:
    """Return 'xlsx' (zip), 'xls' (OLE2), 'html' (HTML-ish), or None."""
    if not b:
        return None
    head8 = b[:8]
    if head8.startswith(b"PK\x03\x04"):
        return "xlsx"
    if head8.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        return "xls"
    head512 = b[:512].lower()
    if b"<html" in head512 or b"<table" in head512:
        return "html"
    return None

def _read_xls_with_xlrd(data: bytes) -> pd.DataFrame:
    """Read legacy .xls bytes via xlrd directly (bypasses pandas engine/version checks)."""
    import xlrd  # ensure xlrd==1.2.0 installed
    book = xlrd.open_workbook(file_contents=data, on_demand=True)
    sheet = book.sheet_by_index(0)
    rows = []
    nrows, ncols = sheet.nrows, sheet.ncols
    for r in range(nrows):
        row = []
        for c in range(ncols):
            val = sheet.cell_value(r, c)
            row.append("" if val is None else str(val))
        rows.append(row)
    return pd.DataFrame(rows)

def _read_excel_table(src: Union[str, os.PathLike, io.IOBase]) -> pd.DataFrame:
    """Robust Excel/HTML loader for Streamlit uploads."""
    # clone bytes
    if hasattr(src, "read"):
        try:
            pos = src.tell()
        except Exception:
            pos = None
        src.seek(0)
        data = src.read()
        if isinstance(data, str):
            data = data.encode("utf-8", errors="ignore")
        if pos is not None:
            src.seek(pos)
    else:
        with open(src, "rb") as f:
            data = f.read()

    kind = _sniff_excel_kind_from_bytes(data)

    raw = None
    tried = []

    # XLSX → openpyxl
    if kind == "xlsx":
        try:
            raw = pd.read_excel(io.BytesIO(data), header=None, dtype=str, engine="openpyxl", sheet_name=0)
        except Exception as e:
            tried.append(("openpyxl(xlsx)", e))

    # XLS → xlrd-direct, then pyexcel-xls
    if raw is None and (kind == "xls" or kind is None):
        try:
            raw = _read_xls_with_xlrd(data)
        except Exception as e:
            tried.append(("xlrd-direct(xls)", e))
            try:
                from pyexcel_xls import get_data as xls_get_data
                book = xls_get_data(io.BytesIO(data))
                sheet = next((sh for sh in book.values() if len(sh) > 0), [])
                if not sheet:
                    raise ValueError("pyexcel_xls found no sheets.")
                raw = pd.DataFrame(sheet)
            except Exception as e2:
                tried.append(("pyexcel-xls(xls)", e2))

    # HTML-ish → read_html
    if raw is None and (kind == "html" or kind is None):
        try:
            tables = pd.read_html(io.BytesIO(data))
            raw = tables[0]
        except Exception as e:
            tried.append(("read_html", e))

    if raw is None:
        msg = "Could not read Excel file with available engines:\n" + "\n".join(f"- {eng}: {exc}" for eng, exc in tried)
        raise RuntimeError(msg)

    raw = raw.astype(str)

    # Find header row containing 'Spectrum'
    h1 = None
    for i in range(min(20, len(raw))):
        row = [str(x) if pd.notna(x) else "" for x in raw.iloc[i].tolist()]
        if any(re.search(r"\bSpectrum\b", cell, re.IGNORECASE) for cell in row):
            h1 = i
            break
    if h1 is None:
        raise ValueError("Could not find header row containing 'Spectrum' in this file.")

    hdr_cells = [(str(x) if pd.notna(x) else "").strip() for x in raw.iloc[h1].tolist()]

    # Tokenize to normalized tokens
    tokens = []
    for cell in hdr_cells:
        if re.fullmatch(r"Spectrum", cell):
            tokens.append("Spectrum")
        elif cell == "" or cell.lower() in ("substrate", "base"):
            tokens.append("Substrate" if cell.lower() in ("substrate", "base") else "")
        elif re.search(r"Thickn", cell, re.IGNORECASE) and re.search(r"nm", cell, re.IGNORECASE):
            tokens.append("Thickn. [nm]")
        elif re.fullmatch(r"[A-Z][a-z]?\s*(\[%\]|%)", cell):
            el = re.findall(r"[A-Z][a-z]?", cell)[0]
            tokens.append(f"{el} %mass")
        else:
            tokens.append(cell)

    # If 2nd column blank but below says 'Base' → Substrate
    if len(tokens) > 1 and tokens[1] == "" and h1 + 1 < len(raw):
        col1_vals = [str(v).strip().lower() for v in raw.iloc[h1 + 1 : min(h1 + 10, len(raw)), 1].tolist() if pd.notna(v)]
        if any(v == "base" for v in col1_vals):
            tokens[1] = "Substrate"

    # Build canonical names
    cols, current, layer_idx = [], "Substrate", 0
    for tok in tokens:
        if tok == "Spectrum":
            cols.append("Spectrum"); current = "Substrate"
        elif tok == "Substrate":
            cols.append("Substrate"); current = "Substrate"
        elif tok == "Thickn. [nm]":
            layer_idx += 1
            cols.append(f"Layer {layer_idx} Thickness [nm]")
            current = f"Layer {layer_idx}"
        elif tok.endswith(" %mass"):
            el = tok.split()[0]
            cols.append(f"{current} {el} %mass" if current != "Substrate" else f"Substrate {el} %mass")
        else:
            cols.append(tok)

    data = raw.iloc[h1 + 1 :].reset_index(drop=True).copy()
    ncols = data.shape[1]
    if len(cols) != ncols:
        st.warning(f"Header width ({len(cols)}) != data width ({ncols}); padding/truncating.")
        cols = (cols + [f"col{i}" for i in range(len(cols) + 1, ncols + 1)])[:ncols]
    data.columns = cols

    _convert_numeric_inplace(data)
    data = _drop_summary_rows(data)
    return data

def load_xrf_table(source: Union[str, os.PathLike, io.BytesIO, io.StringIO]) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Load .txt / .xls / .xlsx → canonical DataFrame and layers dict."""
    # sniff by extension only to pick TXT vs Excel path
    ext = None
    if isinstance(source, (str, os.PathLike)):
        ext = Path(source).suffix.lower()
    else:
        name = getattr(source, "name", None)
        if name:
            ext = Path(name).suffix.lower()

    if ext == ".txt":
        df = _read_txt_table(source)
    else:
        df = _read_excel_table(source)

    # Build layers dict from canonical names
    layers: Dict[str, Dict] = {"Substrate": {"thickness": None, "elements": []}}
    for c in df.columns:
        m_sub_el = re.fullmatch(r"Substrate ([A-Z][a-z]?) %mass", c)
        if m_sub_el:
            layers["Substrate"]["elements"].append((m_sub_el.group(1), c))
        if re.fullmatch(r"Substrate Thickness \[nm\]", c):
            layers["Substrate"]["thickness"] = c

        m_li_th = re.fullmatch(r"Layer (\d+) Thickness \[nm\]", c)
        if m_li_th:
            L = f"Layer {m_li_th.group(1)}"
            layers.setdefault(L, {"thickness": None, "elements": []})
            layers[L]["thickness"] = c

        m_li_el = re.fullmatch(r"Layer (\d+) ([A-Z][a-z]?) %mass", c)
        if m_li_el:
            L = f"Layer {m_li_el.group(1)}"
            layers.setdefault(L, {"thickness": None, "elements": []})
            layers[L]["elements"].append((m_li_el.group(2), c))
    return df, layers

# ================================================================
#                   MASS%  →  AT.%  (per layer)
# ================================================================
def atomic_mass_u(symbol: str) -> float:
    if pt is not None:
        try:
            el = getattr(pt, symbol)
            m = float(el.mass) if getattr(el, "mass", None) else None
            if m and m > 0:
                return m
        except Exception:
            pass
    return float(FALLBACK_ATOMIC_WEIGHTS.get(symbol, 1.0))

def add_atomic_percent_by_layer(df: pd.DataFrame, layers: Dict[str, Dict]):
    """Create '<Group> <El> [at%]' columns normalized within each group."""
    masses_used, created_cols = {}, []
    for group, info in layers.items():
        elems = info.get("elements", [])
        if not elems:
            continue
        n_parts, el_order = [], []
        for sym, mass_col in elems:
            m_u = atomic_mass_u(sym)
            masses_used[sym] = m_u
            s = pd.to_numeric(df[mass_col], errors="coerce")
            n_parts.append(s / m_u)
            el_order.append(sym)

        denom = None
        for s in n_parts:
            denom = s if denom is None else (denom + s)
        valid = denom > 0

        for sym, n_i in zip(el_order, n_parts):
            out_col = f"{group} {sym} [at%]"
            df[out_col] = pd.NA
            df.loc[valid, out_col] = 100.0 * (n_i[valid] / denom[valid])
            created_cols.append(out_col)
    return df, masses_used, created_cols

# ================================================================
#                  GRID UTILS & PLOTTING
# ================================================================
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

def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    r = pd.Series(np.nan, index=a.index)
    m = (b != 0) & a.notna() & b.notna()
    r.loc[m] = (a[m] / b[m]).astype(float)
    return r

# ================================================================
#                          STREAMLIT UI
# ================================================================
st.set_page_config(page_title="XRF mass% → at.% & Thickness Heatmaps", layout="wide")
st.title("XRF mass% → atomic% (per layer) & thickness heatmaps")

with st.sidebar:
    st.header("1) Load data")
    up = st.file_uploader("Upload XRF file", type=["txt", "xls", "xlsx"])
    drop_summary = st.checkbox("Drop non-numeric Spectrum rows (Mean/Std)", True)

    st.header("2) Grid mapping")
    NX = st.number_input("NX (columns, x)", 1, 2000, 24)
    NY = st.number_input("NY (rows, y)", 1, 2000, 22)
    snake = st.checkbox("Snake (zig-zag by row)?", False)
    flip_x = st.checkbox("Flip X", False)
    flip_y = st.checkbox("Flip Y", False)

    st.header("3) Color scaling")
    use_robust = st.checkbox("Use robust 2–98% limits", True)
    cmap = st.selectbox("Colormap", ["viridis", "plasma", "inferno", "magma", "cividis"], index=0)

    st.header("4) Layer-2 correction (at.%)")
    st.caption("Multiply Layer-2 Cu/Zn/Sn at.% by factors, then re-normalize within Layer-2.")
    cu_corr = st.number_input("Cu factor", min_value=0.0, value=0.96, step=0.01)
    zn_corr = st.number_input("Zn factor", min_value=0.0, value=0.97, step=0.01)
    sn_corr = st.number_input("Sn factor", min_value=0.0, value=1.15, step=0.01)

if up is None:
    st.info("Upload a .txt / .xls / .xlsx to begin.")
    st.stop()

# ---- load data ----
try:
    df, layers = load_xrf_table(up)
except Exception as e:
    st.error("Failed to load file.")
    st.exception(e)
    st.stop()

if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("Parsed dataframe is empty/None. Check the file or header rows.")
    st.stop()

if drop_summary:
    df = _drop_summary_rows(df)

with st.expander("Preview & detected layers", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    st.json(layers)

# ---- mass% -> at% per layer ----
df, masses_used, new_at_cols = add_atomic_percent_by_layer(df, layers)

# ---- optional Layer-2 correction & renorm ----
l2 = layers.get("Layer 2")
if l2 and l2.get("elements"):
    l2_syms = [sym for sym, _col in l2["elements"]]
    l2_at_cols = [f"Layer 2 {sym} [at%]" for sym in l2_syms if f"Layer 2 {sym} [at%]" in df.columns]
    factors = {"Cu": cu_corr, "Zn": zn_corr, "Sn": sn_corr}
    for sym in ("Cu", "Zn", "Sn"):
        col = f"Layer 2 {sym} [at%]"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * float(factors.get(sym, 1.0))
    if l2_at_cols:
        denom = None
        for c in l2_at_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            denom = s if denom is None else (denom + s)
        valid = denom > 0
        for c in l2_at_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            df.loc[valid, c] = 100.0 * (s[valid] / denom[valid])

# ---- common ratios (Layer 2) ----
if all(f"Layer 2 {x} [at%]" in df.columns for x in ("Cu", "Zn", "Sn")):
    df["Layer 2 Cu/(Zn+Sn)"] = safe_ratio(df["Layer 2 Cu [at%]"], df["Layer 2 Zn [at%]"] + df["Layer 2 Sn [at%]"])
    df["Layer 2 Zn/Sn"] = safe_ratio(df["Layer 2 Zn [at%]"], df["Layer 2 Sn [at%]"])
    df["Layer 2 Cu/Sn"] = safe_ratio(df["Layer 2 Cu [at%]"], df["Layer 2 Sn [at%]"])
if all(f"Layer 2 {x} [at%]" in df.columns for x in ("Cu", "Zn", "Sn", "Se")):
    df["Layer 2 Se/(Cu+Zn+Sn)"] = safe_ratio(
        df["Layer 2 Se [at%]"],
        df["Layer 2 Cu [at%]"] + df["Layer 2 Zn [at%]"] + df["Layer 2 Sn [at%]"]
    )

# ---- heatmap selection ----
st.subheader("Heatmap")
plot_candidates: List[str] = []
plot_candidates += [c for c in df.columns if re.search(r"\[at%\]$", str(c))]
plot_candidates += [c for c in df.columns if re.fullmatch(r"(Substrate|Layer \d+) Thickness \[nm\]", str(c))]
for c in ["Layer 2 Cu/(Zn+Sn)", "Layer 2 Zn/Sn", "Layer 2 Cu/Sn", "Layer 2 Se/(Cu+Zn+Sn)"]:
    if c in df.columns:
        plot_candidates.append(c)
if not plot_candidates:
    plot_candidates = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]

plot_col = st.selectbox("Column to plot", options=sorted(set(plot_candidates)), index=0)

# ---- grid + plot ----
vals = pd.to_numeric(df[plot_col], errors="coerce").to_numpy()
Z = to_grid(vals, nx=NX, ny=NY, snake=snake, flip_x=flip_x, flip_y=flip_y)

# auto limits (robust or full range)
if use_robust:
    vmin_auto, vmax_auto = robust_minmax(Z)
else:
    finite = Z[np.isfinite(Z)]
    vmin_auto = float(np.nanmin(finite)) if finite.size else None
    vmax_auto = float(np.nanmax(finite)) if finite.size else None

# --- manual color scale override UI ---
st.markdown("#### Color scale")
manual_scale = st.checkbox("Set vmin/vmax manually", value=False)
if manual_scale:
    # sensible defaults come from the auto limits we just computed
    vmin_default = vmin_auto if vmin_auto is not None else 0.0
    vmax_default = vmax_auto if vmax_auto is not None else 1.0
    vmin_user = st.number_input("vmin", value=float(vmin_default), format="%.6g")
    vmax_user = st.number_input("vmax", value=float(vmax_default), format="%.6g")
    if vmax_user <= vmin_user:
        st.warning("`vmax` must be greater than `vmin`. Falling back to auto limits.")
        vmin_plot, vmax_plot = vmin_auto, vmax_auto
    else:
        vmin_plot, vmax_plot = float(vmin_user), float(vmax_user)
else:
    vmin_plot, vmax_plot = vmin_auto, vmax_auto

# ---- draw figure ----
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
im = ax.imshow(
    Z,
    origin="lower",
    extent=[0, NX, 0, NY],
    aspect="equal",
    interpolation="nearest",
    cmap=cmap,
    vmin=vmin_plot,
    vmax=vmax_plot,
)
cb = fig.colorbar(im, ax=ax)
cb.set_label(plot_col)
ax.set_xlabel("x (index)")
ax.set_ylabel("y (index)")
ax.set_title(f"{plot_col} (row-major{' snake' if snake else ''})")
st.pyplot(fig, clear_figure=True)


# ---- save augmented CSV ----
st.subheader("Save augmented CSV")
fname = st.text_input("Output filename", value="xrf_with_at_percent_and_thickness.csv")
if st.button("Prepare download"):
    out_bytes = io.BytesIO()
    df.to_csv(out_bytes, index=False)
    st.download_button("Download CSV", data=out_bytes.getvalue(), file_name=fname, mime="text/csv")
