"""Streamlit GUI for cytoscan.

Layout:
  sidebar:  folder picker, research config (read/write config.yaml), run button
  main:     title, metric strip, view selector + frame nav, visuals + data

Launch: `cytoscan gui`.
"""
from __future__ import annotations

# lock matplotlib to Agg before any cytoscan import pulls in pyplot
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg", force=True)

import html
import io
import logging
import re
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml

from cytoscan.config import Config
from cytoscan.pipeline import run_pipeline
from cytoscan.preprocessing import scaffold_experiment

try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("cytoscan")
except Exception:
    _VERSION = "unknown"


# ─── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="cytoscan",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── styling ───────────────────────────────────────────────────────────────────
_CSS = """
<style>
  /* hide streamlit chrome we don't need */
  [data-testid="stToolbar"], [data-testid="stStatusWidget"],
  [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
  header[data-testid="stHeader"] { background: transparent; }

  /* force sidebar open and prevent collapse */
  section[data-testid="stSidebar"] {
    min-width: 320px !important; max-width: 320px !important;
    transform: translateX(0) !important; visibility: visible !important;
  }
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
  section[data-testid="stSidebar"] button[kind="header"],
  section[data-testid="stSidebar"] button[aria-label*="ollaps"] {
    display: none !important;
  }

  /* main column padding */
  .block-container { padding: 0.6rem 1.2rem 1rem; max-width: 1500px; }

  /* title */
  .cs-title { font-family: ui-monospace, Menlo, monospace; font-size: 1.6rem;
              font-weight: 600; margin: 0; line-height: 1.1; }
  .cs-sub   { font-family: ui-monospace, Menlo, monospace; font-size: 0.78rem;
              color: #888; margin: 0.05rem 0 0.6rem; }

  /* slim metric strip */
  .cs-metrics { font-family: ui-monospace, Menlo, monospace; font-size: 0.78rem;
                color: #c9d1d9; padding: 0.15rem 0 0.5rem; opacity: 0.92; }
  .cs-metrics b { color: #fff; }
  .cs-metrics .sep { color: #4a4a4a; padding: 0 0.5rem; }

  /* sidebar logo (ascii art) */
  .cs-logo {
    font-family: ui-monospace, Menlo, monospace;
    font-size: 7.5px; line-height: 1.05;
    color: #ffffff;
    margin: -0.6rem 0 0.5rem; padding: 0;
    user-select: none;
  }

  /* sidebar: tighter section headers, no top padding, no scrollbar */
  section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 0.6rem !important;
  }
  section[data-testid="stSidebar"] > div,
  section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
  section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    overflow: hidden !important;
  }
  section[data-testid="stSidebar"] h3 {
    font-family: ui-monospace, Menlo, monospace;
    font-size: 0.78rem; color: #9ba3af;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 0.6rem 0 0.2rem;
  }
  section[data-testid="stSidebar"] h3:first-of-type { margin-top: 0; }
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 0.78rem; margin: 0 0 0.15rem !important;
  }

  /* sidebar inputs: clean, consistent border/fill across number & text */
  section[data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"],
  section[data-testid="stSidebar"] .stTextInput div[data-baseweb="input"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 6px !important; box-shadow: none !important;
  }
  section[data-testid="stSidebar"] .stNumberInput input,
  section[data-testid="stSidebar"] .stTextInput input {
    background: transparent !important; border: none !important;
    box-shadow: none !important;
  }
  .stNumberInput button { display: none !important; }

  /* tabs, slider */
  .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
  .stTabs [data-baseweb="tab"] {
    font-family: ui-monospace, Menlo, monospace; padding: 0.3rem 0;
  }
  .stSlider { padding-top: 0.25rem; padding-bottom: 0; }

  /* run button: dim while spinner alive */
  body:has([data-testid="stSpinner"]) button[kind="primary"] {
    opacity: 0.4 !important; cursor: wait !important; pointer-events: none;
  }

  /* frame caption rendered outside the image slot so it can't get clipped */
  .cs-frame-caption {
    text-align: center; font-family: ui-monospace, Menlo, monospace;
    font-size: 0.78rem; color: #888; margin: 0.35rem 0 0;
  }

  /* fixed-height image slot, image at native size, black bars fill the rest */
  [data-testid="stImage"] {
    height: 58vh;
    background: #000; border-radius: 4px; overflow: hidden;
    display: flex !important; align-items: center; justify-content: center;
  }
  [data-testid="stImage"] img {
    max-height: 100% !important; max-width: 100% !important;
    width: auto !important; height: auto !important;
    margin: 0 !important;
    image-rendering: -webkit-optimize-contrast;
  }

  /* floating banner */
  .cs-banner {
    position: fixed; right: 1.2rem; max-width: 420px;
    padding: 0.6rem 0.9rem; border-radius: 6px;
    font-family: ui-monospace, Menlo, monospace; font-size: 0.82rem;
    line-height: 1.35; z-index: 99999;
    box-shadow: 0 6px 18px rgba(0,0,0,0.55);
    animation: cs-in 0.22s ease-out;
  }
  .cs-banner-success { background: #0e2e16; color: #bcf2c8; border: 1px solid #2a7a3e; }
  .cs-banner-error   { background: #3a0e10; color: #ffc4c4; border: 1px solid #8a2c30; }
  .cs-banner-info    { background: #102536; color: #c0e0ff; border: 1px solid #2c5a8a; }
  @keyframes cs-in {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# clear persisted sidebar-closed state so reloads don't flash-collapse it
components.html(
    "<script>try { for (const k of Object.keys(window.parent.localStorage)) "
    "if (k.toLowerCase().includes('sidebar')) window.parent.localStorage.removeItem(k); } "
    "catch (_) {}</script>",
    height=0,
)


# ─── helpers ───────────────────────────────────────────────────────────────────
def _banner(level: str, msg: str, top_rem: float = 1.0, auto_dismiss_ms: int = 0) -> None:
    safe = html.escape(msg)
    dismiss = ""
    if auto_dismiss_ms > 0:
        eid = f"b{abs(hash(msg)) % 10**8}"
        dismiss = (f"<script>setTimeout(()=>{{const e=window.parent.document.getElementById('{eid}');"
                   f"if(e)e.style.display='none';}},{auto_dismiss_ms});</script>")
        safe_div = f'<div id="{eid}" class="cs-banner cs-banner-{level}" style="top:{top_rem}rem">{safe}</div>'
    else:
        safe_div = f'<div class="cs-banner cs-banner-{level}" style="top:{top_rem}rem">{safe}</div>'
    st.markdown(safe_div + dismiss, unsafe_allow_html=True)


def _pick_directory() -> str | None:
    try:
        if sys.platform == "darwin":
            script = ('tell application "System Events" to activate\n'
                      'POSIX path of (choose folder with prompt "select experiment directory")\n')
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True, timeout=600)
            return r.stdout.strip() or None
        tk_script = ("import sys, tkinter as tk; from tkinter import filedialog\n"
                     "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
                     "sys.stdout.write(filedialog.askdirectory(title='select experiment directory') or '')\n"
                     "r.destroy()\n")
        r = subprocess.run([sys.executable, "-c", tk_script],
                           capture_output=True, text=True, timeout=600)
        return r.stdout.strip() or None
    except Exception:
        return None


def _load_yaml_config(exp_dir: Path | None) -> dict:
    if not exp_dir or not (exp_dir / "config.yaml").exists():
        return {}
    try:
        with open(exp_dir / "config.yaml") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return {}


def _write_research_overrides(exp_dir: Path, research: dict) -> None:
    raw = _load_yaml_config(exp_dir)
    raw.setdefault("research", {}).update(research)
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)


class _CytoscanOnly(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "cytoscan" or record.name.startswith("cytoscan.")


class _ErrorCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []
    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _capture_run(exp_dir: Path, verbose: bool, overrides: dict) -> tuple[bool, str, list[str]]:
    buf = io.StringIO()
    py_root = logging.getLogger()
    cs_root = logging.getLogger("cytoscan")
    prev = (py_root.level, py_root.handlers[:],
            cs_root.level, cs_root.handlers[:], cs_root.propagate)

    py_root.handlers.clear(); cs_root.handlers.clear()
    cs_root.propagate = True
    stream = logging.StreamHandler(buf)
    stream.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S"))
    errors = _ErrorCollector()
    cs_only = _CytoscanOnly()
    stream.addFilter(cs_only)
    errors.addFilter(cs_only)
    py_root.addHandler(stream); py_root.addHandler(errors)
    py_root.setLevel(logging.DEBUG if verbose else logging.INFO)
    cs_root.setLevel(logging.DEBUG if verbose else logging.INFO)

    ok = True
    try:
        scaffold_experiment(exp_dir)
        _write_research_overrides(exp_dir, overrides)
        cfg = Config.load(str(exp_dir / "config.yaml"))
        run_pipeline(cfg, exp_dir)
    except SystemExit as e:
        ok = e.code in (0, None)
        if not ok and not errors.messages:
            errors.messages.append(f"experiment exited with code {e.code}. check log.")
    except Exception as e:
        ok = False
        errors.messages.append(f"{type(e).__name__}: {e}")
    finally:
        py_root.handlers.clear(); cs_root.handlers.clear()
        py_root.setLevel(prev[0])
        for h in prev[1]: py_root.addHandler(h)
        cs_root.setLevel(prev[2])
        for h in prev[3]: cs_root.addHandler(h)
        cs_root.propagate = prev[4]
    return ok, buf.getvalue(), errors.messages


def _parse_invalid_reasons(summary: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    if not summary.exists():
        return out
    in_block = False
    for line in summary.read_text().splitlines():
        if line.startswith("invalid_frames:"):
            in_block = True
            continue
        if in_block:
            s = line.strip()
            if not s.startswith("frame"):
                break
            try:
                head, reason = s.split(":", 1)
                out[int(head.replace("frame", ""))] = reason.strip()
            except ValueError:
                pass
    return out


def _frame_index_from_filename(p: Path) -> int:
    m = re.search(r"frame(\d+)", p.name)
    return int(m.group(1)) if m else -1


# ─── header ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<p class="cs-title">◎ cytoscan</p>'
    f'<p class="cs-sub">microfluidic cell perception | v{_VERSION}</p>',
    unsafe_allow_html=True,
)


# ─── sidebar ───────────────────────────────────────────────────────────────────
_LOGO = (
    "          |                             \n"
    ",---.,   .|--- ,---.,---.,---.,---.,---.\n"
    "|    |   ||    |   |`---.|    ,---||   |\n"
    "`---'`---|`---'`---'`---'`---'`---^`   '\n"
    "     `---'"
)
_LOGO_HTML = html.escape(_LOGO).replace(" ", "&nbsp;").replace("\n", "<br>")

with st.sidebar:
    st.markdown(f'<div class="cs-logo">{_LOGO_HTML}</div>', unsafe_allow_html=True)
    st.markdown("### experiment")
    if st.button("📁 browse for folder…", use_container_width=True):
        picked = _pick_directory()
        if picked:
            st.session_state["exp_dir"] = picked
            st.rerun()

    exp_dir_str = st.text_input(
        "directory",
        value=st.session_state.get("exp_dir", ""),
        placeholder="/path/to/experiment",
        label_visibility="collapsed",
    )
    st.session_state["exp_dir"] = exp_dir_str
    exp_dir = Path(exp_dir_str).expanduser() if exp_dir_str else None

    st.markdown("### research config")
    research = _load_yaml_config(exp_dir).get("research", {})
    if not isinstance(research, dict):
        research = {}
    pixel_size_um = st.number_input(
        "pixel_size_um", min_value=0.001, max_value=100.0,
        value=float(research.get("pixel_size_um", 2.119)),
        step=0.001, format="%.3f",
    )
    cell_diameter_um = st.number_input(
        "cell_diameter_um", min_value=0.1, max_value=1000.0,
        value=float(research.get("cell_diameter_um", 10.0)),
        step=0.5, format="%.2f",
    )
    channel_width_um = st.number_input(
        "channel_width_um", min_value=1.0, max_value=10000.0,
        value=float(research.get("channel_width_um", 600.0)),
        step=10.0, format="%.1f",
    )
    sector_length_um = st.number_input(
        "sector_length_um", min_value=1.0, max_value=100000.0,
        value=float(research.get("sector_length_um", 1000.0)),
        step=100.0, format="%.1f",
    )
    left_fluid = st.selectbox(
        "left_fluid", options=["dex", "peg"],
        index=0 if str(research.get("left_fluid", "dex")) == "dex" else 1,
    )

    verbose = st.toggle("verbose logging (DEBUG)", value=False)
    run_clicked = st.button("▶ start experiment", type="primary", use_container_width=True)


# ─── run handler ───────────────────────────────────────────────────────────────
if run_clicked:
    if not exp_dir or not exp_dir.is_dir():
        _banner("error", f"directory does not exist: {exp_dir_str or '(empty)'}")
    else:
        overrides = {
            "pixel_size_um":    float(pixel_size_um),
            "cell_diameter_um": float(cell_diameter_um),
            "channel_width_um": float(channel_width_um),
            "sector_length_um": float(sector_length_um),
            "left_fluid":       str(left_fluid),
        }
        with st.spinner("processing experiment..."):
            ok, log_text, errors = _capture_run(exp_dir, verbose, overrides)
        if ok:
            _banner("success", "experiment completed", auto_dismiss_ms=3500)
            st.session_state["ran_this_session"] = str(exp_dir)
        else:
            for i, msg in enumerate(errors or ["experiment failed. see log."]):
                _banner("error", msg, top_rem=1.0 + i * 3.2)
        with st.expander("log output", expanded=not ok):
            st.code(log_text or "(no log output)", language="text")


# ─── viewer guards ─────────────────────────────────────────────────────────────
if not exp_dir or not exp_dir.is_dir():
    _banner("info", "select an experiment directory in the sidebar to begin.")
    st.stop()

out_visuals = exp_dir / "output" / "visuals"
out_data    = exp_dir / "output" / "data"
visual_files = sorted(out_visuals.glob("*.png")) if out_visuals.is_dir() else []
cells_path   = out_data / "cells.csv"
frames_path  = out_data / "frames.csv"
iface_path   = out_data / "interface.csv"
summary_path = out_data / "summary.txt"

if not visual_files and not out_data.is_dir():
    _banner("info", "no outputs found yet. start an experiment to generate visuals + CSVs.")
    st.stop()

if st.session_state.get("ran_this_session") != str(exp_dir):
    _banner("info", "viewing a previous experiment's outputs. start a new experiment to refresh.")


# ─── load data ─────────────────────────────────────────────────────────────────
cells_df   = pd.read_csv(cells_path)  if cells_path.exists()  else None
frames_df  = pd.read_csv(frames_path) if frames_path.exists() else None
iface_df   = pd.read_csv(iface_path)  if iface_path.exists()  else None
invalids   = _parse_invalid_reasons(summary_path)

actual_indices = [_frame_index_from_filename(p) for p in visual_files]


# ─── metric strip ──────────────────────────────────────────────────────────────
n_visuals = len(visual_files)
n_valid   = len(frames_df) if frames_df is not None else 0
n_cells   = len(cells_df)  if cells_df  is not None else 0
mean_w    = (f"{frames_df['mean_channel_width_um'].mean():.0f} um"
             if frames_df is not None and not frames_df.empty else "n/a")
st.markdown(
    f'<div class="cs-metrics">'
    f'frames <b>{n_visuals}</b><span class="sep">·</span>'
    f'valid <b>{n_valid}</b><span class="sep">·</span>'
    f'cells <b>{n_cells}</b><span class="sep">·</span>'
    f'avg channel width <b>{mean_w}</b></div>',
    unsafe_allow_html=True,
)


# ─── frame nav state ───────────────────────────────────────────────────────────
n_frames = len(visual_files)
if "frame_pos" not in st.session_state:
    st.session_state["frame_pos"] = 0
if n_frames:
    st.session_state["frame_pos"] = min(st.session_state["frame_pos"], n_frames - 1)
else:
    st.session_state["frame_pos"] = 0

def _prev():
    st.session_state["frame_pos"] = max(0, st.session_state["frame_pos"] - 1)
def _next():
    st.session_state["frame_pos"] = min(max(0, n_frames - 1), st.session_state["frame_pos"] + 1)


# ─── view selector + arrow buttons ─────────────────────────────────────────────
view_options = ["both", "visuals", "data"] if n_frames > 0 else ["data"]
left_top, right_top = st.columns([3, 1])
with left_top:
    mode = st.radio("view", view_options,
                    horizontal=True, label_visibility="collapsed")
with right_top:
    _, prev_col, next_col = st.columns([4, 1, 1])
    show_arrows = n_frames > 0 and mode in ("both", "visuals")
    if show_arrows:
        prev_col.button("◀", key="prev_btn", on_click=_prev, use_container_width=True,
                        disabled=st.session_state["frame_pos"] <= 0)
        next_col.button("▶", key="next_btn", on_click=_next, use_container_width=True,
                        disabled=st.session_state["frame_pos"] >= n_frames - 1)

if n_frames == 0:
    _banner("info", "no valid frames were exported for this experiment.")


# ─── render: visuals ───────────────────────────────────────────────────────────
def _render_visual(container) -> None:
    if n_frames == 0:
        container.info("no visuals exported.")
        return
    if n_frames == 1:
        pos = 0
        st.session_state["frame_pos"] = 0
        container.caption("frame")
    else:
        pos = container.slider("frame", 0, n_frames - 1,
                               key="frame_pos", label_visibility="collapsed")
    actual = actual_indices[pos]
    container.image(str(visual_files[pos]), use_container_width=False)
    container.markdown(
        f'<div class="cs-frame-caption">frame {actual} · {visual_files[pos].name}</div>',
        unsafe_allow_html=True,
    )


# ─── render: per-frame cell distribution ───────────────────────────────────────
_CATS = ["peg", "int_peg", "int", "int_dex", "dex"]

def _render_distribution(container, frame_idx: int) -> None:
    container.markdown(f"**cell distribution · frame {frame_idx}**")
    slot = container.container(height=230, border=False)
    if frames_df is None or frames_df.empty:
        slot.caption("no frame data available")
        return
    sel = frames_df[frames_df["frame_index"] == frame_idx]
    if sel.empty:
        reason = invalids.get(frame_idx, "not present in frames.csv")
        slot.error(f"⚠ frame {frame_idx} invalid. reason: {reason}")
        return
    row = sel.iloc[0]
    df = pd.DataFrame({
        "category": _CATS,
        "pct":   [float(row[f"n_{c}_pct"]) for c in _CATS],
        "count": [int(row[f"n_{c}"])       for c in _CATS],
    })
    cat_colors = {
        "peg":     "#3a7bd5",
        "int_peg": "#9ec4f0",
        "int":     "#dddddd",
        "int_dex": "#f3a6a6",
        "dex":     "#d94747",
    }
    axis_order = list(reversed(_CATS))
    bars = (alt.Chart(df).mark_bar().encode(
        x=alt.X("category:N", sort=axis_order, title=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=12)),
        y=alt.Y("pct:Q", title="%", scale=alt.Scale(
            domain=[0, max(100.0, df["pct"].max() * 1.1)])),
        color=alt.Color("category:N",
                        scale=alt.Scale(domain=_CATS,
                                        range=[cat_colors[c] for c in _CATS]),
                        legend=None),
        tooltip=["category", "count", alt.Tooltip("pct:Q", format=".2f")],
    ).properties(height=180))
    labels = (alt.Chart(df).mark_text(dy=-8, color="white", fontSize=12).encode(
        x=alt.X("category:N", sort=axis_order), y=alt.Y("pct:Q"),
        text=alt.Text("label:N"),
    ).transform_calculate(
        label="datum.count + ' (' + format(datum.pct, '.1f') + '%)'"))
    slot.altair_chart(bars + labels, use_container_width=True)


# ─── render: data tabs ─────────────────────────────────────────────────────────
def _render_data(container) -> None:
    pos = st.session_state.get("frame_pos", 0)
    if actual_indices:
        actual = actual_indices[min(pos, len(actual_indices) - 1)]
    elif frames_df is not None and not frames_df.empty:
        actual = int(frames_df["frame_index"].iloc[0])
    else:
        actual = 0
    tabs = container.tabs(["frames", "cells", "interface", "summary"])
    with tabs[0]:
        _render_distribution(st, actual)
        if frames_df is not None:
            st.markdown("**raw frames.csv**")
            st.dataframe(frames_df, use_container_width=True, height=180)
        else:
            st.info("frames.csv not found")
    with tabs[1]:
        if cells_df is not None:
            view = cells_df[cells_df["frame_index"] == actual]
            st.caption(f"frame {actual} · {len(view)} cells")
            st.dataframe(view, use_container_width=True, height=440)
        else:
            st.info("cells.csv not found")
    with tabs[2]:
        if iface_df is not None:
            view = iface_df[iface_df["frame_index"] == actual]
            st.caption(f"frame {actual} · {len(view)} samples")
            st.dataframe(view, use_container_width=True, height=440)
        else:
            st.info("interface.csv not found")
    with tabs[3]:
        if summary_path.exists():
            st.code(summary_path.read_text(), language="text")
        else:
            st.info("summary.txt not found")


# ─── layout dispatch ───────────────────────────────────────────────────────────
if mode == "visuals":
    _render_visual(st)
elif mode == "data":
    _render_data(st)
else:
    col_a, col_b = st.columns([1, 1], gap="large")
    _render_visual(col_a)
    _render_data(col_b)


# ─── arrow-key hotkeys ─────────────────────────────────────────────────────────
if visual_files and mode in ("both", "visuals"):
    components.html(
        """
        <script>
          (function() {
            const doc = window.parent.document;
            if (doc.__cyKeyDown)
              doc.removeEventListener('keydown', doc.__cyKeyDown, true);

            const findBtn = (txt) => {
              for (const b of doc.querySelectorAll('button'))
                if ((b.textContent || '').trim() === txt) return b;
              return null;
            };

            const onKey = (e) => {
              if (e.target && ['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
              if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
              e.preventDefault(); e.stopImmediatePropagation();
              if (doc.activeElement && doc.activeElement !== doc.body) {
                try { doc.activeElement.blur(); } catch (_) {}
              }
              const btn = findBtn(e.key === 'ArrowLeft' ? '◀' : '▶');
              if (btn && !btn.disabled) btn.click();
            };
            doc.__cyKeyDown = onKey;
            doc.addEventListener('keydown', onKey, true);
          })();
        </script>
        """,
        height=0,
    )
