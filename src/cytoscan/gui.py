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

  /* keep streamlit's tooltip help icons visible next to widget labels */
  section[data-testid="stSidebar"] [data-testid="stTooltipIcon"],
  section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"],
  section[data-testid="stSidebar"] [data-testid="InfoIcon"] {
    display: inline-flex !important;
    opacity: 0.7 !important;
    visibility: visible !important;
    margin-left: 0.3rem;
  }

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
  /* sidebar scrolls only when content overflows; thin native scrollbar */
  section[data-testid="stSidebar"] > div,
  section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
  section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    scrollbar-width: thin;
  }
  section[data-testid="stSidebar"] ::-webkit-scrollbar { width: 6px; }
  section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.12); border-radius: 3px;
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
  /* hide only the +/- step buttons (by their aria-label / testid) so the
     tooltip help button next to the widget label stays visible. */
  .stNumberInput button[aria-label*="ncrement"],
  .stNumberInput button[aria-label*="ecrement"],
  .stNumberInput button[data-testid*="Step"] {
    display: none !important;
  }

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
  /* sidebar spinner renders inline below the run button. small, mono, blue */
  section[data-testid="stSidebar"] [data-testid="stSpinner"] {
    font-family: ui-monospace, Menlo, monospace;
    font-size: 0.78rem; color: #c9d1d9;
    margin: 0.45rem 0 0; padding: 0;
    background: transparent; border: none;
  }
  section[data-testid="stSidebar"] [data-testid="stSpinner"] > div {
    gap: 0.4rem; align-items: center;
  }
  section[data-testid="stSidebar"] [data-testid="stSpinner"] svg,
  section[data-testid="stSidebar"] [data-testid="stSpinner"] i {
    width: 12px !important; height: 12px !important;
    border-color: #4a9eff transparent #4a9eff transparent !important;
  }
  @keyframes cs-in {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes cs-dots {
    0%   { content: ''; }
    25%  { content: '.'; }
    50%  { content: '..'; }
    75%, 100% { content: '...'; }
  }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# clear persisted sidebar-closed state so reloads don't flash-collapse it,
# and stop number inputs from swallowing wheel events (so users can scroll
# the sidebar while their cursor is over an input).
components.html(
    """
    <script>
      const doc = window.parent.document;
      try {
        for (const k of Object.keys(window.parent.localStorage))
          if (k.toLowerCase().includes('sidebar'))
            window.parent.localStorage.removeItem(k);
      } catch (_) {}

      if (!doc.__cyWheelGuard) {
        doc.__cyWheelGuard = true;
        // when the wheel fires over a sidebar input, scroll the sidebar
        // explicitly instead of letting the input swallow the event.
        doc.addEventListener('wheel', (e) => {
          const t = e.target;
          if (!t || !t.closest) return;
          const sidebar = t.closest('section[data-testid="stSidebar"]');
          if (!sidebar) return;
          const onWidget = t.closest('input, textarea, select, '
            + '[data-baseweb="input"], [data-baseweb="select"], '
            + '[data-baseweb="textarea"]');
          if (!onWidget) return;
          // find the scrollable container inside the sidebar
          const scroller = sidebar.querySelector(
            '[data-testid="stSidebarUserContent"], [data-testid="stSidebarContent"]'
          ) || sidebar;
          scroller.scrollTop += e.deltaY;
          e.preventDefault();
          if (doc.activeElement && doc.activeElement.blur) {
            try { doc.activeElement.blur(); } catch (_) {}
          }
        }, { passive: false, capture: true });
      }
    </script>
    """,
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


def _write_overrides(exp_dir: Path, overrides: dict[str, dict]) -> None:
    """Merge UI overrides (section -> {key: value}) into config.yaml, preserving
    every other section."""
    raw = _load_yaml_config(exp_dir)
    for section, kvs in overrides.items():
        raw.setdefault(section, {}).update(kvs)
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
        _write_overrides(exp_dir, overrides)
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
    _cfg_raw = _load_yaml_config(exp_dir)
    research   = _cfg_raw.get("research", {})       if isinstance(_cfg_raw.get("research"),       dict) else {}
    cell_det   = _cfg_raw.get("cell_detection", {}) if isinstance(_cfg_raw.get("cell_detection"), dict) else {}
    visuals    = _cfg_raw.get("export_visuals", {}) if isinstance(_cfg_raw.get("export_visuals"), dict) else {}

    pixel_size_um = st.number_input(
        "pixel_size_um", min_value=0.001, max_value=100.0,
        value=float(research.get("pixel_size_um", 2.119)),
        step=0.001, format="%.3f",
        help="physical size of one pixel in micrometers",
    )
    cell_diameter_um = st.number_input(
        "cell_diameter_um", min_value=0.1, max_value=1000.0,
        value=float(research.get("cell_diameter_um", 10.0)),
        step=0.5, format="%.2f",
        help="typical cell diameter in micrometers",
    )
    channel_width_um = st.number_input(
        "channel_width_um", min_value=1.0, max_value=10000.0,
        value=float(research.get("channel_width_um", 600.0)),
        step=10.0, format="%.1f",
        help="physical width of the microfluidic channel in micrometers",
    )
    sector_length_um = st.number_input(
        "sector_length_um", min_value=1.0, max_value=100000.0,
        value=float(research.get("sector_length_um", 1000.0)),
        step=100.0, format="%.1f",
        help="vertical extent kept around the origin marker (±sector/2 per frame)",
    )
    left_fluid = st.selectbox(
        "left_fluid", options=["dex", "peg"],
        index=0 if str(research.get("left_fluid", "dex")) == "dex" else 1,
        help="which fluid sits on the left half of the channel",
    )
    cell_threshold = st.number_input(
        "cell_detection_threshold", min_value=0, max_value=255,
        value=int(cell_det.get("threshold", 100)), step=1,
        help="brightness cutoff (0 to 255) for marking pixels as cells in the fluorescent frame",
    )
    _frames = ["brightfield", "fluorescent", "mixed"]
    exported_frame = st.selectbox(
        "exported_frame", options=_frames,
        index=_frames.index(str(visuals.get("exported_frame", "brightfield")))
              if str(visuals.get("exported_frame", "brightfield")) in _frames else 0,
        help="which raw frame to draw detections onto in the output visuals",
    )

    verbose = st.toggle("verbose logging (DEBUG)", value=False,
                        help="record detailed per frame diagnostics in the log output")
    run_clicked = st.button("▶ start experiment", type="primary", use_container_width=True)
    spinner_slot = st.empty()   # spinner during a run lives here


# ─── run handler ───────────────────────────────────────────────────────────────
if run_clicked:
    if not exp_dir or not exp_dir.is_dir():
        _banner("error", f"directory does not exist: {exp_dir_str or '(empty)'}")
    else:
        overrides = {
            "research": {
                "pixel_size_um":    float(pixel_size_um),
                "cell_diameter_um": float(cell_diameter_um),
                "channel_width_um": float(channel_width_um),
                "sector_length_um": float(sector_length_um),
                "left_fluid":       str(left_fluid),
            },
            "cell_detection": {"threshold": int(cell_threshold)},
            "export_visuals": {"exported_frame": str(exported_frame)},
        }
        with spinner_slot, st.spinner("processing experiment"):
            ok, log_text, errors = _capture_run(exp_dir, verbose, overrides)
        if ok:
            st.session_state["ran_this_session"] = str(exp_dir)
        # persist run state across reruns so frame switches don't wipe it
        st.session_state["last_log_text"]   = log_text
        st.session_state["last_log_ok"]     = ok
        st.session_state["last_log_dir"]    = str(exp_dir)
        st.session_state["last_log_errors"] = errors or []

# render status banner + log expander whenever there's a stored run for this dir
if (exp_dir and st.session_state.get("last_log_dir") == str(exp_dir)
        and st.session_state.get("last_log_text") is not None):
    if st.session_state.get("last_log_ok", True):
        _banner("success", "experiment completed")
    else:
        for i, msg in enumerate(st.session_state.get("last_log_errors")
                                or ["experiment failed. see log."]):
            _banner("error", msg, top_rem=1.0 + i * 3.2)
    with st.expander("log output", expanded=not st.session_state.get("last_log_ok", True)):
        st.code(st.session_state["last_log_text"] or "(no log output)", language="text")


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
    show_arrows = n_frames > 0
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
    if n_frames > 1:
        st.slider("frame", 0, n_frames - 1, key="frame_pos", label_visibility="collapsed")
    elif n_frames == 1:
        st.session_state["frame_pos"] = 0
    _render_data(st)
else:
    col_a, col_b = st.columns([1, 1], gap="large")
    _render_visual(col_a)
    _render_data(col_b)


# ─── arrow-key hotkeys ─────────────────────────────────────────────────────────
if n_frames > 0:
    components.html(
        """
        <script>
          (function() {
            const doc = window.parent.document;
            const INITIAL_DELAY = 300;   // ms before auto-repeat starts
            const REPEAT_DELAY  = 90;    // ms between subsequent clicks while held

            // tear down any previous listeners + timers so reruns don't stack
            if (doc.__cyKeyDown) doc.removeEventListener('keydown', doc.__cyKeyDown, true);
            if (doc.__cyKeyUp)   doc.removeEventListener('keyup',   doc.__cyKeyUp,   true);
            if (doc.__cyRepeat)  { clearTimeout(doc.__cyRepeat); doc.__cyRepeat = null; }

            const findBtn = (txt) => {
              for (const b of doc.querySelectorAll('button'))
                if ((b.textContent || '').trim() === txt) return b;
              return null;
            };

            const clickArrow = (key) => {
              const btn = findBtn(key === 'ArrowLeft' ? '◀' : '▶');
              if (btn && !btn.disabled) { btn.click(); return true; }
              return false;
            };

            const stopRepeat = () => {
              if (doc.__cyRepeat) { clearTimeout(doc.__cyRepeat); doc.__cyRepeat = null; }
              doc.__cyHeldKey = null;
            };

            const onKeyDown = (e) => {
              if (e.target && ['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
              if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
              e.preventDefault(); e.stopImmediatePropagation();
              if (e.repeat) return;   // ignore OS auto-repeat, our timer drives it
              if (doc.activeElement && doc.activeElement !== doc.body) {
                try { doc.activeElement.blur(); } catch (_) {}
              }
              // first click is immediate, then schedule the auto-repeat loop
              clickArrow(e.key);
              stopRepeat();
              doc.__cyHeldKey = e.key;
              const tick = () => {
                if (doc.__cyHeldKey !== e.key) return;
                if (!clickArrow(e.key)) { stopRepeat(); return; }
                doc.__cyRepeat = setTimeout(tick, REPEAT_DELAY);
              };
              doc.__cyRepeat = setTimeout(tick, INITIAL_DELAY);
            };

            const onKeyUp = (e) => {
              if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
              e.preventDefault(); e.stopImmediatePropagation();
              stopRepeat();
            };

            doc.__cyKeyDown = onKeyDown;
            doc.__cyKeyUp   = onKeyUp;
            doc.addEventListener('keydown', onKeyDown, true);
            doc.addEventListener('keyup',   onKeyUp,   true);
            // safety: if the page loses focus mid-hold, stop the loop
            window.parent.addEventListener('blur', stopRepeat, { once: false });
          })();
        </script>
        """,
        height=0,
    )
