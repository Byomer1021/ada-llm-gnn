"""
Web UI: upload an IFC file -> accessible or not + the failing ADA rule (+ report, 3D picture).

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.ada.ifc_reader import read_ifc  # noqa: E402
from src.ada.report import _limit, _value  # noqa: E402
from src.agents.main_agent import load_config  # noqa: E402
from src.pipeline import check_building  # noqa: E402

st.set_page_config(page_title="ADA accessibility check", layout="wide")
cfg = load_config()
models_ready = all((ROOT / cfg["agents"]["models_dir"] / f"{a}.pt").exists() for a in ("door", "room", "route"))

st.title("ADA accessibility check")
st.caption("Upload a single-storey IFC file. Three GNN agents check doors, rooms and the route from the "
           "entrance; the main agent gives the verdict (2010 ADA Standards, wheelchair user).")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Main agent", ["auto", "local", "openai", "rule"],
                            index=["auto", "local", "openai", "rule"].index(cfg["main_agent"]["provider"]),
                            help="auto = local Qwen 7B, then GPT-4.1-mini. rule = no LLM (baseline).")
    agents = st.radio("Agents", ["gnn", "oracle"], index=0 if models_ready else 1,
                      help="oracle = rule engine in place of the GNNs; ONLY for testing before training.")
    if not models_ready:
        st.warning("Trained GNN agents not found in models/. Using the oracle (testing only).")

up = st.file_uploader("IFC file", type=["ifc"])
if up:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / up.name
        path.write_bytes(up.getvalue())
        try:
            with st.spinner("Reading the IFC file and running the agents..."):
                b = read_ifc(str(path))
                b["building_id"] = path.stem
                res = check_building(b, agents=agents, provider=provider, cfg=cfg)
        except Exception as e:
            st.error(f"The check failed: {e}")
            st.stop()
        d, fnd = res["decision"], res["findings"]
        left, right = st.columns([3, 2])
        with left:
            if d["label"] == "accessible":
                st.success("ACCESSIBLE")
            else:
                st.error("NOT ACCESSIBLE")
            if d.get("explanation"):
                st.write(d["explanation"])
            if d["failing_rules"]:
                st.subheader("Failing ADA rules")
                st.table([{"Element": f["element"], "ADA rule": f["rule"], "Requirement": _limit(f["rule"]),
                           "Value (mm)": _value(fnd, f["element"], f["rule"])} for f in d["failing_rules"]])
            if d.get("warnings"):
                st.subheader("Warnings")
                for w in d["warnings"]:
                    st.write("- " + w)
            if fnd.get("data_issues"):
                with st.expander(f"Data issues in the file ({len(fnd['data_issues'])})"):
                    for i in fnd["data_issues"]:
                        st.write("- " + i)
            st.caption(f"Agents: {fnd['source']} · main agent: {d.get('provider')} · {res['seconds']} s")
            st.download_button("Download report (.md)", res["report"], file_name=f"{path.stem}_report.md")
        with right:
            try:
                from render_3d import render
                png = Path(tmp) / "view.png"
                render(str(path), str(png), failing={f["element"] for f in d["failing_rules"]})
                st.image(png.read_bytes(), caption="Failing doors in red")
            except Exception as e:  # geometry the renderer cannot handle
                st.info(f"No 3D picture for this file ({e.__class__.__name__}).")
        with st.expander("What each agent found"):
            st.markdown(res["report"].split("## What the agents found", 1)[-1])
