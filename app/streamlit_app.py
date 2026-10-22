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
st.caption("Upload an IFC file. Three GNN agents check doors, rooms and the route from the entrance; the main "
           "agent gives the verdict (2010 ADA Standards, wheelchair user). The scope is one storey: in a "
           "multi-storey file the entrance storey is checked.")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Main agent", ["auto", "local", "openai", "rule"],
                            index=["auto", "local", "openai", "rule"].index(cfg["main_agent"]["provider"]),
                            on_change=lambda: st.session_state.pop("llm_down", None),
                            help="auto = local Qwen 7B, then GPT-4.1-mini. rule = no LLM: traces the path over "
                                 "the agents' findings. If no LLM can be reached, the rule main agent decides "
                                 "(and the result says so).")
    agents = st.radio("Agents", ["gnn", "oracle"], index=0 if models_ready else 1,
                      help="oracle = rule engine in place of the GNNs; ONLY for testing before training.")
    if not models_ready:
        st.warning(f"Trained GNN agents not found in {cfg['agents']['models_dir']}/. Using the oracle (testing only).")

up = st.file_uploader("IFC file", type=["ifc"])
if up:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / up.name
        path.write_bytes(up.getvalue())
        try:
            with st.spinner("Reading the IFC file and running the agents..."):
                b = read_ifc(str(path))
                b["building_id"] = path.stem
                # an LLM that could not be reached earlier in this session is not tried again (each try
                # waits ~10-15 s); changing the main agent setting, or reloading the page, retries it
                down = st.session_state.get("llm_down")
                use = "rule" if down and down[0] == provider else provider
                res = check_building(b, agents=agents, provider=use, cfg=cfg, rule_fallback=True)
                if res["decision"].get("fallback_reason") and res["decision"]["provider"].startswith("rule"):
                    st.session_state["llm_down"] = (provider, res["decision"]["fallback_reason"])
                elif use != provider:
                    res["decision"]["fallback_reason"] = f"Not tried again in this session: {down[1]}"
        except Exception as e:
            st.error(f"The check failed: {e}")
            st.stop()
        d, fnd = res["decision"], res["findings"]
        storey = b.get("storey", {})
        left, right = st.columns([3, 2])
        with left:
            if d.get("fallback_reason"):
                st.warning(f"Main agent: {d['provider']} (fallback). {d['fallback_reason']}")
            if d["label"] == "accessible":
                st.success("ACCESSIBLE")
            else:
                st.error("NOT ACCESSIBLE")
            if storey.get("storeys_in_file", 1) > 1:
                st.caption(f"Checked storey: {storey['name']} ({storey['how']}; {storey['storeys_in_file']} "
                           f"storeys in the file)")
            if d.get("explanation"):
                st.write(d["explanation"])
            if d.get("not_checked"):
                st.info(f"The result holds for the rules that could be checked: {len(d['not_checked'])} rule checks "
                        f"had no value in the file (see 'Not checked').")
            if d["failing_rules"]:
                st.subheader("Failing ADA rules")
                st.table([{"Element": f["element"], "ADA rule": f["rule"], "Requirement": _limit(f["rule"]),
                           "Value (mm)": _value(fnd, f["element"], f["rule"])} for f in d["failing_rules"]])
            if d.get("warnings"):
                st.subheader("Warnings")
                for w in d["warnings"]:
                    st.write("- " + w)
            if d.get("not_checked"):
                with st.expander(f"Not checked: value not in the file ({len(d['not_checked'])})"):
                    st.table([{"Element": x["element"], "ADA rule": x["rule"], "Requirement": _limit(x["rule"])}
                              for x in d["not_checked"]])
            est = [{"Element": e["id"], "Value": k.replace("_mm", ""), "mm": round(e["values"][k]) if e["values"].get(k)
                    is not None else None, "How": how}
                   for e in fnd["doors"] + fnd["rooms"] for k, how in e.get("estimated", {}).items()]
            if est:
                with st.expander(f"Estimated values: not in the file, estimated from its geometry ({len(est)})"):
                    st.table(est)
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
                render(str(path), str(png), failing={f["element"] for f in d["failing_rules"]}, building=b)
                st.image(png.read_bytes(), caption="Failing doors in red")
            except Exception as e:  # geometry the renderer cannot handle
                st.info(f"No 3D picture for this file ({e.__class__.__name__}).")
        with st.expander("What each agent found"):
            st.markdown(res["report"].split("## What the agents found", 1)[-1])
