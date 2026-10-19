# ADA accessibility checking of IFC buildings with LLMs and GNN agents

CSE 495 graduation project, Gebze Technical University.

A large LLM (teacher) designs labelled single-storey IFC buildings. The rule engine and
an architecture council verify them, and the knowledge is distilled into a small LLM
(student). Each IFC file becomes a graph (room = node, door = edge). Three GNN agents
(door, room, route) check their own ADA rules in parallel, and an LLM main agent
(Qwen 7B, or GPT-4.1-mini if that is not enough) combines their findings into the
verdict: **accessible / not accessible + the failing ADA rule**. The verdict is shown
in a report and a web UI.

## Setup

Python 3.10 or newer.

```bash
cd ada-llm-gnn
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install torch            # or the command for your machine from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Check that everything works

```bash
python -m pytest                    # 117 tests (GNN tests are skipped if torch is missing)
python scripts/demo_pipeline.py     # one building: JSON -> IFC -> read -> graph -> agents -> main agent -> report
```
The demo writes `data/demo/demo.ifc`, `demo_report.md` and `demo_3d.png`. It runs without
an API key or trained model: it uses the *oracle* agents (the rule engine in place of the
GNNs, for testing only) and the *rule* main agent (no LLM).

## 1. IFC generation (teacher -> verify -> student)

```bash
# no API at all: fake answers, to test the whole chain. They are written to separate files
# (data/llm/teacher_dryrun_*.jsonl, data/synthetic/teacher_dryrun/) and never mix with the real data;
# the next steps read them with --dry-run (council, --export-sft) or --sources teacher_dryrun_approved (dataset)
python -m src.llm.generate --dry-run --n 50 --write-ifc

# teacher with Nemotron (key from build.nvidia.com)
#   Windows PowerShell:  $env:NVIDIA_API_KEY="nvapi-..."
#   Mac/Linux:           export NVIDIA_API_KEY=nvapi-...
python -m src.llm.generate --role teacher --provider nemotron --n 20 --write-ifc

# teacher with GPT-4.1 (OPENAI_API_KEY); note: no reasoning trace is returned
python -m src.llm.generate --role teacher --provider openai --model gpt-4.1 --n 20 --write-ifc

# architecture council: pictures + review sheet, then collect the filled sheets
python scripts/council.py packet
python scripts/council.py collect data/council/filled/*.csv

# council-approved teacher answers -> student training file
python -m src.llm.generate --export-sft

# student: LoRA fine-tuning of Qwen 7B on a GPU (Colab / lab PC); settings in config/student.yaml
pip install -r requirements-student.txt
python -m src.llm.train_student                      # -> models/student/<time>/adapter, experiments/<time>_student/
# serve the base model + the adapter with vLLM (no merge needed), then generate with the student
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules qwen-ada=models/student/<time>/adapter --port 8000
python -m src.llm.generate --role student --provider local --model qwen-ada --n 2000 --workers 8 --write-ifc
```
Results: `data/llm/*.jsonl` (every attempt, verified ones with labels, approved ones),
IFC files in `data/synthetic/<role>/`.

## 2. GNN agents

```bash
python -m src.agents.dataset                         # graphs + 70/15/15 split by IFC file
python -m src.agents.train --agent door  --seeds 0 1 2
python -m src.agents.train --agent room  --seeds 0 1 2
python -m src.agents.train --agent route --conv mpnn_max --seeds 0 1 2
```
Models go to `models/`, metrics to `experiments/`.

## 3. Main agent (LLM)

Option 1, local Qwen 7B with [Ollama](https://ollama.com): `ollama pull qwen2.5:7b` (it serves
`http://localhost:11434/v1`). Option 2, the fallback: set `OPENAI_API_KEY` for GPT-4.1-mini.
Choose in `config/pipeline.yaml` (`provider: auto | local | openai | rule`).

## 4. Check, evaluate, test, UI

```bash
python -m src.pipeline some.ifc                      # one file -> report
python scripts/evaluate.py --provider rule           # test split; also: local / openai / auto, --agents oracle
python scripts/test_real_ifc.py                      # the real IFC files in data/real/ (final test)
streamlit run app/streamlit_app.py                   # web UI: upload an IFC
```

## Useful tools

```bash
python tools/render_3d.py some.ifc picture.png       # 3D picture (council survey, slides)
```

## Folder layout

```
ada-llm-gnn/
├── config/
│   ├── ada_rules.yaml       # every ADA threshold + the building verdict setting
│   └── pipeline.yaml        # agent models, main agent (Qwen 7B / GPT-4.1-mini)
├── prompts/                 # generation (system + user) and main agent prompts
├── src/
│   ├── ada/
│   │   ├── rules.py         # rule engine: verification + labels (make_labels)
│   │   ├── ifc_writer.py    # building JSON -> IFC
│   │   ├── ifc_reader.py    # IFC -> building JSON (property sets + placement)
│   │   ├── graph_builder.py # building -> NetworkX / PyG
│   │   └── report.py        # verdict + findings -> report (Markdown)
│   ├── llm/generate.py      # teacher / student generation + verification + SFT export
│   ├── agents/
│   │   ├── dataset.py       # records + IFC files -> graph dataset, split
│   │   ├── models.py        # door / room / route agents
│   │   ├── train.py         # agent training + metrics
│   │   ├── gnn_agents.py    # run the trained agents -> findings (+ oracle)
│   │   └── main_agent.py    # LLM main agent (+ fallback, + rule baseline)
│   └── pipeline.py          # one IFC: read -> graph -> agents -> main agent -> report
├── scripts/
│   ├── demo_pipeline.py     # end-to-end demo on one building (no API)
│   ├── council.py           # council packet / collect the votes
│   ├── evaluate.py          # evaluation on the synthetic test split
│   └── test_real_ifc.py     # real IFC test
├── app/streamlit_app.py     # web UI
├── tools/render_3d.py       # IFC -> 3D picture
├── tests/                   # rule engine, IFC round trip, leakage, generation, main agent, GNN
├── data/{llm,synthetic,council,datasets,real}/   # not in git
├── models/                  # trained agents (not in git)
├── experiments/             # experiment results
├── docs/figures/            # figures
└── prototype/               # first prototype (old 3-class logic, reference only)
```

## License

MIT, see [LICENSE](LICENSE).
