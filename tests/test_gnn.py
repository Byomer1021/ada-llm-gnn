"""GNN agents: graph -> PyG tensors, forward pass shapes, a tiny training run. Skipped without torch."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.ada.graph_builder import build_graph, to_pyg  # noqa: E402
from src.ada.rules import make_labels  # noqa: E402
from src.agents.gnn_agents import run_agents  # noqa: E402
from src.agents.models import CONVS, build_agent, targets  # noqa: E402
from tests.test_rules import SCENARIOS  # noqa: E402


def data_of(b):
    return to_pyg(build_graph(b, make_labels(b)))


def test_tensor_shapes():
    d = data_of(SCENARIOS[1][1])
    assert d.x.shape == (3, 7) and d.edge_index.shape == (2, 4) and d.edge_attr.shape == (4, 20)
    assert d.door_y.tolist()[1] == [1, 0, 0, 0] and d.reach_y.tolist() == [-1, 1, 0]
    assert d.building_y.item() == 0


@pytest.mark.parametrize("conv", CONVS)
@pytest.mark.parametrize("agent", ["door", "room", "route"])
def test_forward(agent, conv):
    d = data_of(SCENARIOS[0][1])
    out = build_agent(agent, conv)(d)
    y, _ = targets(agent, d)
    assert out.shape == y.shape


def test_tiny_training_lowers_loss():
    from torch_geometric.loader import DataLoader
    torch.manual_seed(0)
    graphs = [data_of(b) for _, b, _ in SCENARIOS]
    model = build_agent("route", "gine", 32, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss()
    first = None
    for _ in range(60):
        for batch in DataLoader(graphs, batch_size=len(graphs)):
            y, m = targets("route", batch)
            loss = lossf(model(batch)[m], y[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = loss.item() if first is None else first
    assert loss.item() < first


def test_run_agents_output_format():
    agents = {a: build_agent(a).eval() for a in ("door", "room", "route")}
    f = run_agents(SCENARIOS[1][1], agents)
    assert f["source"] == "gnn" and {d["id"] for d in f["doors"]} == {"D1", "D2"}
    assert set(f["doors"][0]["fail_prob"]) == {"ADA 404.2.3", "ADA 404.2.5", "ADA 404.2.7", "ADA 404.2.4"}
    assert {r["id"] for r in f["rooms"]} == {"R1", "R2"}
