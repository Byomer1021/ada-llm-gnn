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


def test_input_norm_standardises_values_only():
    graphs = [data_of(b) for name, b, _ in SCENARIOS if "missing" not in name]
    model = build_agent("door")
    model.norm.fit(graphs)
    x, e, d = model.norm(graphs[0])
    allx = torch.cat([model.norm(g)[0] for g in graphs])
    rooms = torch.cat([g.x[:, -1] == 0 for g in graphs])
    assert abs(float(allx[rooms, 0].mean())) < 1e-4  # room width standardised over the training rooms
    assert torch.equal(x[:, 3:], graphs[0].x[:, 3:])  # missing flags and is_exterior unchanged
    assert torch.equal(e[:, 5:], graphs[0].edge_attr[:, 5:])  # door flags and wall one-hots unchanged
    assert "norm.x_mean" in model.state_dict() and "norm.d_hi" in model.state_dict()  # in the checkpoint
    # far outside the training range (real file: 7110 mm clearance) -> clamped, not extrapolated
    far = graphs[0].clone()
    far.door_attr = far.door_attr.clone()
    far.door_attr[:, 3] = 7.11  # maneuvering pull, m
    top = (model.norm.d_hi[3] - model.norm.d_mean[3]) / model.norm.d_std[3]
    assert torch.allclose(model.norm(far)[2][:, 3], top.expand(far.door_attr.shape[0]))
    # order is kept: a wider door never gets a smaller rank
    a, b = graphs[0].clone(), graphs[0].clone()
    a.door_attr, b.door_attr = a.door_attr.clone(), b.door_attr.clone()
    a.door_attr[:, 0], b.door_attr[:, 0] = 0.80, 0.83
    assert bool((model.norm(a)[2][:, 0] <= model.norm(b)[2][:, 0]).all())


def test_input_norm_turns_a_missing_value_into_zero():
    """Not the training minimum: a missing door width must not look like the narrowest door."""
    graphs = [data_of(b) for name, b, _ in SCENARIOS if "missing" not in name]
    model = build_agent("door")
    model.norm.fit(graphs)
    g = graphs[0].clone()
    g.door_attr = g.door_attr.clone()
    g.door_attr[0, 0], g.door_attr[0, 5] = 0.0, 1.0  # D1: clear width missing
    d = model.norm(g)[2]
    assert d[0, 0] == 0 and d[0, 5] == 1
    assert torch.equal(d[1], model.norm(graphs[0])[2][1])  # the other door is unchanged


def test_drop_values_masks_a_door_the_same_everywhere():
    from torch_geometric.loader import DataLoader
    from src.agents.train import drop_values
    graphs = [data_of(b) for name, b, _ in SCENARIOS if "missing" not in name]
    batch = next(iter(DataLoader(graphs, batch_size=len(graphs))))
    out = drop_values(batch, 0.5, torch.Generator().manual_seed(0))
    m = out.door_attr[:, 5:10]
    assert 0 < m.sum() < m.numel()
    assert torch.equal(out.edge_attr[:, 5:10], m.repeat_interleave(2, 0))  # both directions of a door
    assert torch.equal(out.door_attr[:, :5] * m, torch.zeros_like(m))    # a dropped value is 0
    keep = m == 0
    assert torch.equal(out.door_attr[:, :5][keep], batch.door_attr[:, :5][keep])
    assert torch.equal(out.x[:, :2], batch.x[:, :2])                     # room width / length never
    ext = batch.x[:, -1] == 1
    assert torch.equal(out.x[ext], batch.x[ext])                         # EXTERIOR untouched
    assert torch.equal(out.door_y, batch.door_y)                         # targets of the full building
    assert float(batch.door_attr[:, 5:10].sum()) == 0                    # the input batch is not changed


def test_run_agents_missing_value_not_checked_or_inferred():
    from tests.test_rules import two_rooms
    b = two_rooms()
    next(d for d in b["doors"] if d["id"] == "D2")["clear_width_mm"] = None
    agents = {a: build_agent(a).eval() for a in ("door", "room", "route")}
    nc = {d["id"]: d for d in run_agents(b, agents)["doors"]}["D2"]
    inf = {d["id"]: d for d in run_agents(b, agents, missing="infer")["doors"]}["D2"]
    assert nc["fail_prob"]["ADA 404.2.3"] is None and nc["not_checked"] == ["ADA 404.2.3"] and "inferred" not in nc
    assert inf["fail_prob"]["ADA 404.2.3"] is not None and inf["not_checked"] == []
    assert inf["inferred"] == ["ADA 404.2.3"] and inf["fail_prob"]["ADA 404.2.5"] == nc["fail_prob"]["ADA 404.2.5"]


def test_quantile_norm_ranks_clamps_and_keeps_order():
    graphs = [data_of(b) for name, b, _ in SCENARIOS if "missing" not in name]
    model = build_agent("door", norm="quantile")
    model.norm.fit(graphs)
    x, e, d = model.norm(graphs[0])
    assert d[:, :5].min() >= -1 and d[:, :5].max() <= 1
    assert torch.equal(e[:, 5:], graphs[0].edge_attr[:, 5:]) and torch.equal(x[:, 3:], graphs[0].x[:, 3:])
    far = graphs[0].clone()
    far.door_attr = far.door_attr.clone()
    far.door_attr[:, 3] = 7.11  # beyond every training value -> the top rank, not an extrapolation
    assert torch.allclose(model.norm(far)[2][:, 3], torch.ones(far.door_attr.shape[0]))
    a, b = graphs[0].clone(), graphs[0].clone()
    a.door_attr, b.door_attr = a.door_attr.clone(), b.door_attr.clone()
    a.door_attr[:, 0], b.door_attr[:, 0] = 0.80, 0.83
    assert bool((model.norm(a)[2][:, 0] <= model.norm(b)[2][:, 0]).all())
    assert "norm.d_q" in model.state_dict()
    rebuilt = build_agent("door", norm="quantile")  # what load_agents does with config["norm"]
    rebuilt.load_state_dict(model.state_dict())


def test_rescale_puts_the_threshold_at_one_half():
    from src.agents.gnn_agents import rescale
    t = 0.8
    assert rescale(t, t) == 0.5 and rescale(0.0, t) == 0.0 and rescale(1.0, t) == 1.0
    ps = [0.0, 0.3, 0.79, 0.8, 0.81, 0.95, 1.0]
    out = [rescale(p, t) for p in ps]
    assert out == sorted(out) and rescale(0.79, t) < 0.5 <= rescale(0.81, t)
    assert rescale(0.37, 0.5) == 0.37


def test_calibrate_chooses_on_the_given_split(monkeypatch):
    import numpy as np
    import src.agents.train as T
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.6, 0.62, 0.1, 0.2, 0.9, 0.95])  # at 0.5: two false alarms
    monkeypatch.setattr(T, "predict", lambda model, agent, graphs: (y, p))
    assert T.calibrate(None, "room", []) == 0.63  # perfect F1 from 0.63 to 0.90; closest to 0.5 wins
    # route predicts "reachable"; the failing class is "not reachable" (0). At 0.5 the reachable rooms
    # at 0.40 and 0.38 would be called unreachable; perfect from 0.11 to 0.38, closest to 0.5 wins
    monkeypatch.setattr(T, "predict", lambda model, agent, graphs: (1 - y, 1 - p))
    assert T.calibrate(None, "route", []) == 0.38
    yd, pd = np.stack([y] * 4, 1), np.stack([p] * 4, 1)
    monkeypatch.setattr(T, "predict", lambda model, agent, graphs: (yd, pd))
    assert T.calibrate(None, "door", []) == [0.63] * 4


def test_run_agents_applies_saved_thresholds():
    agents = {a: build_agent(a).eval() for a in ("door", "room", "route")}
    base = run_agents(SCENARIOS[1][1], agents)
    agents["door"].thresholds, agents["room"].thresholds = [0.9, 0.5, 0.5, 0.5], 0.2
    tuned = run_agents(SCENARIOS[1][1], agents)
    from src.agents.gnn_agents import rescale
    d0, d1 = base["doors"][0]["fail_prob"], tuned["doors"][0]["fail_prob"]
    assert abs(d1["ADA 404.2.3"] - rescale(d0["ADA 404.2.3"], 0.9)) < 2e-3  # both are rounded to 3 places
    assert d1["ADA 404.2.5"] == d0["ADA 404.2.5"]


def test_transit_graph_of_two_rooms():
    """EXTERIOR -D1- R1 -D2- R2, D1 on the south wall of R1, D2 on its east wall (adjacent)."""
    from src.agents.models import DOOR_DIM, transit
    from src.ada.graph_builder import SIDES
    from tests.test_rules import two_rooms
    d = data_of(two_rooms())
    nodes, tei, feats, into = transit(d.x, d.edge_index, d.edge_attr)
    names = d.node_names
    assert sorted(names[i] for i in into.tolist()) == ["R1", "R1", "R2"]  # enter R1 by D1, R1 by D2, R2 by D2
    assert int(nodes[:, DOOR_DIM].sum()) == 1                             # one entrance
    assert tei.shape == (2, 1)                                            # one crossing: R1 from D1 to D2
    w = len(SIDES)
    assert SIDES[int(feats[0, :w].argmax())] == "S" and SIDES[int(feats[0, w:2 * w].argmax())] == "E"
    assert names[int(into[tei[1, 0]])] == "R2"


def test_transit_route_agent_shape_and_training():
    from torch_geometric.loader import DataLoader
    torch.manual_seed(0)
    graphs = [data_of(b) for _, b, _ in SCENARIOS]
    model = build_agent("route", "transit", 32, 3)
    model.norm.fit(graphs)
    d = graphs[0]
    assert model(d).shape == d.reach_y.shape
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss()
    first = None
    for _ in range(80):
        for batch in DataLoader(graphs, batch_size=len(graphs)):
            y, m = targets("route", batch)
            loss = lossf(model(batch)[m], y[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = loss.item() if first is None else first
    assert loss.item() < first
    with pytest.raises(ValueError):
        build_agent("door", "transit")


def test_run_agents_output_format():
    agents = {a: build_agent(a).eval() for a in ("door", "room", "route")}
    f = run_agents(SCENARIOS[1][1], agents)
    assert f["source"] == "gnn" and {d["id"] for d in f["doors"]} == {"D1", "D2"}
    assert set(f["doors"][0]["fail_prob"]) == {"ADA 404.2.3", "ADA 404.2.5", "ADA 404.2.7", "ADA 404.2.4"}
    assert {r["id"] for r in f["rooms"]} == {"R1", "R2"}
