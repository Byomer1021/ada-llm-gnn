"""The agents' input must never contain answer fields (label leakage)."""
from src.ada.graph_builder import EDGE_FEATURES, FORBIDDEN, NODE_FEATURES, build_graph
from src.ada.rules import make_labels
from tests.test_rules import SCENARIOS


def test_feature_lists_have_no_answer_fields():
    assert not (set(NODE_FEATURES) | set(EDGE_FEATURES)) & FORBIDDEN


def test_graph_features_ignore_extra_answer_fields():
    b = SCENARIOS[1][1]
    b = {**b, "intended_verdict": {"label": "not_accessible", "failing_rules": []}}
    b["doors"] = [{**d, "HandicapAccessible": True} for d in b["doors"]]
    g = build_graph(b)
    for _, data in g.nodes(data=True):
        assert len(data["x"]) == len(NODE_FEATURES)
    for _, _, data in g.edges(data=True):
        assert len(data["x"]) == len(EDGE_FEATURES)


def test_graph_without_labels_has_no_targets():
    g = build_graph(SCENARIOS[1][1])  # check time: no labels, no rule engine
    assert g.graph["accessible"] is None
    assert all(d["reach_y"] == -1 and d["turning_y"] == -1 for _, d in g.nodes(data=True))
    assert all(d["door_y"] is None for _, _, d in g.edges(data=True))


def test_graph_targets_from_stored_labels():
    b = SCENARIOS[1][1]  # D2 narrow
    g = build_graph(b, make_labels(b))
    assert g.graph["accessible"] == 0
    assert g.nodes["R1"]["reach_y"] == 1 and g.nodes["R2"]["reach_y"] == 0 and g.nodes["EXTERIOR"]["reach_y"] == -1
    assert g.edges["R1", "R2"]["door_y"] == [1, 0, 0, 0]  # fails ADA 404.2.3 only
    assert g.edges["R1", "R2"]["side"] == {"R1": "E", "R2": "W"}
