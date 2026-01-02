import pytest
import json
from CAPABILITY.PIPELINES.pipeline_dag import topo_sort, _parse_dag_spec, DagSpec, DagEdge

def test_dag_spec_validation():
    # TEST 1: Valid Spec
    valid_spec = {
        "dag_version": "1.0.0",
        "dag_id": "test-dag",
        "nodes": ["A", "B", "C"],
        "edges": [
            {"from": "A", "to": "B", "requires": ["CHAIN.json"]},
            {"from": "B", "to": "C", "requires": ["STATE.json"]}
        ]
    }
    
    spec = _parse_dag_spec(valid_spec)
    assert spec.dag_id == "test-dag"
    assert len(spec.nodes) == 3
    assert len(spec.edges) == 2

    # TEST 2: Invalid Version
    invalid_version = {
        "dag_version": "0.9.0",
        "dag_id": "test-dag",
        "nodes": ["A"],
        "edges": []
    }
    with pytest.raises(ValueError, match="DAG_INVALID_VERSION"):
        _parse_dag_spec(invalid_version)

    # TEST 3: Edge referencing missing node
    invalid_node = {
        "dag_version": "1.0.0",
        "dag_id": "test-dag",
        "nodes": ["A", "B"], 
        "edges": [
            {"from": "B", "to": "C", "requires": []} # C is missing from nodes
        ]
    }
    with pytest.raises(ValueError):
        _parse_dag_spec(invalid_node)
        
    # TEST 4: Cycle detection (topo_sort)
    cycle_spec_dict = {
        "dag_version": "1.0.0",
        "dag_id": "cycle-dag",
        "nodes": ["A", "B"],
        "edges": [
            {"from": "A", "to": "B", "requires": ["STATE.json"]},
            {"from": "B", "to": "A", "requires": ["STATE.json"]}
        ]
    }
    spec_cycle = _parse_dag_spec(cycle_spec_dict)
    with pytest.raises(ValueError, match="DAG_CYCLE_DETECTED"):
        topo_sort(spec_cycle)

def test_dag_topo_sort():
    spec = DagSpec(
        dag_id="topo-test",
        nodes=["C", "A", "B", "D"],
        edges=[
            DagEdge(src="A", dst="B", requires=[]),
            DagEdge(src="B", dst="C", requires=[]),
            DagEdge(src="A", dst="C", requires=[])
        ]
    )
    order = topo_sort(spec)
    assert order == ["A", "B", "C", "D"]

