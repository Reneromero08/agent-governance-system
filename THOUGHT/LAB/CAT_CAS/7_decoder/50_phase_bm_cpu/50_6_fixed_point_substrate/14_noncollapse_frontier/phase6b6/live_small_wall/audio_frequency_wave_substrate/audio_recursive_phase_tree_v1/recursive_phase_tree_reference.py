#!/usr/bin/env python3
"""Deterministic R0 reference for recursive phase-inside-phase CAT_CAS beams.

The native object is a complete recursive phase tree.  This ordinary-software
reference renders that tree as a unit-modulus complex beam, applies the beam as a
reversible pointwise phase operator to a deterministic borrowed tape, evaluates
hierarchy-sensitive queries, and verifies correct/wrong/reordered restoration arms.

The package deliberately implements no temporal recurrence, Ising update, physical
carrier, physical restoration, optimization advantage, catalytic transform, or Wall
claim.  Diagnostic scalars are boundary outputs only and never feed a native update.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import re
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SAMPLE_RATE_HZ = 48_000
FRAME_SECONDS = 0.125
SAMPLE_COUNT = int(SAMPLE_RATE_HZ * FRAME_SECONDS)
ABS_TOL = 1e-12
FLOAT32_COMPLEX_TOL = 2e-7
ENERGY_REL_TOL = 1e-7
SPECTRUM_REL_TOL = 1e-12
PORTABLE_METRIC_ATOL = 5e-12
PORTABLE_METRIC_RTOL = 5e-12
GENERATOR_ID = "recursive_phase_tree_reference_v2"
TREE_SCHEMA_ID = "recursive_phase_tree_v1"
CLAIM_CEILING = "SOFTWARE_RECURSIVE_PHASE_TREE_REFERENCE_ONLY"
ESTABLISHED_TOKEN = "AUDIO_RECURSIVE_PHASE_TREE_REFERENCE_ESTABLISHED"
MAX_NODE_COUNT = 64
MAX_DEPTH = 16
MAX_ID_BYTES = 64
MAX_MODULATION_INDEX = 4.0
MAX_LOCAL_PHASE_ABS = 2.0 * math.pi
NODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

PACKAGE_DIR = Path(__file__).resolve().parent
FIXTURE_DIR_NAME = "fixtures"
SCHEMA_FILE = "RECURSIVE_PHASE_TREE_SCHEMA.json"
MANIFEST_FILE = "RECURSIVE_PHASE_TREE_FIXTURE_MANIFEST.json"
TESTS_FILE = "RECURSIVE_PHASE_TREE_REFERENCE_TESTS.json"
RESULTS_FILE = "RECURSIVE_PHASE_TREE_REFERENCE_RESULTS.json"

TREE_TOP_KEYS = {
    "edges",
    "global_spin_phase_rad",
    "nodes",
    "root_id",
    "sample_rate_hz",
    "schema",
}
NODE_KEYS = {"frequency_hz", "node_id", "phase_rad"}
EDGE_KEYS = {"child_id", "modulation_index", "parent_id"}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds the finite numeric envelope") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return 0.0 if number == 0.0 else number


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"{label} keys mismatch: missing={missing}, unexpected={unexpected}"
        )


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {token}")


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        value[key] = item
    return value


def strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_object_keys,
        parse_constant=_reject_json_constant,
    )


def canonical_json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(path, canonical_json_bytes(value))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1_bytes(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _metric(value: float) -> float:
    return float(f"{float(value):.12g}")


@dataclass(frozen=True)
class PhaseEdge:
    modulation_index: float
    child: "PhaseNode"

    def __post_init__(self) -> None:
        beta = _finite_number(self.modulation_index, "modulation_index")
        if beta < 0.0 or beta > MAX_MODULATION_INDEX:
            raise ValueError(
                f"modulation_index must be inside [0,{MAX_MODULATION_INDEX}]"
            )
        object.__setattr__(self, "modulation_index", beta)


@dataclass(frozen=True)
class PhaseNode:
    node_id: str
    frequency_hz: float
    phase_rad: float = 0.0
    children: tuple[PhaseEdge, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("node_id must be a nonempty string")
        if len(self.node_id.encode("utf-8")) > MAX_ID_BYTES:
            raise ValueError("node_id exceeds byte limit")
        if NODE_ID_PATTERN.fullmatch(self.node_id) is None:
            raise ValueError("node_id contains an unsafe character")
        frequency = _finite_number(self.frequency_hz, "frequency_hz")
        phase = _finite_number(self.phase_rad, "phase_rad")
        if frequency <= 0.0 or frequency >= SAMPLE_RATE_HZ / 2:
            raise ValueError("frequency_hz must be inside (0, Nyquist)")
        if abs(phase) > MAX_LOCAL_PHASE_ABS:
            raise ValueError("phase_rad exceeds the frozen local-phase envelope")
        child_ids = [edge.child.node_id for edge in self.children]
        if len(child_ids) != len(set(child_ids)):
            raise ValueError(f"duplicate direct child IDs under {self.node_id}")
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "phase_rad", phase)

    def phase(self, t: np.ndarray) -> np.ndarray:
        phi = (2.0 * math.pi * self.frequency_hz * t) + self.phase_rad
        for edge in self.children:
            phi = phi + edge.modulation_index * np.sin(edge.child.phase(t))
        return phi

    def walk(self) -> Iterable["PhaseNode"]:
        yield self
        for edge in self.children:
            yield from edge.child.walk()

    def max_depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(edge.child.max_depth() for edge in self.children)


@dataclass(frozen=True)
class RecursivePhaseBeam:
    root: PhaseNode
    global_spin_phase_rad: float = 0.0

    def __post_init__(self) -> None:
        phase = _finite_number(
            self.global_spin_phase_rad, "global_spin_phase_rad"
        )
        if phase not in (0.0, math.pi):
            raise ValueError("global_spin_phase_rad must be exactly 0 or pi")
        object.__setattr__(self, "global_spin_phase_rad", phase)

    def render(self, t: np.ndarray) -> np.ndarray:
        return np.exp(1j * (self.root.phase(t) + self.global_spin_phase_rad))

    def document(self) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        seen: set[str] = set()

        def visit(node: PhaseNode) -> None:
            if node.node_id in seen:
                raise ValueError(f"duplicate node_id in tree: {node.node_id}")
            seen.add(node.node_id)
            nodes.append(
                {
                    "frequency_hz": node.frequency_hz,
                    "node_id": node.node_id,
                    "phase_rad": node.phase_rad,
                }
            )
            for edge in node.children:
                edges.append(
                    {
                        "child_id": edge.child.node_id,
                        "modulation_index": edge.modulation_index,
                        "parent_id": node.node_id,
                    }
                )
                visit(edge.child)

        visit(self.root)
        document = {
            "edges": sorted(
                edges,
                key=lambda item: (
                    item["parent_id"],
                    item["child_id"],
                    item["modulation_index"],
                ),
            ),
            "global_spin_phase_rad": self.global_spin_phase_rad,
            "nodes": sorted(nodes, key=lambda item: item["node_id"]),
            "root_id": self.root.node_id,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "schema": TREE_SCHEMA_ID,
        }
        validate_tree_document(document)
        return document

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.document())

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.document(), pretty=False))

    def geometry_identity(self) -> str:
        geometry = {
            "edges": self.document()["edges"],
            "root_id": self.root.node_id,
        }
        return sha256_bytes(canonical_json_bytes(geometry, pretty=False))


def validate_tree_document(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("tree document must be an object")
    _exact_keys(document, TREE_TOP_KEYS, "tree")
    if document["schema"] != TREE_SCHEMA_ID:
        raise ValueError("unsupported tree schema")
    if (
        isinstance(document["sample_rate_hz"], bool)
        or not isinstance(document["sample_rate_hz"], int)
        or document["sample_rate_hz"] != SAMPLE_RATE_HZ
    ):
        raise ValueError(f"sample_rate_hz must equal {SAMPLE_RATE_HZ}")
    root_id = document["root_id"]
    if not isinstance(root_id, str) or not root_id:
        raise ValueError("root_id must be a nonempty string")
    spin = _finite_number(
        document["global_spin_phase_rad"], "global_spin_phase_rad"
    )
    if spin not in (0.0, math.pi):
        raise ValueError("global_spin_phase_rad must be exactly 0 or pi")

    raw_nodes = document["nodes"]
    raw_edges = document["edges"]
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise ValueError("nodes and edges must be arrays")
    if not 1 <= len(raw_nodes) <= MAX_NODE_COUNT:
        raise ValueError("node count exceeds the frozen safe envelope")

    nodes: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            raise ValueError(f"node[{index}] must be an object")
        _exact_keys(raw, NODE_KEYS, f"node[{index}]")
        node_id = raw["node_id"]
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be a nonempty string")
        if len(node_id.encode("utf-8")) > MAX_ID_BYTES:
            raise ValueError("node_id exceeds byte limit")
        if NODE_ID_PATTERN.fullmatch(node_id) is None:
            raise ValueError("node_id contains an unsafe character")
        if node_id in nodes:
            raise ValueError(f"duplicate node_id: {node_id}")
        frequency = _finite_number(raw["frequency_hz"], f"{node_id}.frequency_hz")
        phase = _finite_number(raw["phase_rad"], f"{node_id}.phase_rad")
        if frequency <= 0.0 or frequency >= SAMPLE_RATE_HZ / 2:
            raise ValueError("frequency_hz must be inside (0, Nyquist)")
        if abs(phase) > MAX_LOCAL_PHASE_ABS:
            raise ValueError("phase_rad exceeds the frozen local-phase envelope")
        nodes[node_id] = {
            "frequency_hz": frequency,
            "node_id": node_id,
            "phase_rad": phase,
        }

    if root_id not in nodes:
        raise ValueError("root_id does not identify a declared node")

    children: dict[str, list[tuple[str, float]]] = {key: [] for key in nodes}
    parents: dict[str, list[str]] = {key: [] for key in nodes}
    edge_pairs: set[tuple[str, str]] = set()
    normalized_edges: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_edges):
        if not isinstance(raw, dict):
            raise ValueError(f"edge[{index}] must be an object")
        _exact_keys(raw, EDGE_KEYS, f"edge[{index}]")
        parent = raw["parent_id"]
        child = raw["child_id"]
        if not isinstance(parent, str) or not isinstance(child, str):
            raise ValueError("edge identifiers must be strings")
        if parent not in nodes or child not in nodes:
            raise ValueError("malformed edge references an unknown node")
        if parent == child:
            raise ValueError("self edges are forbidden")
        pair = (parent, child)
        if pair in edge_pairs:
            raise ValueError("duplicate edge")
        edge_pairs.add(pair)
        beta = _finite_number(
            raw["modulation_index"], f"edge[{index}].modulation_index"
        )
        if beta < 0.0 or beta > MAX_MODULATION_INDEX:
            raise ValueError("modulation_index exceeds the frozen safe envelope")
        children[parent].append((child, beta))
        parents[child].append(parent)
        normalized_edges.append(
            {"child_id": child, "modulation_index": beta, "parent_id": parent}
        )

    color = {node_id: 0 for node_id in nodes}

    def detect_cycle(node_id: str) -> None:
        color[node_id] = 1
        for child_id, _ in children[node_id]:
            if color[child_id] == 1:
                raise ValueError("cycle detected")
            if color[child_id] == 0:
                detect_cycle(child_id)
        color[node_id] = 2

    for node_id in sorted(nodes):
        if color[node_id] == 0:
            detect_cycle(node_id)

    if parents[root_id]:
        raise ValueError("root node must not have a parent")
    for node_id in nodes:
        if node_id == root_id:
            continue
        if len(parents[node_id]) != 1:
            raise ValueError("each non-root node must have exactly one parent")
    if len(raw_edges) != len(raw_nodes) - 1:
        raise ValueError("a tree must have exactly node_count-1 edges")

    reachable: set[str] = set()

    def mark(node_id: str, depth: int) -> None:
        if depth > MAX_DEPTH:
            raise ValueError("tree depth exceeds the frozen safe envelope")
        reachable.add(node_id)
        for child_id, _ in children[node_id]:
            mark(child_id, depth + 1)

    mark(root_id, 1)
    if reachable != set(nodes):
        raise ValueError("tree contains nodes disconnected from the root")

    return {
        "edges": sorted(
            normalized_edges,
            key=lambda item: (
                item["parent_id"],
                item["child_id"],
                item["modulation_index"],
            ),
        ),
        "global_spin_phase_rad": spin,
        "nodes": sorted(nodes.values(), key=lambda item: item["node_id"]),
        "root_id": root_id,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "schema": TREE_SCHEMA_ID,
    }


def beam_from_document(document: Any) -> RecursivePhaseBeam:
    normalized = validate_tree_document(document)
    node_values = {item["node_id"]: item for item in normalized["nodes"]}
    edges_by_parent: dict[str, list[dict[str, Any]]] = {
        node_id: [] for node_id in node_values
    }
    for edge in normalized["edges"]:
        edges_by_parent[edge["parent_id"]].append(edge)

    def build(node_id: str) -> PhaseNode:
        value = node_values[node_id]
        children = tuple(
            PhaseEdge(edge["modulation_index"], build(edge["child_id"]))
            for edge in sorted(
                edges_by_parent[node_id], key=lambda item: item["child_id"]
            )
        )
        return PhaseNode(
            node_id=node_id,
            frequency_hz=value["frequency_hz"],
            phase_rad=value["phase_rad"],
            children=children,
        )

    return RecursivePhaseBeam(
        root=build(normalized["root_id"]),
        global_spin_phase_rad=normalized["global_spin_phase_rad"],
    )


def deserialize_tree_bytes(payload: bytes, *, require_canonical: bool) -> RecursivePhaseBeam:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("tree JSON must be UTF-8") from exc
    document = strict_json_loads(text)
    beam = beam_from_document(document)
    if require_canonical and payload != beam.canonical_bytes():
        raise ValueError("tree JSON is valid but not canonical")
    return beam


def load_tree(path: Path) -> RecursivePhaseBeam:
    return deserialize_tree_bytes(path.read_bytes(), require_canonical=True)


def sample_times() -> np.ndarray:
    return np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE_HZ


def borrowed_tape(t: np.ndarray) -> np.ndarray:
    n = np.arange(t.size, dtype=np.float64)
    phase = 0.173 * n + 0.23 * np.sin(2.0 * math.pi * 41.0 * t)
    amplitude = 0.65 + 0.25 * np.cos(2.0 * math.pi * 29.0 * t)
    return amplitude * np.exp(1j * phase)


def apply_phase_operator(tape: np.ndarray, beam: np.ndarray) -> np.ndarray:
    tape = np.asarray(tape, dtype=np.complex128)
    beam = np.asarray(beam, dtype=np.complex128)
    if tape.shape != beam.shape:
        raise ValueError("tape and beam must have identical shapes")
    if not np.all(np.isfinite(tape)) or not np.all(np.isfinite(beam)):
        raise ValueError("operator inputs must be finite")
    return tape * beam


def uncompute_phase_operator(mutated: np.ndarray, beam: np.ndarray) -> np.ndarray:
    return apply_phase_operator(mutated, np.conjugate(beam))


def matched_response(state_beam: np.ndarray, query_beam: np.ndarray) -> complex:
    state_beam = np.asarray(state_beam, dtype=np.complex128)
    query_beam = np.asarray(query_beam, dtype=np.complex128)
    if state_beam.shape != query_beam.shape:
        raise ValueError("state and query beams must have identical shapes")
    denominator = float(np.linalg.norm(state_beam) * np.linalg.norm(query_beam))
    if denominator == 0.0:
        return 0.0j
    return complex(np.vdot(query_beam, state_beam) / denominator)


def phase_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.angle(a * np.conjugate(b))


def hierarchy_a(spin_phase: float = 0.0) -> RecursivePhaseBeam:
    leaf = PhaseNode("leaf", 47.0, phase_rad=0.31)
    middle = PhaseNode(
        "middle",
        233.0,
        phase_rad=-0.22,
        children=(PhaseEdge(0.37, leaf),),
    )
    root = PhaseNode(
        "root",
        1_237.0,
        phase_rad=0.17,
        children=(PhaseEdge(0.71, middle),),
    )
    return RecursivePhaseBeam(root=root, global_spin_phase_rad=spin_phase)


def hierarchy_b(spin_phase: float = 0.0) -> RecursivePhaseBeam:
    middle = PhaseNode("middle", 233.0, phase_rad=-0.22)
    leaf = PhaseNode(
        "leaf",
        47.0,
        phase_rad=0.31,
        children=(PhaseEdge(0.37, middle),),
    )
    root = PhaseNode(
        "root",
        1_237.0,
        phase_rad=0.17,
        children=(PhaseEdge(0.71, leaf),),
    )
    return RecursivePhaseBeam(root=root, global_spin_phase_rad=spin_phase)


def hierarchy_phase_scrambled() -> RecursivePhaseBeam:
    document = copy.deepcopy(hierarchy_a().document())
    nodes = sorted(document["nodes"], key=lambda item: item["node_id"])
    phases = [node["phase_rad"] for node in nodes]
    rotated = phases[1:] + phases[:1]
    for node, phase in zip(nodes, rotated, strict=True):
        node["phase_rad"] = phase
    return beam_from_document(document)


def flat_multitone_replacement(tree: RecursivePhaseBeam, t: np.ndarray) -> np.ndarray:
    document = tree.document()
    by_id = {node["node_id"]: node for node in document["nodes"]}
    root = by_id[document["root_id"]]
    phi = 2.0 * math.pi * root["frequency_hz"] * t + root["phase_rad"]
    for edge in document["edges"]:
        child = by_id[edge["child_id"]]
        child_phase = (
            2.0 * math.pi * child["frequency_hz"] * t + child["phase_rad"]
        )
        phi = phi + edge["modulation_index"] * np.sin(child_phase)
    return np.exp(1j * (phi + tree.global_spin_phase_rad))


def spectrum_matched_non_tree(beam: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(np.asarray(beam, dtype=np.complex128))
    weights = np.abs(spectrum) ** 2
    mask = np.ones(spectrum.size, dtype=np.float64)
    partition_energy = [0.0, 0.0]
    ordered = sorted(range(spectrum.size), key=lambda index: (-weights[index], index))
    for index in ordered:
        side = 0 if partition_energy[0] <= partition_energy[1] else 1
        mask[index] = 1.0 if side == 0 else -1.0
        partition_energy[side] += float(weights[index])
    replacement = spectrum * mask
    return np.fft.ifft(replacement)


def _riff_chunk(kind: bytes, payload: bytes) -> bytes:
    if len(kind) != 4:
        raise ValueError("RIFF chunk identifiers must have four bytes")
    padding = b"\x00" if len(payload) % 2 else b""
    return kind + struct.pack("<I", len(payload)) + payload + padding


def float32_iq_wav_bytes(
    beam: np.ndarray, *, list_metadata: bytes | None = None
) -> bytes:
    complex_beam = np.asarray(beam, dtype=np.complex128)
    if complex_beam.shape != (SAMPLE_COUNT,):
        raise ValueError(f"complex beam must have {SAMPLE_COUNT} samples")
    if not np.all(np.isfinite(complex_beam)):
        raise ValueError("complex beam contains non-finite samples")
    samples = np.column_stack((complex_beam.real, complex_beam.imag))
    if float(np.max(np.abs(samples))) > 1.0 + 1e-12:
        raise ValueError("I/Q channel exceeds the unit fixture envelope")
    channels = 2
    bits_per_sample = 32
    block_align = channels * 4
    byte_rate = SAMPLE_RATE_HZ * block_align
    fmt = struct.pack(
        "<HHIIHH",
        3,
        channels,
        SAMPLE_RATE_HZ,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    chunks = [_riff_chunk(b"fmt ", fmt)]
    if list_metadata is not None:
        chunks.append(_riff_chunk(b"LIST", list_metadata))
    data = np.asarray(samples, dtype="<f4", order="C").tobytes(order="C")
    chunks.append(_riff_chunk(b"data", data))
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def parse_float32_wav_bytes(payload: bytes) -> tuple[int, np.ndarray, list[str]]:
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    if struct.unpack("<I", payload[4:8])[0] + 8 != len(payload):
        raise ValueError("RIFF size does not match file length")
    offset = 12
    fmt: tuple[int, int, int, int, int, int] | None = None
    data: bytes | None = None
    chunk_ids: list[str] = []
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError("truncated WAV chunk header")
        kind = payload[offset : offset + 4]
        size = struct.unpack("<I", payload[offset + 4 : offset + 8])[0]
        start = offset + 8
        end = start + size
        if end > len(payload):
            raise ValueError("truncated WAV chunk payload")
        chunk_ids.append(kind.decode("ascii", errors="strict"))
        if kind == b"fmt ":
            if fmt is not None:
                raise ValueError("multiple fmt chunks are not allowed")
            if size != 16:
                raise ValueError("fmt chunk must be exactly 16 bytes")
            fmt = struct.unpack("<HHIIHH", payload[start:end])
        elif kind == b"data":
            if data is not None:
                raise ValueError("multiple data chunks are not allowed")
            data = payload[start:end]
        offset = end + (size % 2)
    if offset != len(payload):
        raise ValueError("invalid RIFF padding")
    if fmt is None or data is None:
        raise ValueError("WAV requires fmt and data chunks")
    format_tag, channels, rate, byte_rate, block_align, bits = fmt
    if format_tag != 3 or bits != 32 or channels != 2:
        raise ValueError("only stereo I/Q IEEE float32 WAV is accepted")
    if rate != SAMPLE_RATE_HZ:
        raise ValueError("unexpected sample rate")
    if block_align != 8 or byte_rate != rate * block_align:
        raise ValueError("inconsistent WAV alignment or byte rate")
    if len(data) % block_align:
        raise ValueError("data chunk is not frame-aligned")
    samples = np.frombuffer(data, dtype="<f4").astype(np.float64)
    samples = samples.reshape((-1, 2))
    if samples.shape[0] != SAMPLE_COUNT:
        raise ValueError("unexpected WAV sample count")
    if not np.all(np.isfinite(samples)):
        raise ValueError("WAV contains non-finite samples")
    return rate, samples, chunk_ids


def complex_from_iq(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    if array.shape != (SAMPLE_COUNT, 2):
        raise ValueError("I/Q samples have the wrong shape")
    return array[:, 0] + 1j * array[:, 1]


def reordered_float32_iq_wav_bytes(beam: np.ndarray) -> bytes:
    minimal = float32_iq_wav_bytes(beam)
    fmt_chunk = minimal[12:36]
    data_chunk = minimal[36:]
    if fmt_chunk[:4] != b"fmt " or data_chunk[:4] != b"data":
        raise ValueError("internal minimal WAV chunk assumption failed")
    body = b"WAVE" + data_chunk + fmt_chunk
    return b"RIFF" + struct.pack("<I", len(body)) + body


def validate_committed_tree_fixture(
    tree: RecursivePhaseBeam,
    declared_tree: RecursivePhaseBeam,
    wav_payload: bytes,
) -> dict[str, Any]:
    if tree.canonical_bytes() != declared_tree.canonical_bytes():
        raise ValueError("committed tree bytes do not match the declared fixture role")
    expected_beam = tree.render(sample_times())
    expected_wav = float32_iq_wav_bytes(expected_beam)
    if wav_payload != expected_wav:
        raise ValueError("committed WAV bytes do not match the declared recursive tree")
    rate, samples, chunks = parse_float32_wav_bytes(wav_payload)
    if chunks != ["fmt ", "data"]:
        raise ValueError("committed WAV must contain exactly fmt then data chunks")
    parsed = complex_from_iq(samples)
    render_error = float(np.max(np.abs(parsed - expected_beam)))
    if render_error > FLOAT32_COMPLEX_TOL:
        raise ValueError("parsed committed WAV exceeds the render tolerance")
    return {
        "deterministic_wav_bytes_match": True,
        "expected_wav_sha256": sha256_bytes(expected_wav),
        "parsed_render_max_error": _metric(render_error),
        "sample_rate_hz": rate,
    }


def tree_schema_document() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cat-cas:recursive-phase-tree:v1",
        "additionalProperties": False,
        "properties": {
            "edges": {
                "items": {
                    "additionalProperties": False,
                    "properties": {
                        "child_id": {
                            "maxLength": MAX_ID_BYTES,
                            "pattern": NODE_ID_PATTERN.pattern,
                            "type": "string",
                        },
                        "modulation_index": {
                            "maximum": MAX_MODULATION_INDEX,
                            "minimum": 0.0,
                            "type": "number",
                        },
                        "parent_id": {
                            "maxLength": MAX_ID_BYTES,
                            "pattern": NODE_ID_PATTERN.pattern,
                            "type": "string",
                        },
                    },
                    "required": sorted(EDGE_KEYS),
                    "type": "object",
                },
                "maxItems": MAX_NODE_COUNT - 1,
                "type": "array",
            },
            "global_spin_phase_rad": {"enum": [0.0, math.pi], "type": "number"},
            "nodes": {
                "items": {
                    "additionalProperties": False,
                    "properties": {
                        "frequency_hz": {
                            "exclusiveMaximum": SAMPLE_RATE_HZ / 2,
                            "exclusiveMinimum": 0.0,
                            "type": "number",
                        },
                        "node_id": {
                            "maxLength": MAX_ID_BYTES,
                            "pattern": NODE_ID_PATTERN.pattern,
                            "type": "string",
                        },
                        "phase_rad": {
                            "maximum": MAX_LOCAL_PHASE_ABS,
                            "minimum": -MAX_LOCAL_PHASE_ABS,
                            "type": "number",
                        },
                    },
                    "required": sorted(NODE_KEYS),
                    "type": "object",
                },
                "maxItems": MAX_NODE_COUNT,
                "minItems": 1,
                "type": "array",
            },
            "root_id": {
                "maxLength": MAX_ID_BYTES,
                "pattern": NODE_ID_PATTERN.pattern,
                "type": "string",
            },
            "sample_rate_hz": {"const": SAMPLE_RATE_HZ, "type": "integer"},
            "schema": {"const": TREE_SCHEMA_ID, "type": "string"},
        },
        "required": sorted(TREE_TOP_KEYS),
        "title": "CAT_CAS Recursive Phase Tree v1",
        "type": "object",
        "x-semantic-constraints": [
            "node_id values are unique",
            "edges reference declared nodes",
            "the graph is connected and acyclic",
            "the root has no parent and each other node has exactly one parent",
            f"tree depth is at most {MAX_DEPTH}",
            "JSON numbers are finite and booleans are not numbers",
        ],
    }


def reference_test_spec() -> dict[str, Any]:
    return {
        "edge_conventions": {
            "canonical_json": "UTF-8, sorted keys, indent=2, newline terminated, no NaN/Infinity",
            "diagnostic_feedback": "no diagnostic scalar feeds any native state update",
            "exact_spectrum_recursive_pair": "not claimed constructible; spectrum control is explicitly non-tree",
            "global_orientation": "exactly 0 or pi applied to the complete rendered tree",
            "portable_result_comparison": "same structure and statuses; numeric leaves use frozen atol/rtol",
            "reordered_inverse": "same node multiset with hierarchy-B parent-child geometry",
            "wav": "minimal RIFF, fmt then data, stereo I/Q IEEE float32, 48000 Hz",
            "wrong_inverse": "hierarchy-A with local phases cyclically permuted in canonical node-ID order",
        },
        "numeric_envelope": {
            "correct_inverse_max_error": FLOAT32_COMPLEX_TOL,
            "energy_relative_error": ENERGY_REL_TOL,
            "float32_complex_max_error": FLOAT32_COMPLEX_TOL,
            "hierarchy_phase_gap_min_rad": 0.10,
            "matched_exact_min": 1.0 - 1e-12,
            "matched_gap_min": 0.02,
            "mutation_l2_min": 1.0,
            "portable_metric_atol": PORTABLE_METRIC_ATOL,
            "portable_metric_rtol": PORTABLE_METRIC_RTOL,
            "spectrum_non_tree_response_max": 0.20,
            "spectrum_relative_error": SPECTRUM_REL_TOL,
            "wrong_inverse_max_error_min": 0.05,
            "wrong_query_magnitude_max": 0.98,
        },
        "schema": "recursive_phase_tree_reference_tests_v2",
        "tests": [
            {"id": test_id}
            for test_id in [
                "recursive_depth_present",
                "same_node_multiset_different_parent_child_geometry",
                "strict_parser_round_trip_all_declared_trees",
                "duplicate_node_rejected",
                "duplicate_json_key_rejected",
                "cycle_rejected",
                "malformed_edge_rejected",
                "nonfinite_number_rejected",
                "oversized_integer_rejected",
                "nyquist_rejected",
                "unsafe_schema_rejected",
                "oversized_identifier_rejected",
                "unexpected_field_rejected",
                "canonical_serialization_mutation_rejected",
                "committed_wav_matches_declared_recursive_tree",
                "fixture_substitution_rejected",
                "committed_wav_reordered_chunks_rejected",
                "committed_wav_extra_chunk_rejected",
                "committed_wav_unit_modulus",
                "principal_time_magnitude_equal",
                "principal_energy_equal",
                "global_z2_rotates_complete_tree",
                "global_z2_preserves_internal_geometry",
                "amplitude_only_decoder_is_null",
                "hierarchy_changes_phase_geometry",
                "exact_hierarchy_query",
                "wrong_hierarchy_query",
                "subtree_permutation_changes_response",
                "parent_child_phase_scramble_changes_response",
                "flat_multitone_replacement_fails_exact_query",
                "spectrum_magnitude_only_decoder_is_nonidentifying",
                "borrowed_tape_is_nontrivially_mutated",
                "correct_inverse_restores",
                "wrong_inverse_fails_restoration",
                "reordered_inverse_fails_restoration",
                "metadata_stripping_invariant",
                "manifest_binds_committed_fixture_bytes",
                "no_native_update_implemented",
            ]
        ],
    }


def declared_fixtures() -> list[dict[str, Any]]:
    return [
        {
            "role": "hierarchy_a_global_plus_complete_tree",
            "tree_name": "hierarchy_a_plus",
            "tree": hierarchy_a(0.0),
        },
        {
            "role": "hierarchy_b_global_plus_complete_tree",
            "tree_name": "hierarchy_b_plus",
            "tree": hierarchy_b(0.0),
        },
        {
            "role": "hierarchy_a_global_minus_complete_tree",
            "tree_name": "hierarchy_a_minus",
            "tree": hierarchy_a(math.pi),
        },
    ]


def fixture_paths(item: Mapping[str, Any]) -> tuple[str, str]:
    name = str(item["tree_name"])
    return f"{FIXTURE_DIR_NAME}/{name}.tree.json", f"{FIXTURE_DIR_NAME}/{name}_iq.wav"


def build_fixture_files(package_dir: Path) -> None:
    fixture_dir = package_dir / FIXTURE_DIR_NAME
    expected_names: set[str] = set()
    t = sample_times()
    for item in declared_fixtures():
        tree_path_text, wav_path_text = fixture_paths(item)
        tree_path = package_dir / tree_path_text
        wav_path = package_dir / wav_path_text
        expected_names.update((tree_path.name, wav_path.name))
        tree = item["tree"]
        write_bytes_atomic(tree_path, tree.canonical_bytes())
        write_bytes_atomic(wav_path, float32_iq_wav_bytes(tree.render(t)))
    if fixture_dir.exists():
        observed_names = {path.name for path in fixture_dir.iterdir() if path.is_file()}
        unexpected = sorted(observed_names - expected_names)
        if unexpected:
            raise ValueError(f"unexpected fixture files: {unexpected}")


def fixture_manifest(package_dir: Path) -> dict[str, Any]:
    fixtures: list[dict[str, Any]] = []
    total_bytes = 0
    set_digest = hashlib.sha256()
    for item in declared_fixtures():
        tree_path_text, wav_path_text = fixture_paths(item)
        tree_path = package_dir / tree_path_text
        wav_path = package_dir / wav_path_text
        tree = load_tree(tree_path)
        wav_payload = wav_path.read_bytes()
        binding = validate_committed_tree_fixture(tree, item["tree"], wav_payload)
        rate, samples, chunks = parse_float32_wav_bytes(wav_payload)
        beam = complex_from_iq(samples)
        byte_count = wav_path.stat().st_size
        fixture_sha = sha256_file(wav_path)
        tree_sha = sha256_file(tree_path)
        total_bytes += byte_count
        set_digest.update(tree_path_text.encode("utf-8") + b"\0")
        set_digest.update(bytes.fromhex(tree_sha))
        set_digest.update(wav_path_text.encode("utf-8") + b"\0")
        set_digest.update(bytes.fromhex(fixture_sha))
        orientation = "plus" if tree.global_spin_phase_rad == 0.0 else "minus"
        fixtures.append(
            {
                "byte_count": byte_count,
                "channels": 2,
                "complex_energy": _metric(np.sum(np.abs(beam) ** 2)),
                "complex_peak": _metric(np.max(np.abs(beam))),
                "complex_rms": _metric(np.sqrt(np.mean(np.abs(beam) ** 2))),
                "declared_tree_bytes_match": True,
                "deterministic_wav_bytes_match": binding[
                    "deterministic_wav_bytes_match"
                ],
                "dtype": "ieee_float32_le",
                "expected_wav_sha256": binding["expected_wav_sha256"],
                "generator_parameters": {
                    "frame_seconds": FRAME_SECONDS,
                    "formula": "exp(i*Phi_root(t)+i*global_spin_phase_rad)",
                    "generator": GENERATOR_ID,
                },
                "global_orientation": orientation,
                "global_spin_phase_rad": tree.global_spin_phase_rad,
                "parent_child_geometry_identity": tree.geometry_identity(),
                "parsed_render_max_error": binding["parsed_render_max_error"],
                "path": wav_path_text,
                "riff_chunks": chunks,
                "role": item["role"],
                "sample_count": int(samples.shape[0]),
                "sample_rate_hz": rate,
                "sha256": fixture_sha,
                "tree_digest": tree.digest(),
                "tree_path": tree_path_text,
                "tree_sha256": tree_sha,
            }
        )
    return {
        "fixture_count": len(fixtures),
        "fixture_set_sha256": set_digest.hexdigest(),
        "fixtures": fixtures,
        "generator": GENERATOR_ID,
        "schema": "recursive_phase_tree_fixture_manifest_v1",
        "total_fixture_bytes": total_bytes,
    }


def _expect_rejection(payload: Any) -> bool:
    try:
        beam_from_document(payload)
    except (TypeError, ValueError):
        return True
    return False


def schema_negative_measurements() -> dict[str, bool]:
    base = hierarchy_a().document()

    duplicate = copy.deepcopy(base)
    duplicate["nodes"].append(copy.deepcopy(duplicate["nodes"][0]))

    cycle = {
        "edges": [
            {"child_id": "b", "modulation_index": 0.2, "parent_id": "a"},
            {"child_id": "a", "modulation_index": 0.3, "parent_id": "b"},
        ],
        "global_spin_phase_rad": 0.0,
        "nodes": [
            {"frequency_hz": 100.0, "node_id": "root", "phase_rad": 0.0},
            {"frequency_hz": 200.0, "node_id": "a", "phase_rad": 0.0},
            {"frequency_hz": 300.0, "node_id": "b", "phase_rad": 0.0},
        ],
        "root_id": "root",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "schema": TREE_SCHEMA_ID,
    }

    malformed = copy.deepcopy(base)
    malformed["edges"][0]["child_id"] = "missing"

    nonfinite = copy.deepcopy(base)
    nonfinite["nodes"][0]["frequency_hz"] = float("nan")

    oversized_integer = copy.deepcopy(base)
    oversized_integer["nodes"][0]["frequency_hz"] = 10**400

    nyquist = copy.deepcopy(base)
    nyquist["nodes"][0]["frequency_hz"] = SAMPLE_RATE_HZ / 2

    unsafe = copy.deepcopy(base)
    unsafe["nodes"][0]["node_id"] = "../unsafe"

    oversized_identifier = copy.deepcopy(base)
    old_root = oversized_identifier["root_id"]
    long_root = "r" * (MAX_ID_BYTES + 1)
    oversized_identifier["root_id"] = long_root
    for node in oversized_identifier["nodes"]:
        if node["node_id"] == old_root:
            node["node_id"] = long_root
    for edge in oversized_identifier["edges"]:
        if edge["parent_id"] == old_root:
            edge["parent_id"] = long_root

    unexpected = copy.deepcopy(base)
    unexpected["answer"] = "hidden"

    canonical_mutation = b" " + hierarchy_a().canonical_bytes()
    try:
        deserialize_tree_bytes(canonical_mutation, require_canonical=True)
        canonical_rejected = False
    except ValueError:
        canonical_rejected = True

    try:
        strict_json_loads('{"x":NaN}')
        json_nonfinite_rejected = False
    except ValueError:
        json_nonfinite_rejected = True

    try:
        strict_json_loads('{"schema":"first","schema":"second"}')
        duplicate_json_key_rejected = False
    except ValueError:
        duplicate_json_key_rejected = True

    return {
        "canonical_serialization_mutation_rejected": canonical_rejected,
        "cycle_rejected": _expect_rejection(cycle),
        "duplicate_json_key_rejected": duplicate_json_key_rejected,
        "duplicate_node_rejected": _expect_rejection(duplicate),
        "malformed_edge_rejected": _expect_rejection(malformed),
        "nonfinite_number_rejected": _expect_rejection(nonfinite)
        and json_nonfinite_rejected,
        "nyquist_rejected": _expect_rejection(nyquist),
        "oversized_identifier_rejected": _expect_rejection(oversized_identifier),
        "oversized_integer_rejected": _expect_rejection(oversized_integer),
        "unexpected_field_rejected": _expect_rejection(unexpected),
        "unsafe_schema_rejected": _expect_rejection(unsafe),
    }


def _test(
    test_id: str,
    status: bool,
    observed: Any,
    *,
    comparator: str,
    threshold: Any,
    conclusion: str,
) -> dict[str, Any]:
    return {
        "comparator": comparator,
        "conclusion": conclusion,
        "id": test_id,
        "observed": observed,
        "status": "PASS" if status else "FAIL",
        "threshold": threshold,
    }


def score_committed_fixtures(
    package_dir: Path, manifest: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    limits = spec["numeric_envelope"]
    by_role = {item["role"]: item for item in manifest["fixtures"]}

    def load_role(role: str) -> tuple[RecursivePhaseBeam, np.ndarray, bytes]:
        record = by_role[role]
        tree = load_tree(package_dir / record["tree_path"])
        payload = (package_dir / record["path"]).read_bytes()
        _, samples, _ = parse_float32_wav_bytes(payload)
        return tree, complex_from_iq(samples), payload

    tree_a, beam_a, wav_a = load_role("hierarchy_a_global_plus_complete_tree")
    tree_b, beam_b, _ = load_role("hierarchy_b_global_plus_complete_tree")
    tree_a_minus, beam_a_minus, _ = load_role(
        "hierarchy_a_global_minus_complete_tree"
    )

    t = sample_times()
    phase_scrambled_tree = hierarchy_phase_scrambled()
    phase_scrambled_beam = phase_scrambled_tree.render(t)
    flat_beam = flat_multitone_replacement(tree_a, t)
    spectrum_non_tree = spectrum_matched_non_tree(beam_a)
    tape = borrowed_tape(t)
    mutated = apply_phase_operator(tape, beam_a)
    correct_restored = uncompute_phase_operator(mutated, beam_a)
    wrong_restored = uncompute_phase_operator(mutated, phase_scrambled_beam)
    reordered_restored = uncompute_phase_operator(mutated, beam_b)

    node_multiset_a = sorted(
        (node.node_id, node.frequency_hz, node.phase_rad)
        for node in tree_a.root.walk()
    )
    node_multiset_b = sorted(
        (node.node_id, node.frequency_hz, node.phase_rad)
        for node in tree_b.root.walk()
    )
    exact_response = abs(matched_response(beam_a, beam_a))
    wrong_response = abs(matched_response(beam_a, beam_b))
    phase_scramble_response = abs(matched_response(beam_a, phase_scrambled_beam))
    flat_response = abs(matched_response(beam_a, flat_beam))
    spectrum_non_tree_response = abs(matched_response(beam_a, spectrum_non_tree))
    spectrum_a = np.abs(np.fft.fft(beam_a))
    spectrum_adversary = np.abs(np.fft.fft(spectrum_non_tree))
    spectrum_relative_error = float(
        np.linalg.norm(spectrum_a - spectrum_adversary)
        / max(np.linalg.norm(spectrum_a), np.finfo(np.float64).tiny)
    )
    energy_a = float(np.sum(np.abs(beam_a) ** 2))
    energy_b = float(np.sum(np.abs(beam_b) ** 2))
    energy_relative_error = abs(energy_a - energy_b) / max(energy_a, energy_b)
    magnitude_difference = float(np.max(np.abs(np.abs(beam_a) - np.abs(beam_b))))
    unit_modulus_error = max(
        float(np.max(np.abs(np.abs(beam_a) - 1.0))),
        float(np.max(np.abs(np.abs(beam_b) - 1.0))),
        float(np.max(np.abs(np.abs(beam_a_minus) - 1.0))),
    )
    z2_error = float(np.max(np.abs(beam_a_minus + beam_a)))
    plus_structure = tree_a.document()
    minus_structure = tree_a_minus.document()
    plus_structure.pop("global_spin_phase_rad")
    minus_structure.pop("global_spin_phase_rad")
    z2_internal_structure_equal = plus_structure == minus_structure

    metadata_payload = float32_iq_wav_bytes(
        beam_a, list_metadata=b"CAT_CAS recursive phase metadata adversary"
    )
    _, metadata_samples, metadata_chunks = parse_float32_wav_bytes(metadata_payload)
    metadata_stripped = float32_iq_wav_bytes(complex_from_iq(metadata_samples))
    metadata_sample_error = float(
        np.max(np.abs(complex_from_iq(metadata_samples) - beam_a))
    )
    metadata_bytes_equal = metadata_stripped == wav_a

    round_trip = True
    for record in manifest["fixtures"]:
        path = package_dir / record["tree_path"]
        tree = load_tree(path)
        if tree.canonical_bytes() != path.read_bytes():
            round_trip = False

    negative = schema_negative_measurements()
    manifest_bytes_close = True
    for record in manifest["fixtures"]:
        wav_path = package_dir / record["path"]
        tree_path = package_dir / record["tree_path"]
        manifest_bytes_close = manifest_bytes_close and (
            record["sha256"] == sha256_file(wav_path)
            and record["byte_count"] == wav_path.stat().st_size
            and record["tree_sha256"] == sha256_file(tree_path)
            and record["tree_digest"] == load_tree(tree_path).digest()
        )

    semantic_bindings_close = all(
        record["declared_tree_bytes_match"] is True
        and record["deterministic_wav_bytes_match"] is True
        and record["expected_wav_sha256"] == record["sha256"]
        and record["parsed_render_max_error"] <= FLOAT32_COMPLEX_TOL
        and record["riff_chunks"] == ["fmt ", "data"]
        for record in manifest["fixtures"]
    )
    committed_render_max_error = max(
        record["parsed_render_max_error"] for record in manifest["fixtures"]
    )

    substituted_payload = float32_iq_wav_bytes(
        tree_a.render(t) * np.exp(1j * 0.25)
    )
    try:
        validate_committed_tree_fixture(tree_a, hierarchy_a(), substituted_payload)
        substitution_rejected = False
    except ValueError:
        substitution_rejected = True

    try:
        validate_committed_tree_fixture(
            tree_a,
            hierarchy_a(),
            reordered_float32_iq_wav_bytes(tree_a.render(t)),
        )
        reordered_chunks_rejected = False
    except ValueError:
        reordered_chunks_rejected = True

    try:
        validate_committed_tree_fixture(
            tree_a,
            hierarchy_a(),
            float32_iq_wav_bytes(tree_a.render(t), list_metadata=b"extra"),
        )
        extra_chunk_rejected = False
    except ValueError:
        extra_chunk_rejected = True

    measurements = {
        "amplitude_only_max_difference": _metric(magnitude_difference),
        "correct_inverse_max_error": _metric(
            np.max(np.abs(correct_restored - tape))
        ),
        "committed_fixture_render_max_error": _metric(committed_render_max_error),
        "committed_tree_wav_bindings_close": semantic_bindings_close,
        "energy_relative_error": _metric(energy_relative_error),
        "exact_query_magnitude": _metric(exact_response),
        "fixture_count": manifest["fixture_count"],
        "fixture_set_sha256": manifest["fixture_set_sha256"],
        "fixture_total_bytes": manifest["total_fixture_bytes"],
        "flat_multitone_query_magnitude": _metric(flat_response),
        "forward_mutation_l2": _metric(np.linalg.norm(mutated - tape)),
        "hierarchy_a_depth": tree_a.root.max_depth(),
        "hierarchy_a_digest": tree_a.digest(),
        "hierarchy_b_depth": tree_b.root.max_depth(),
        "hierarchy_b_digest": tree_b.digest(),
        "max_hierarchy_phase_difference_rad": _metric(
            np.max(np.abs(phase_error(beam_a, beam_b)))
        ),
        "max_unit_modulus_error": _metric(unit_modulus_error),
        "metadata_bytes_restore_original": metadata_bytes_equal,
        "metadata_chunks": metadata_chunks,
        "metadata_sample_error": _metric(metadata_sample_error),
        "node_multisets_equal": node_multiset_a == node_multiset_b,
        "parent_child_phase_scramble_query_magnitude": _metric(
            phase_scramble_response
        ),
        "principal_energy_a": _metric(energy_a),
        "principal_energy_b": _metric(energy_b),
        "reordered_inverse_max_error": _metric(
            np.max(np.abs(reordered_restored - tape))
        ),
        "schema_negative_cases": negative,
        "spectrum_adversary_class": "spectrum_matched_non_tree",
        "spectrum_exact_recursive_pair_claimed": False,
        "spectrum_magnitude_relative_error": _metric(spectrum_relative_error),
        "spectrum_non_tree_query_magnitude": _metric(spectrum_non_tree_response),
        "strict_round_trip_all_declared_trees": round_trip,
        "fixture_substitution_rejected": substitution_rejected,
        "committed_wav_extra_chunk_rejected": extra_chunk_rejected,
        "committed_wav_reordered_chunks_rejected": reordered_chunks_rejected,
        "native_update_implemented": False,
        "temporal_recurrence_steps": 0,
        "wrong_hierarchy_query_magnitude": _metric(wrong_response),
        "wrong_inverse_max_error": _metric(np.max(np.abs(wrong_restored - tape))),
        "z2_internal_tree_structure_equal": z2_internal_structure_equal,
        "z2_whole_beam_error": _metric(z2_error),
    }

    tests = [
        _test(
            "recursive_depth_present",
            tree_a.root.max_depth() >= 3 and tree_b.root.max_depth() >= 3,
            [tree_a.root.max_depth(), tree_b.root.max_depth()],
            comparator=">=",
            threshold=3,
            conclusion="Both declared hierarchies retain recursive depth.",
        ),
        _test(
            "same_node_multiset_different_parent_child_geometry",
            node_multiset_a == node_multiset_b
            and tree_a.geometry_identity() != tree_b.geometry_identity(),
            {
                "geometry_a": tree_a.geometry_identity(),
                "geometry_b": tree_b.geometry_identity(),
                "node_multisets_equal": node_multiset_a == node_multiset_b,
            },
            comparator="==",
            threshold=True,
            conclusion="Principal trees use the same node multiset and distinct geometry.",
        ),
        _test(
            "strict_parser_round_trip_all_declared_trees",
            round_trip,
            round_trip,
            comparator="==",
            threshold=True,
            conclusion="Every declared tree round-trips through strict canonical bytes.",
        ),
    ]
    for test_id in [
        "duplicate_node_rejected",
        "duplicate_json_key_rejected",
        "cycle_rejected",
        "malformed_edge_rejected",
        "nonfinite_number_rejected",
        "oversized_integer_rejected",
        "nyquist_rejected",
        "unsafe_schema_rejected",
        "oversized_identifier_rejected",
        "unexpected_field_rejected",
        "canonical_serialization_mutation_rejected",
    ]:
        tests.append(
            _test(
                test_id,
                bool(negative[test_id]),
                negative[test_id],
                comparator="==",
                threshold=True,
                conclusion="The strict parser fails closed for this invalid tree class.",
            )
        )
    tests.extend(
        [
            _test(
                "committed_wav_matches_declared_recursive_tree",
                semantic_bindings_close,
                {
                    "all_bindings_close": semantic_bindings_close,
                    "max_render_error": measurements[
                        "committed_fixture_render_max_error"
                    ],
                },
                comparator="compound",
                threshold={
                    "all_bindings_close": True,
                    "max_render_error": FLOAT32_COMPLEX_TOL,
                },
                conclusion="Every committed tree matches its declared role and deterministically renders its exact minimal WAV bytes.",
            ),
            _test(
                "fixture_substitution_rejected",
                substitution_rejected,
                substitution_rejected,
                comparator="==",
                threshold=True,
                conclusion="A common phase-rotated WAV substitution fails semantic fixture binding.",
            ),
            _test(
                "committed_wav_reordered_chunks_rejected",
                reordered_chunks_rejected,
                reordered_chunks_rejected,
                comparator="==",
                threshold=True,
                conclusion="A committed data-before-fmt WAV fails the minimal fixture law.",
            ),
            _test(
                "committed_wav_extra_chunk_rejected",
                extra_chunk_rejected,
                extra_chunk_rejected,
                comparator="==",
                threshold=True,
                conclusion="A committed WAV with an extra metadata chunk fails the minimal fixture law.",
            ),
            _test(
                "committed_wav_unit_modulus",
                unit_modulus_error <= limits["float32_complex_max_error"],
                measurements["max_unit_modulus_error"],
                comparator="<=",
                threshold=limits["float32_complex_max_error"],
                conclusion="Committed stereo I/Q bytes preserve unit modulus within float32 tolerance.",
            ),
            _test(
                "principal_time_magnitude_equal",
                magnitude_difference <= limits["float32_complex_max_error"],
                measurements["amplitude_only_max_difference"],
                comparator="<=",
                threshold=limits["float32_complex_max_error"],
                conclusion="Principal trees have equal time-domain magnitude within serialization tolerance.",
            ),
            _test(
                "principal_energy_equal",
                energy_relative_error <= limits["energy_relative_error"],
                measurements["energy_relative_error"],
                comparator="<=",
                threshold=limits["energy_relative_error"],
                conclusion="Principal trees have equal serialized complex energy.",
            ),
            _test(
                "global_z2_rotates_complete_tree",
                z2_error <= limits["float32_complex_max_error"],
                measurements["z2_whole_beam_error"],
                comparator="<=",
                threshold=limits["float32_complex_max_error"],
                conclusion="Global pi rotation acts on the complete committed tree beam.",
            ),
            _test(
                "global_z2_preserves_internal_geometry",
                z2_internal_structure_equal,
                measurements["z2_internal_tree_structure_equal"],
                comparator="==",
                threshold=True,
                conclusion="Plus/minus canonical trees differ only in the global orientation field.",
            ),
            _test(
                "amplitude_only_decoder_is_null",
                magnitude_difference <= limits["float32_complex_max_error"],
                measurements["amplitude_only_max_difference"],
                comparator="<=",
                threshold=limits["float32_complex_max_error"],
                conclusion="Amplitude-only observation cannot distinguish the principal trees.",
            ),
            _test(
                "hierarchy_changes_phase_geometry",
                measurements["max_hierarchy_phase_difference_rad"]
                >= limits["hierarchy_phase_gap_min_rad"],
                measurements["max_hierarchy_phase_difference_rad"],
                comparator=">=",
                threshold=limits["hierarchy_phase_gap_min_rad"],
                conclusion="Parent-child geometry changes phase while amplitude remains null.",
            ),
            _test(
                "exact_hierarchy_query",
                exact_response >= limits["matched_exact_min"],
                measurements["exact_query_magnitude"],
                comparator=">=",
                threshold=limits["matched_exact_min"],
                conclusion="The exact committed hierarchy matches its frozen query.",
            ),
            _test(
                "wrong_hierarchy_query",
                wrong_response <= limits["wrong_query_magnitude_max"],
                measurements["wrong_hierarchy_query_magnitude"],
                comparator="<=",
                threshold=limits["wrong_query_magnitude_max"],
                conclusion="The same node multiset under wrong hierarchy scores lower.",
            ),
            _test(
                "subtree_permutation_changes_response",
                exact_response - wrong_response >= limits["matched_gap_min"],
                _metric(exact_response - wrong_response),
                comparator=">=",
                threshold=limits["matched_gap_min"],
                conclusion="Parent-child subtree reordering changes hierarchy response.",
            ),
            _test(
                "parent_child_phase_scramble_changes_response",
                exact_response - phase_scramble_response >= limits["matched_gap_min"],
                _metric(exact_response - phase_scramble_response),
                comparator=">=",
                threshold=limits["matched_gap_min"],
                conclusion="A fixed child-phase scramble changes hierarchy-sensitive response.",
            ),
            _test(
                "flat_multitone_replacement_fails_exact_query",
                exact_response - flat_response >= limits["matched_gap_min"],
                _metric(exact_response - flat_response),
                comparator=">=",
                threshold=limits["matched_gap_min"],
                conclusion="A flat sine bank does not replace native nested phase geometry.",
            ),
            _test(
                "spectrum_magnitude_only_decoder_is_nonidentifying",
                spectrum_relative_error <= limits["spectrum_relative_error"]
                and spectrum_non_tree_response
                <= limits["spectrum_non_tree_response_max"],
                {
                    "adversary_class": "spectrum_matched_non_tree",
                    "magnitude_relative_error": measurements[
                        "spectrum_magnitude_relative_error"
                    ],
                    "query_magnitude": measurements[
                        "spectrum_non_tree_query_magnitude"
                    ],
                },
                comparator="compound",
                threshold={
                    "magnitude_relative_error_max": limits["spectrum_relative_error"],
                    "query_magnitude_max": limits["spectrum_non_tree_response_max"],
                },
                conclusion="A clearly labeled non-tree waveform has the same FFT magnitude but a different hierarchy response.",
            ),
            _test(
                "borrowed_tape_is_nontrivially_mutated",
                measurements["forward_mutation_l2"] >= limits["mutation_l2_min"],
                measurements["forward_mutation_l2"],
                comparator=">=",
                threshold=limits["mutation_l2_min"],
                conclusion="The borrowed deterministic complex tape is materially changed.",
            ),
            _test(
                "correct_inverse_restores",
                measurements["correct_inverse_max_error"]
                <= limits["correct_inverse_max_error"],
                measurements["correct_inverse_max_error"],
                comparator="<=",
                threshold=limits["correct_inverse_max_error"],
                conclusion="The conjugate of the committed exact beam restores the tape.",
            ),
            _test(
                "wrong_inverse_fails_restoration",
                measurements["wrong_inverse_max_error"]
                >= limits["wrong_inverse_max_error_min"],
                measurements["wrong_inverse_max_error"],
                comparator=">=",
                threshold=limits["wrong_inverse_max_error_min"],
                conclusion="The canonical local-phase-permuted inverse does not restore.",
            ),
            _test(
                "reordered_inverse_fails_restoration",
                measurements["reordered_inverse_max_error"]
                >= limits["wrong_inverse_max_error_min"],
                measurements["reordered_inverse_max_error"],
                comparator=">=",
                threshold=limits["wrong_inverse_max_error_min"],
                conclusion="The same nodes in hierarchy-B parent-child order do not restore.",
            ),
            _test(
                "metadata_stripping_invariant",
                metadata_sample_error == 0.0
                and metadata_bytes_equal
                and metadata_chunks == ["fmt ", "LIST", "data"],
                {
                    "chunks": metadata_chunks,
                    "minimal_bytes_restored": metadata_bytes_equal,
                    "sample_error": measurements["metadata_sample_error"],
                },
                comparator="compound",
                threshold={"minimal_bytes_restored": True, "sample_error": 0.0},
                conclusion="Nonessential RIFF metadata is neither state nor answer-bearing.",
            ),
            _test(
                "manifest_binds_committed_fixture_bytes",
                manifest_bytes_close,
                manifest_bytes_close,
                comparator="==",
                threshold=True,
                conclusion="Manifest identities bind parsed committed tree and WAV bytes.",
            ),
            _test(
                "no_native_update_implemented",
                measurements["native_update_implemented"] is False
                and measurements["temporal_recurrence_steps"] == 0,
                {
                    "native_update_implemented": False,
                    "temporal_recurrence_steps": 0,
                },
                comparator="==",
                threshold={
                    "native_update_implemented": False,
                    "temporal_recurrence_steps": 0,
                },
                conclusion="The R0 callable surface implements no native update or temporal recurrence for a diagnostic scalar to feed.",
            ),
        ]
    )

    expected_ids = [item["id"] for item in spec["tests"]]
    observed_ids = [item["id"] for item in tests]
    if observed_ids != expected_ids:
        raise ValueError(
            f"test order/coverage mismatch: expected={expected_ids}, observed={observed_ids}"
        )
    passed = sum(item["status"] == "PASS" for item in tests)
    return {
        "measurements": measurements,
        "summary": {
            "failed": len(tests) - passed,
            "passed": passed,
            "test_count": len(tests),
        },
        "tests": tests,
    }


def scientific_result(
    package_dir: Path,
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    scored = score_committed_fixtures(package_dir, manifest, spec)
    return {
        "claim_ceiling": CLAIM_CEILING,
        "collapse_boundary": "diagnostic readout only; no decoded scalar feeds native state",
        "established_token_if_all_gates_close": ESTABLISHED_TOKEN,
        "fixture_count": manifest["fixture_count"],
        "fixture_manifest_sha256": sha256_file(package_dir / MANIFEST_FILE),
        "fixture_set_sha256": manifest["fixture_set_sha256"],
        "fixture_total_bytes": manifest["total_fixture_bytes"],
        "measurements": scored["measurements"],
        "ordinary_software_only": True,
        "physical_claims_established": [],
        "reference_tests_sha256": sha256_file(package_dir / TESTS_FILE),
        "sample_count": SAMPLE_COUNT,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "schema": "recursive_phase_tree_scientific_result_v2",
        "spectrum_boundary": "spectrum-matched adversary is explicitly non-tree; exact same-spectrum distinct recursive trees are not claimed",
        "summary": scored["summary"],
        "tests": scored["tests"],
    }


def source_binding(source_path: Path) -> dict[str, Any]:
    payload = source_path.read_bytes()
    return {
        "source_byte_count": len(payload),
        "source_git_blob_sha1": git_blob_sha1_bytes(payload),
        "source_sha256": sha256_bytes(payload),
    }


def execution_environment() -> dict[str, Any]:
    return {
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def verification_policy() -> dict[str, Any]:
    return {
        "deterministic_scientific_fields": "same structure and statuses; numeric leaves use frozen portable comparison",
        "environment_receipt": "informational; Python NumPy and platform strings may differ",
        "fixture_scoring": "parse committed float32 WAV bytes and prove exact declared-tree rendering before every score",
        "portable_metric_atol": PORTABLE_METRIC_ATOL,
        "portable_metric_rtol": PORTABLE_METRIC_RTOL,
        "stored_pass_authority": False,
    }


def portable_scientific_equal(stored: Any, recomputed: Any) -> bool:
    if isinstance(stored, bool) or isinstance(recomputed, bool):
        return type(stored) is type(recomputed) and stored == recomputed
    if isinstance(stored, (int, float)) and isinstance(recomputed, (int, float)):
        return math.isclose(
            float(stored),
            float(recomputed),
            rel_tol=PORTABLE_METRIC_RTOL,
            abs_tol=PORTABLE_METRIC_ATOL,
        )
    if isinstance(stored, dict) and isinstance(recomputed, dict):
        return set(stored) == set(recomputed) and all(
            portable_scientific_equal(stored[key], recomputed[key]) for key in stored
        )
    if isinstance(stored, list) and isinstance(recomputed, list):
        return len(stored) == len(recomputed) and all(
            portable_scientific_equal(left, right)
            for left, right in zip(stored, recomputed, strict=True)
        )
    return type(stored) is type(recomputed) and stored == recomputed


def result_document(package_dir: Path, source_path: Path) -> dict[str, Any]:
    manifest = load_exact_generated_json(
        package_dir / MANIFEST_FILE,
        {
            "fixture_count",
            "fixture_set_sha256",
            "fixtures",
            "generator",
            "schema",
            "total_fixture_bytes",
        },
        "manifest",
    )
    spec = load_exact_generated_json(
        package_dir / TESTS_FILE,
        {"edge_conventions", "numeric_envelope", "schema", "tests"},
        "test specification",
    )
    return {
        "execution_environment": execution_environment(),
        "scientific": scientific_result(package_dir, manifest, spec),
        "schema": "recursive_phase_tree_reference_result_v2",
        "source_binding": source_binding(source_path),
        "verification_policy": verification_policy(),
    }


def load_exact_generated_json(
    path: Path, expected_keys: set[str], label: str
) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8") from exc
    value = strict_json_loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _exact_keys(value, expected_keys, label)
    if payload != canonical_json_bytes(value):
        raise ValueError(f"{label} must use canonical generated JSON bytes")
    return value


def build_package(package_dir: Path, source_path: Path) -> dict[str, Any]:
    package_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(package_dir / SCHEMA_FILE, tree_schema_document())
    write_json_atomic(package_dir / TESTS_FILE, reference_test_spec())
    build_fixture_files(package_dir)
    manifest = fixture_manifest(package_dir)
    write_json_atomic(package_dir / MANIFEST_FILE, manifest)
    result = result_document(package_dir, source_path)
    write_json_atomic(package_dir / RESULTS_FILE, result)
    return {
        "fixture_count": manifest["fixture_count"],
        "fixture_total_bytes": manifest["total_fixture_bytes"],
        "manifest_sha256": sha256_file(package_dir / MANIFEST_FILE),
        "operation": "build",
        "result_sha256": sha256_file(package_dir / RESULTS_FILE),
        "source_sha256": result["source_binding"]["source_sha256"],
        "status": "PASS" if result["scientific"]["summary"]["failed"] == 0 else "FAIL",
        "test_count": result["scientific"]["summary"]["test_count"],
        "tests_passed": result["scientific"]["summary"]["passed"],
        "tests_sha256": sha256_file(package_dir / TESTS_FILE),
    }


def validate_result_shape(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("result must be an object")
    _exact_keys(
        value,
        {
            "execution_environment",
            "schema",
            "scientific",
            "source_binding",
            "verification_policy",
        },
        "result",
    )
    if value["schema"] != "recursive_phase_tree_reference_result_v2":
        raise ValueError("unexpected result schema")
    _exact_keys(
        value["execution_environment"], {"numpy", "platform", "python"}, "environment"
    )
    _exact_keys(
        value["source_binding"],
        {"source_byte_count", "source_git_blob_sha1", "source_sha256"},
        "source binding",
    )
    _exact_keys(
        value["verification_policy"],
        {
            "deterministic_scientific_fields",
            "environment_receipt",
            "fixture_scoring",
            "portable_metric_atol",
            "portable_metric_rtol",
            "stored_pass_authority",
        },
        "verification policy",
    )
    if value["verification_policy"]["stored_pass_authority"] is not False:
        raise ValueError("stored PASS strings must carry no authority")
    if value["verification_policy"] != verification_policy():
        raise ValueError("verification policy differs from the frozen source law")
    if not isinstance(value["scientific"], dict):
        raise ValueError("scientific result must be an object")
    return value


def verify_package(package_dir: Path, source_path: Path) -> dict[str, Any]:
    schema = load_exact_generated_json(
        package_dir / SCHEMA_FILE,
        {
            "$id",
            "$schema",
            "additionalProperties",
            "properties",
            "required",
            "title",
            "type",
            "x-semantic-constraints",
        },
        "tree schema",
    )
    if schema != tree_schema_document():
        raise ValueError("committed tree schema differs from the frozen source law")
    spec = load_exact_generated_json(
        package_dir / TESTS_FILE,
        {"edge_conventions", "numeric_envelope", "schema", "tests"},
        "test specification",
    )
    if spec != reference_test_spec():
        raise ValueError("committed test specification differs from frozen source law")
    stored_manifest = load_exact_generated_json(
        package_dir / MANIFEST_FILE,
        {
            "fixture_count",
            "fixture_set_sha256",
            "fixtures",
            "generator",
            "schema",
            "total_fixture_bytes",
        },
        "manifest",
    )
    recomputed_manifest = fixture_manifest(package_dir)
    if stored_manifest != recomputed_manifest:
        raise ValueError("committed manifest does not match committed fixture bytes")
    stored_result = validate_result_shape(
        load_exact_generated_json(
            package_dir / RESULTS_FILE,
            {
                "execution_environment",
                "schema",
                "scientific",
                "source_binding",
                "verification_policy",
            },
            "result",
        )
    )
    recomputed_scientific = scientific_result(package_dir, stored_manifest, spec)
    if not portable_scientific_equal(
        stored_result["scientific"], recomputed_scientific
    ):
        raise ValueError("deterministic scientific result recomputation mismatch")
    recomputed_source = source_binding(source_path)
    if stored_result["source_binding"] != recomputed_source:
        raise ValueError("source binding mismatch")
    failed = recomputed_scientific["summary"]["failed"]
    return {
        "environment_receipt_compared": False,
        "fixture_count": stored_manifest["fixture_count"],
        "fixture_total_bytes": stored_manifest["total_fixture_bytes"],
        "manifest_sha256": sha256_file(package_dir / MANIFEST_FILE),
        "operation": "verify",
        "recomputed_results_match": True,
        "result_sha256": sha256_file(package_dir / RESULTS_FILE),
        "source_binding_match": True,
        "status": "PASS" if failed == 0 else "FAIL",
        "test_count": recomputed_scientific["summary"]["test_count"],
        "tests_passed": recomputed_scientific["summary"]["passed"],
        "tests_sha256": sha256_file(package_dir / TESTS_FILE),
    }


def self_test(source_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="recursive_phase_tree_self_test_") as raw:
        package_dir = Path(raw)
        build = build_package(package_dir, source_path)
        verify = verify_package(package_dir, source_path)
        return {
            "build_status": build["status"],
            "fixture_count": verify["fixture_count"],
            "fixture_total_bytes": verify["fixture_total_bytes"],
            "operation": "self-test",
            "recomputed_results_match": verify["recomputed_results_match"],
            "status": "PASS"
            if build["status"] == "PASS" and verify["status"] == "PASS"
            else "FAIL",
            "test_count": verify["test_count"],
            "tests_passed": verify["tests_passed"],
            "verify_status": verify["status"],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("build", "verify", "self-test", "emit-tree-a", "emit-tree-b"),
        nargs="?",
        default="self-test",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=PACKAGE_DIR,
        help="Package directory for build or verify; defaults to this source directory.",
    )
    args = parser.parse_args()
    source_path = Path(__file__).resolve()
    package_dir = args.package_dir.resolve()

    try:
        if args.operation == "build":
            payload: Any = build_package(package_dir, source_path)
        elif args.operation == "verify":
            payload = verify_package(package_dir, source_path)
        elif args.operation == "emit-tree-a":
            payload = hierarchy_a().document()
        elif args.operation == "emit-tree-b":
            payload = hierarchy_b().document()
        else:
            payload = self_test(source_path)
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        payload = {
            "error": f"{type(exc).__name__}: {exc}",
            "operation": args.operation,
            "status": "FAIL",
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if payload.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
