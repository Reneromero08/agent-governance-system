from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
ENGINE_SOURCE = PACKAGE_DIR / "phase_native_engine.py"


def load_engine() -> Any:
    existing = sys.modules.get("phase_native_engine_shared")
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        "phase_native_engine_shared", ENGINE_SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load phase-native engine")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = load_engine()


def programs() -> tuple[Any, ...]:
    return (
        engine.PhaseProgram(
            name="prospective_affine_mod7",
            radix=7,
            register_count=3,
            input_registers=(0, 1),
            output_registers=(2,),
            statements=(
                engine.PhaseCall("DOUBLE_ADD", (0, 2)),
                engine.PhaseInstruction("ADD", (0, 2)),
                engine.PhaseCall("DOUBLE_ADD", (1, 2)),
                engine.PhaseInstruction("ROT", (2, 4)),
            ),
            computational_class="modular arithmetic",
        ),
        engine.PhaseProgram(
            name="prospective_binary_add3",
            radix=2,
            register_count=12,
            input_registers=(0, 1, 2, 3, 4, 5),
            output_registers=(6, 8, 10, 11),
            statements=(
                engine.PhaseCall("XOR2_INTO", (0, 3, 6)),
                engine.PhaseCall("AND_INTO", (0, 3, 7)),
                engine.PhaseCall("XOR3_INTO", (1, 4, 7, 8)),
                engine.PhaseCall("MAJORITY3_INTO", (1, 4, 7, 9)),
                engine.PhaseCall("XOR3_INTO", (2, 5, 9, 10)),
                engine.PhaseCall("MAJORITY3_INTO", (2, 5, 9, 11)),
            ),
            computational_class="multi-stage binary arithmetic",
        ),
        engine.PhaseProgram(
            name="prospective_mux_xor_pipeline",
            radix=2,
            register_count=6,
            input_registers=(0, 1, 2, 3),
            output_registers=(4, 5),
            statements=(
                engine.PhaseCall("MUX_INTO", (0, 1, 2, 4)),
                engine.PhaseInstruction("ADD", (4, 5)),
                engine.PhaseInstruction("ADD", (3, 5)),
            ),
            computational_class="conditional pipeline",
        ),
        engine.PhaseProgram(
            name="prospective_route_compose_mod7",
            radix=7,
            register_count=3,
            input_registers=(0, 1, 2),
            output_registers=(0, 1, 2),
            statements=(
                engine.PhaseCall("REVERSE3", (0, 1, 2)),
                engine.PhaseInstruction("ROT", (0, 2)),
                engine.PhaseInstruction("ADD", (1, 2)),
            ),
            computational_class="sequence routing and relational composition",
        ),
    )


def all_inputs(program: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(
        itertools.product(
            range(program.radix),
            repeat=len(program.input_registers),
        )
    )


def ordered_cases() -> tuple[tuple[Any, tuple[int, ...]], ...]:
    return tuple(
        (program, values)
        for program in programs()
        for values in all_inputs(program)
    )
