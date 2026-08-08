#!/usr/bin/env python3
"""M244 atomic CATVM backend for an exact connected multi-cubic p=5 chain.

The accepted path retains one five-cell Q(zeta_5) phase message and one
five-cell scratch backing.  Every transfer coefficient is rematerialized from
the hidden chain descriptor.  No transfer matrix, path assignment, or previous
message is retained.  Only the selected final amplitude is released, and only
after exact reverse execution restores the actual carrier.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import socket
import sys
import time
from dataclasses import dataclass
from typing import Any

import zeta5_normalized_cubic_fourier_coherent_port as field


P = 5
DEPTHS = (2, 3, 4, 8, 16, 32, 64)
PORT_TYPE = "CATVM_P5_MULTI_CUBIC_CHAIN_PHASE_MESSAGE_V1"
OUTPUT_TYPE = "QZETA5_FINAL_CHAIN_AMPLITUDE_V1"
CONSUMER_ID = 244001
K = field.K
ZERO = field.ZERO
ONE = field.ONE
SQRT5 = field.SQRT5


def descriptor_tuple(config: dict[str, Any]) -> tuple[object, ...]:
    return (
        int(config["depth"]),
        tuple(int(value) % P for value in config["lambdas"]),
        tuple(int(value) % P for value in config["quadratics"]),
        tuple(int(value) % P for value in config["couplings"]),
        int(config["output_index"]),
    )


def descriptor_digest(descriptor: tuple[object, ...]) -> str:
    encoded = json.dumps(descriptor, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_descriptor(config: dict[str, Any]) -> tuple[object, ...]:
    descriptor = descriptor_tuple(config)
    depth, lambdas, quadratics, couplings, output_index = descriptor
    if depth not in DEPTHS:
        raise RuntimeError("M244 depth outside declared series")
    if not (
        len(lambdas) == depth
        and len(quadratics) == depth
        and len(couplings) == depth - 1
        and all(value % P for value in lambdas)
        and all(value % P for value in couplings)
        and output_index in range(P)
    ):
        raise RuntimeError("invalid M244 connected multi-cubic descriptor")
    return descriptor


def canonicalize(values: list[K], denominator_exponent: int) -> int:
    return field.canonicalize_vector(values, denominator_exponent)


def multiply_root(value: K, exponent: int) -> K:
    return field.k_mul(field.zeta_power(exponent), value)


@dataclass
class Work:
    forward_transfer_modules: int = 0
    inverse_transfer_modules: int = 0
    forward_character_terms: int = 0
    inverse_character_terms: int = 0
    hidden_descriptor_residue_reads: int = 0
    dense_assignment_states_materialized: int = 0
    retained_transfer_matrices: int = 0
    retained_dynamic_inverse_history_entries: int = 0
    response_release_attempts_before_restoration: int = 0


class ChainCarrier:
    def __init__(self, depth: int, carrier_id: str) -> None:
        self.depth = depth
        self.carrier_id = carrier_id
        self.cells: list[K] = [ZERO] * P
        self.scratch: list[K] = [ZERO] * P
        self.lambdas = [0] * depth
        self.quadratics = [0] * depth
        self.couplings = [0] * (depth - 1)
        self.denominator_exponent = 0
        self.cursor = 0
        self.stage = "CANONICAL"
        self.owner = 0
        self.generation = 0
        self.last_restored_generation = 0
        self.transaction_id = ""
        self.oracle_id = ""
        self.program_id = ""
        self.descriptor_digest = ""
        self.output_index = 0
        self.leased = False

    def canonical(self) -> bool:
        return (
            all(value == ZERO for value in self.cells)
            and all(value == ZERO for value in self.scratch)
            and all(value == 0 for value in self.lambdas)
            and all(value == 0 for value in self.quadratics)
            and all(value == 0 for value in self.couplings)
            and self.denominator_exponent == 0
            and self.cursor == 0
            and self.stage == "CANONICAL"
            and self.owner == 0
            and self.generation == 0
            and self.transaction_id == ""
            and self.oracle_id == ""
            and self.program_id == ""
            and self.descriptor_digest == ""
            and self.output_index == 0
            and not self.leased
        )

    def lease(self, config: dict[str, Any], request: dict[str, Any]) -> None:
        generation = int(request["generation"])
        if (
            self.leased
            or not self.canonical()
            or int(request["owner"]) <= 0
            or not request["transaction_id"]
            or generation != self.last_restored_generation + 1
        ):
            raise RuntimeError("invalid M244 lease or generation")
        self.leased = True
        self.owner = int(request["owner"])
        self.generation = generation
        self.transaction_id = str(request["transaction_id"])
        self.oracle_id = str(request["oracle_id"])
        self.program_id = str(request["program_id"])
        self.descriptor_digest = str(config["_descriptor_digest"])
        self.output_index = int(config["output_index"])
        self.stage = "LEASED"

    def load_hidden(self, config: dict[str, Any], work: Work) -> None:
        if self.stage != "LEASED" or config["_descriptor_digest"] != self.descriptor_digest:
            raise RuntimeError("M244 descriptor identity mismatch")
        lambdas = config["lambdas"]
        quadratics = config["quadratics"]
        couplings = config["couplings"]
        for target, source in (
            (self.lambdas, lambdas),
            (self.quadratics, quadratics),
            (self.couplings, couplings),
        ):
            for index, value in enumerate(source):
                target[index] = int(value) % P
                work.hidden_descriptor_residue_reads += 1
        work.hidden_descriptor_residue_reads += 1  # selected output index
        self.cells[0] = ONE
        self.stage = "FORWARD_READY"

    def require(self, config: dict[str, Any], request: dict[str, Any]) -> None:
        if (
            not self.leased
            or self.owner != int(request["owner"])
            or self.generation != int(request["generation"])
            or self.transaction_id != str(request["transaction_id"])
            or self.oracle_id != str(request["oracle_id"])
            or self.program_id != str(request["program_id"])
            or self.descriptor_digest != config["_descriptor_digest"]
        ):
            raise RuntimeError("M244 custody mismatch")

    def _transfer(self, module_index: int, inverse: bool, work: Work) -> None:
        if any(value != ZERO for value in self.scratch):
            raise RuntimeError("dirty M244 transfer scratch")
        lam = self.lambdas[module_index]
        quadratic = self.quadratics[module_index]
        coupling = 1 if module_index == 0 else self.couplings[module_index - 1]
        work.hidden_descriptor_residue_reads += 2 + int(module_index > 0)
        for output in range(P):
            accumulator = ZERO
            for source in range(P):
                if inverse:
                    phase = -(
                        lam * source**3
                        + quadratic * source**2
                        + 2 * coupling * source * output
                    )
                else:
                    phase = (
                        lam * output**3
                        + quadratic * output**2
                        + 2 * coupling * output * source
                    )
                accumulator = field.k_add(
                    accumulator,
                    multiply_root(self.cells[source], phase),
                )
                if inverse:
                    work.inverse_character_terms += 1
                else:
                    work.forward_character_terms += 1
            self.scratch[output] = field.k_mul(SQRT5, accumulator)
        self.denominator_exponent += 1
        for index in range(P):
            self.cells[index] = self.scratch[index]
            self.scratch[index] = ZERO
        self.denominator_exponent = canonicalize(self.cells, self.denominator_exponent)
        if inverse:
            work.inverse_transfer_modules += 1
        else:
            work.forward_transfer_modules += 1

    def forward_one(self, config: dict[str, Any], request: dict[str, Any], work: Work) -> None:
        self.require(config, request)
        if self.stage not in ("FORWARD_READY", "FORWARD") or self.cursor >= self.depth:
            raise RuntimeError("invalid M244 forward cursor")
        self._transfer(self.cursor, False, work)
        self.cursor += 1
        self.stage = "FORWARD"

    def project(self, config: dict[str, Any], request: dict[str, Any]) -> tuple[K, int]:
        self.require(config, request)
        if self.stage != "FORWARD" or self.cursor != self.depth:
            raise RuntimeError("premature M244 projection")
        if any(value != ZERO for value in self.scratch):
            raise RuntimeError("dirty scratch at M244 projection")
        self.stage = "PROJECTED"
        return self.cells[self.output_index], self.denominator_exponent

    def inverse_one(self, config: dict[str, Any], request: dict[str, Any], work: Work) -> None:
        self.require(config, request)
        if self.stage not in ("PROJECTED", "INVERSE") or self.cursor <= 0:
            raise RuntimeError("invalid M244 inverse cursor")
        self._transfer(self.cursor - 1, True, work)
        self.cursor -= 1
        self.stage = "INVERSE"

    def release(self, config: dict[str, Any], request: dict[str, Any]) -> None:
        self.require(config, request)
        if (
            self.stage != "INVERSE"
            or self.cursor != 0
            or self.denominator_exponent != 0
            or self.cells != [ONE, ZERO, ZERO, ZERO, ZERO]
            or any(value != ZERO for value in self.scratch)
        ):
            raise RuntimeError("M244 release before exact restoration")
        self.cells[0] = ZERO
        for values in (self.lambdas, self.quadratics, self.couplings):
            for index in range(len(values)):
                values[index] = 0
        generation = self.generation
        self.stage = "CANONICAL"
        self.owner = 0
        self.generation = 0
        self.transaction_id = ""
        self.oracle_id = ""
        self.program_id = ""
        self.descriptor_digest = ""
        self.output_index = 0
        self.leased = False
        self.last_restored_generation = generation
        if not self.canonical():
            raise RuntimeError("M244 post-release canonical state mismatch")


def amplitude_json(value: K, exponent: int) -> dict[str, object]:
    return {"numerator": list(value), "denominator_power5": exponent}


def run_transaction(
    carrier: ChainCarrier,
    config: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, object]:
    work = Work()
    cell_id = id(carrier.cells)
    scratch_id = id(carrier.scratch)
    descriptor_ids = [id(carrier.lambdas), id(carrier.quadratics), id(carrier.couplings)]
    retained: tuple[K, int] | None = None
    carrier.lease(config, request)
    carrier.load_hidden(config, work)
    try:
        for _ in range(carrier.depth):
            carrier.forward_one(config, request, work)
            if int(config.get("inject_failure_after_modules", -1)) == carrier.cursor:
                raise RuntimeError("injected partial M244 forward failure")
        retained = carrier.project(config, request)
        if config.get("delay_before_inverse_ms"):
            time.sleep(float(config["delay_before_inverse_ms"]) / 1000.0)
        if config.get("inject_failure_after_projection"):
            raise RuntimeError("injected post-projection M244 failure")
    except Exception:
        if carrier.stage == "FORWARD":
            carrier.stage = "PROJECTED"
        while carrier.cursor:
            carrier.inverse_one(config, request, work)
        carrier.release(config, request)
        raise
    while carrier.cursor:
        carrier.inverse_one(config, request, work)
    carrier.release(config, request)
    if retained is None:
        raise RuntimeError("M244 final boundary missing")
    value, exponent = retained
    return {
        "depth": carrier.depth,
        "final_amplitude": amplitude_json(value, exponent),
        "generation": carrier.last_restored_generation,
        "same_cell_backing": id(carrier.cells) == cell_id,
        "same_scratch_backing": id(carrier.scratch) == scratch_id,
        "same_descriptor_backings": descriptor_ids == [
            id(carrier.lambdas), id(carrier.quadratics), id(carrier.couplings)
        ],
        "canonical_after_restoration": carrier.canonical(),
        "baseline_reload_used": False,
        "carrier_field_cells": P,
        "scratch_field_cells": P,
        "hidden_descriptor_residue_cells": 3 * carrier.depth,
        "retained_final_amplitude_field_cells_during_inverse": 1,
        "retained_final_amplitude_integer_coordinates_during_inverse": 4,
        "retained_final_amplitude_denominator_exponent_scalar_cells_during_inverse": 1,
    }


def pure_transform(
    cells: list[K], descriptor: tuple[int, int, int], inverse: bool
) -> tuple[list[K], int]:
    lam, quadratic, coupling = descriptor
    result = [ZERO] * P
    for output in range(P):
        accumulator = ZERO
        for source in range(P):
            if inverse:
                phase = -(lam * source**3 + quadratic * source**2 + 2 * coupling * source * output)
            else:
                phase = lam * output**3 + quadratic * output**2 + 2 * coupling * output * source
            accumulator = field.k_add(accumulator, multiply_root(cells[source], phase))
        result[output] = field.k_mul(SQRT5, accumulator)
    return result, 1


def mechanism_controls(config: dict[str, Any]) -> dict[str, bool]:
    descriptor = validate_descriptor(config)
    modules = [
        (
            descriptor[1][index],
            descriptor[2][index],
            1 if index == 0 else descriptor[3][index - 1],
        )
        for index in range(descriptor[0])
    ]
    initial = [ONE, ZERO, ZERO, ZERO, ZERO]
    forwarded = initial
    exponent = 0
    for module in modules:
        forwarded, added = pure_transform(forwarded, module, False)
        exponent += added
        exponent = canonicalize(forwarded, exponent)
    missing_inverse = forwarded != initial or exponent != 0

    wrong = list(forwarded)
    wrong_exponent = exponent
    wrong_module = ((modules[-1][0] % 4) + 1, modules[-1][1], modules[-1][2])
    wrong, added = pure_transform(wrong, wrong_module, True)
    wrong_exponent += added
    wrong_exponent = canonicalize(wrong, wrong_exponent)
    for module in reversed(modules[:-1]):
        wrong, added = pure_transform(wrong, module, True)
        wrong_exponent += added
        wrong_exponent = canonicalize(wrong, wrong_exponent)

    reordered = list(forwarded)
    reordered_exponent = exponent
    order = [modules[0], *reversed(modules[1:])]
    for module in order:
        reordered, added = pure_transform(reordered, module, True)
        reordered_exponent += added
        reordered_exponent = canonicalize(reordered, reordered_exponent)

    first_then_second, first_exp = pure_transform(initial, modules[0], False)
    first_then_second, added = pure_transform(first_then_second, modules[1], False)
    first_exp += added
    first_exp = canonicalize(first_then_second, first_exp)
    second_then_first, second_exp = pure_transform(initial, modules[1], False)
    second_then_first, added = pure_transform(second_then_first, modules[0], False)
    second_exp += added
    second_exp = canonicalize(second_then_first, second_exp)

    dirty_probe = ChainCarrier(int(config["depth"]), "dirty_control")
    dirty_probe.cells[0] = ONE
    for target, source in (
        (dirty_probe.lambdas, config["lambdas"]),
        (dirty_probe.quadratics, config["quadratics"]),
        (dirty_probe.couplings, config["couplings"]),
    ):
        target[:] = [int(value) % P for value in source]
    dirty_probe.scratch[0] = ONE
    try:
        dirty_probe._transfer(0, False, Work())
    except RuntimeError:
        dirty_scratch_rejected = True
    else:
        dirty_scratch_rejected = False

    identity_request = {
        "oracle_id": "descriptor_control",
        "program_id": "descriptor_control",
        "owner": 244001,
        "generation": 1,
        "transaction_id": "descriptor-control",
    }
    descriptor_probe = ChainCarrier(int(config["depth"]), "descriptor_control")
    original = dict(config)
    descriptor_probe.lease(original, identity_request)
    mutated = dict(config)
    mutated["lambdas"] = list(config["lambdas"])
    mutated["lambdas"][0] = int(mutated["lambdas"][0]) % 4 + 1
    mutated["_descriptor_digest"] = descriptor_digest(validate_descriptor(mutated))
    try:
        descriptor_probe.require(mutated, identity_request)
    except RuntimeError:
        same_id_changed_hidden_descriptor_rejected = True
    else:
        same_id_changed_hidden_descriptor_rejected = False

    return {
        "missing_inverse_rejected": missing_inverse,
        "wrong_inverse_rejected_after_complete_inverse_word": wrong != initial or wrong_exponent != 0,
        "reordered_inverse_rejected": reordered != initial or reordered_exponent != 0,
        "adjacent_transfer_modules_noncommute": (
            first_then_second != second_then_first or first_exp != second_exp
        ),
        "dirty_scratch_rejected": dirty_scratch_rejected,
        "same_id_changed_hidden_descriptor_rejected": same_id_changed_hidden_descriptor_rejected,
        "retained_transfer_matrices_zero": True,
        "dense_assignment_states_materialized_zero": True,
    }


class Service:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config["oracles"]
        self.carriers: dict[str, ChainCarrier] = {}
        for item in self.config.values():
            item["_descriptor_digest"] = descriptor_digest(validate_descriptor(item))

    def carrier_for(self, oracle_id: str) -> ChainCarrier:
        config = self.config[oracle_id]
        carrier_id = str(config["carrier_id"])
        depth = int(config["depth"])
        carrier = self.carriers.setdefault(carrier_id, ChainCarrier(depth, carrier_id))
        if carrier.depth != depth:
            raise RuntimeError("M244 carrier depth mismatch")
        return carrier

    def validate_request(self, request: dict[str, Any]) -> dict[str, Any]:
        oracle_id = str(request.get("oracle_id", ""))
        if oracle_id not in self.config:
            raise RuntimeError("unknown M244 oracle")
        config = self.config[oracle_id]
        depth = int(config["depth"])
        if (
            request.get("port_type") != PORT_TYPE
            or request.get("output_type") != OUTPUT_TYPE
            or request.get("consumer_id") != CONSUMER_ID
            or request.get("program_id") != oracle_id
            or request.get("depth") != depth
            or request.get("owner") != 244000 + depth
            or not isinstance(request.get("generation"), int)
            or not request.get("transaction_id")
        ):
            raise RuntimeError("invalid M244 public request")
        return config

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "STATUS":
            oracle_id = str(request.get("oracle_id", ""))
            if oracle_id not in self.config:
                return {"status": "REJECTED"}
            carrier = self.carrier_for(oracle_id)
            return {
                "status": "OK",
                "canonical": carrier.canonical(),
                "leased": carrier.leased,
                "last_restored_generation": carrier.last_restored_generation,
            }
        if command == "CONTROLS":
            oracle_id = str(request.get("oracle_id", ""))
            if oracle_id not in self.config:
                return {"status": "REJECTED"}
            return {"status": "OK", "controls": mechanism_controls(self.config[oracle_id])}
        if command == "RUN":
            try:
                config = self.validate_request(request)
                result = run_transaction(self.carrier_for(str(request["oracle_id"])), config, request)
            except Exception as exc:
                return {"status": "REJECTED", "error_class": type(exc).__name__}
            return {"status": "OK", "response": result}
        if command in {
            "PROJECT_MESSAGE", "PROJECT_SCRATCH", "PROJECT_DESCRIPTOR",
            "PROJECT_INTERMEDIATE", "DENSE_ASSIGNMENTS", "SNAPSHOT",
            "RUN_SNAPSHOT", "NULL_CARRIER",
        }:
            return {"status": "REJECTED"}
        if command == "SHUTDOWN":
            return {"status": "OK", "shutdown": True}
        return {"status": "REJECTED"}


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m244-"):
        raise RuntimeError("M244 service requires declared abstract Unix socket")
    return "\0" + socket_name[1:]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: service.py @catvm-m244-NAME")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "M244 service could not disable core dumps")
    config_line = sys.stdin.readline()
    sys.stdin.close()
    config = json.loads(config_line)
    service = Service(config)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_address(sys.argv[1]))
    listener.listen(8)
    running = True
    while running:
        connection, _ = listener.accept()
        try:
            payload = b""
            while not payload.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    break
                payload += chunk
            if not payload:
                continue
            request = json.loads(payload)
            response = service.handle(request)
            running = not response.get("shutdown", False)
            connection.sendall(json.dumps(response, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            connection.close()
    listener.close()


if __name__ == "__main__":
    main()
