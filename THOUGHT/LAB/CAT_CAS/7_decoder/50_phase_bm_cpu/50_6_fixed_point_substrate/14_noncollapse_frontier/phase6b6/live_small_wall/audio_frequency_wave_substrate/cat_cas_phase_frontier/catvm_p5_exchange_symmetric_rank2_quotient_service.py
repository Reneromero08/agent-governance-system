#!/usr/bin/env python3
"""M245 CATVM backend for an exact exchange-symmetric p=5 rank-two port.

The accepted carrier stores one amplitude for each unordered pair in F5^2.
The 15-cell quotient is exact only for the declared rail-exchange-symmetric
family.  It is not a general compression of labelled 25-state interfaces.
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
ORBITS = tuple((left, right) for left in range(P) for right in range(left, P))
ORBIT_INDEX = {pair: index for index, pair in enumerate(ORBITS)}
WIDTH = len(ORBITS)
PORT_TYPE = "CATVM_P5_EXCHANGE_SYMMETRIC_RANK2_ORBIT_PORT_V1"
OUTPUT_TYPE = "QZETA5_FINAL_UNORDERED_PAIR_AMPLITUDE_V1"
CONSUMER_ID = 245001
K = field.K
ZERO = field.ZERO
ONE = field.ONE


def ordered_orbit(pair: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    left, right = pair
    return (pair,) if left == right else (pair, (right, left))


def descriptor_tuple(config: dict[str, Any]) -> tuple[object, ...]:
    return (
        int(config["depth"]),
        tuple(int(value) % P for value in config["lambdas"]),
        tuple(int(value) % P for value in config["quadratics"]),
        tuple(int(value) % P for value in config["rungs"]),
        tuple(int(value) % P for value in config["couplings"]),
        tuple(sorted(int(value) % P for value in config["output_orbit"])),
    )


def descriptor_digest(descriptor: tuple[object, ...]) -> str:
    encoded = json.dumps(descriptor, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_descriptor(config: dict[str, Any]) -> tuple[object, ...]:
    descriptor = descriptor_tuple(config)
    depth, lambdas, quadratics, rungs, couplings, output_orbit = descriptor
    if depth != 4:
        raise RuntimeError("M245 accepts only the declared four-module diagnostic")
    if not (
        len(lambdas) == depth
        and len(quadratics) == depth
        and len(rungs) == depth
        and len(couplings) == depth - 1
        and all(value for value in lambdas)
        and all(value for value in rungs)
        and all(value for value in couplings)
        and output_orbit in ORBIT_INDEX
    ):
        raise RuntimeError("invalid M245 symmetric rank-two descriptor")
    if "left_lambdas" in config or "right_lambdas" in config:
        raise RuntimeError("exchange-breaking descriptor rejected by 15-cell port")
    return descriptor


def multiply_root(value: K, exponent: int) -> K:
    return field.k_mul(field.zeta_power(exponent), value)


def phase(parameters: tuple[int, int, int], pair: tuple[int, int]) -> int:
    lam, quadratic, rung = parameters
    left, right = pair
    return (
        lam * (left**3 + right**3)
        + quadratic * (left**2 + right**2)
        + 2 * rung * left * right
    )


@dataclass
class Work:
    forward_modules: int = 0
    inverse_modules: int = 0
    forward_character_terms: int = 0
    inverse_character_terms: int = 0
    hidden_descriptor_residue_reads: int = 0
    dense25_state_materializations: int = 0
    transfer_matrices_materialized: int = 0
    retained_dynamic_inverse_history_entries: int = 0


class OrbitCarrier:
    def __init__(self, carrier_id: str) -> None:
        self.carrier_id = carrier_id
        self.cells: list[K] = [ZERO] * WIDTH
        self.scratch: list[K] = [ZERO] * WIDTH
        self.lambdas = [0] * 4
        self.quadratics = [0] * 4
        self.rungs = [0] * 4
        self.couplings = [0] * 3
        self.output_orbit = (0, 0)
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
        self.leased = False

    def canonical(self) -> bool:
        return (
            self.cells == [ZERO] * WIDTH
            and self.scratch == [ZERO] * WIDTH
            and self.lambdas == [0] * 4
            and self.quadratics == [0] * 4
            and self.rungs == [0] * 4
            and self.couplings == [0] * 3
            and self.output_orbit == (0, 0)
            and self.denominator_exponent == 0
            and self.cursor == 0
            and self.stage == "CANONICAL"
            and self.owner == 0
            and self.generation == 0
            and self.transaction_id == ""
            and self.oracle_id == ""
            and self.program_id == ""
            and self.descriptor_digest == ""
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
            raise RuntimeError("invalid M245 lease or generation")
        self.leased = True
        self.owner = int(request["owner"])
        self.generation = generation
        self.transaction_id = str(request["transaction_id"])
        self.oracle_id = str(request["oracle_id"])
        self.program_id = str(request["program_id"])
        self.descriptor_digest = str(config["_descriptor_digest"])
        self.stage = "LEASED"

    def load_hidden(self, config: dict[str, Any], work: Work) -> None:
        if self.stage != "LEASED" or config["_descriptor_digest"] != self.descriptor_digest:
            raise RuntimeError("M245 descriptor identity mismatch")
        for target, name in (
            (self.lambdas, "lambdas"),
            (self.quadratics, "quadratics"),
            (self.rungs, "rungs"),
            (self.couplings, "couplings"),
        ):
            source = config[name]
            for index, value in enumerate(source):
                target[index] = int(value) % P
                work.hidden_descriptor_residue_reads += 1
        self.output_orbit = tuple(sorted(int(value) % P for value in config["output_orbit"]))
        work.hidden_descriptor_residue_reads += 2
        self.cells[ORBIT_INDEX[(0, 0)]] = ONE
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
            raise RuntimeError("M245 custody mismatch")

    def _module_parameters(self, module_index: int) -> tuple[tuple[int, int, int], int]:
        parameters = (
            self.lambdas[module_index],
            self.quadratics[module_index],
            self.rungs[module_index],
        )
        coupling = 1 if module_index == 0 else self.couplings[module_index - 1]
        return parameters, coupling

    def _forward(self, module_index: int, work: Work) -> None:
        if self.scratch != [ZERO] * WIDTH:
            raise RuntimeError("dirty M245 forward scratch")
        parameters, coupling = self._module_parameters(module_index)
        work.hidden_descriptor_residue_reads += 3 + int(module_index > 0)
        if module_index == 0:
            if self.cells != [ONE] + [ZERO] * (WIDTH - 1):
                raise RuntimeError("M245 first-module specialization requires canonical input")
            for output_index, output_pair in enumerate(ORBITS):
                self.scratch[output_index] = field.zeta_power(phase(parameters, output_pair))
                work.forward_character_terms += 1
        else:
            for output_index, output_pair in enumerate(ORBITS):
                accumulator = ZERO
                output_phase = phase(parameters, output_pair)
                for input_index, input_pair in enumerate(ORBITS):
                    for ordered_input in ordered_orbit(input_pair):
                        character = output_phase + 2 * coupling * (
                            output_pair[0] * ordered_input[0]
                            + output_pair[1] * ordered_input[1]
                        )
                        accumulator = field.k_add(
                            accumulator,
                            multiply_root(self.cells[input_index], character),
                        )
                        work.forward_character_terms += 1
                self.scratch[output_index] = accumulator
        self.denominator_exponent += 1
        self.denominator_exponent = field.canonicalize_vector(
            self.scratch, self.denominator_exponent
        )
        for index in range(WIDTH):
            self.cells[index] = self.scratch[index]
            self.scratch[index] = ZERO
        work.forward_modules += 1

    def _inverse(self, module_index: int, work: Work) -> None:
        if self.scratch != [ZERO] * WIDTH:
            raise RuntimeError("dirty M245 inverse scratch")
        parameters, coupling = self._module_parameters(module_index)
        work.hidden_descriptor_residue_reads += 3 + int(module_index > 0)
        for input_index, input_pair in enumerate(ORBITS):
            accumulator = ZERO
            for output_index, output_pair in enumerate(ORBITS):
                for ordered_output in ordered_orbit(output_pair):
                    character = -phase(parameters, ordered_output) - 2 * coupling * (
                        ordered_output[0] * input_pair[0]
                        + ordered_output[1] * input_pair[1]
                    )
                    accumulator = field.k_add(
                        accumulator,
                        multiply_root(self.cells[output_index], character),
                    )
                    work.inverse_character_terms += 1
            self.scratch[input_index] = accumulator
        self.denominator_exponent += 1
        self.denominator_exponent = field.canonicalize_vector(
            self.scratch, self.denominator_exponent
        )
        for index in range(WIDTH):
            self.cells[index] = self.scratch[index]
            self.scratch[index] = ZERO
        work.inverse_modules += 1

    def forward_one(self, config: dict[str, Any], request: dict[str, Any], work: Work) -> None:
        self.require(config, request)
        if self.stage not in ("FORWARD_READY", "FORWARD") or self.cursor >= 4:
            raise RuntimeError("invalid M245 forward cursor")
        self._forward(self.cursor, work)
        self.cursor += 1
        self.stage = "FORWARD"

    def project(self, config: dict[str, Any], request: dict[str, Any]) -> tuple[K, int]:
        self.require(config, request)
        if self.stage != "FORWARD" or self.cursor != 4 or self.scratch != [ZERO] * WIDTH:
            raise RuntimeError("premature or dirty M245 projection")
        self.stage = "PROJECTED"
        return self.cells[ORBIT_INDEX[self.output_orbit]], self.denominator_exponent

    def inverse_one(self, config: dict[str, Any], request: dict[str, Any], work: Work) -> None:
        self.require(config, request)
        if self.stage not in ("PROJECTED", "INVERSE") or self.cursor <= 0:
            raise RuntimeError("invalid M245 inverse cursor")
        self._inverse(self.cursor - 1, work)
        self.cursor -= 1
        self.stage = "INVERSE"

    def release(self, config: dict[str, Any], request: dict[str, Any]) -> None:
        self.require(config, request)
        if (
            self.stage != "INVERSE"
            or self.cursor != 0
            or self.denominator_exponent != 0
            or self.cells != [ONE] + [ZERO] * (WIDTH - 1)
            or self.scratch != [ZERO] * WIDTH
        ):
            raise RuntimeError("M245 release before exact restoration")
        self.cells[0] = ZERO
        for values in (self.lambdas, self.quadratics, self.rungs, self.couplings):
            for index in range(len(values)):
                values[index] = 0
        self.output_orbit = (0, 0)
        generation = self.generation
        self.denominator_exponent = 0
        self.cursor = 0
        self.stage = "CANONICAL"
        self.owner = 0
        self.generation = 0
        self.transaction_id = ""
        self.oracle_id = ""
        self.program_id = ""
        self.descriptor_digest = ""
        self.leased = False
        self.last_restored_generation = generation
        if not self.canonical():
            raise RuntimeError("M245 post-release canonical state mismatch")


def amplitude_json(value: K, exponent: int) -> dict[str, object]:
    return {"numerator": list(value), "denominator_power5": exponent}


def run_transaction(
    carrier: OrbitCarrier, config: dict[str, Any], request: dict[str, Any]
) -> dict[str, object]:
    work = Work()
    cell_id = id(carrier.cells)
    scratch_id = id(carrier.scratch)
    descriptor_ids = [
        id(carrier.lambdas), id(carrier.quadratics),
        id(carrier.rungs), id(carrier.couplings),
    ]
    retained: tuple[K, int] | None = None
    carrier.lease(config, request)
    carrier.load_hidden(config, work)
    try:
        for _ in range(4):
            carrier.forward_one(config, request, work)
            if int(config.get("inject_failure_after_modules", -1)) == carrier.cursor:
                raise RuntimeError("injected partial M245 forward failure")
        retained = carrier.project(config, request)
        if config.get("delay_before_inverse_ms"):
            time.sleep(float(config["delay_before_inverse_ms"]) / 1000.0)
        if config.get("inject_failure_after_projection"):
            raise RuntimeError("injected post-projection M245 failure")
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
        raise RuntimeError("M245 final boundary missing")
    value, exponent = retained
    return {
        "final_amplitude": amplitude_json(value, exponent),
        "generation": carrier.last_restored_generation,
        "same_cell_backing": id(carrier.cells) == cell_id,
        "same_scratch_backing": id(carrier.scratch) == scratch_id,
        "same_descriptor_backings": descriptor_ids == [
            id(carrier.lambdas), id(carrier.quadratics),
            id(carrier.rungs), id(carrier.couplings),
        ],
        "canonical_after_restoration": carrier.canonical(),
        "baseline_reload_used": False,
        "carrier_field_cells": WIDTH,
        "scratch_field_cells": WIDTH,
        "hidden_descriptor_residue_cells": 17,
        "retained_final_amplitude_field_cells_during_inverse": 1,
        "forward_character_terms": work.forward_character_terms,
        "inverse_character_terms": work.inverse_character_terms,
        "accepted_descriptor_reads": work.hidden_descriptor_residue_reads,
        "dense25_state_materializations": work.dense25_state_materializations,
        "transfer_matrices_materialized": work.transfer_matrices_materialized,
        "retained_dynamic_inverse_history_entries": work.retained_dynamic_inverse_history_entries,
    }


def pure_transform(
    cells: list[K], parameters: tuple[int, int, int], coupling: int, inverse: bool,
    wrong_multiplicity: bool = False,
) -> tuple[list[K], int]:
    result = [ZERO] * WIDTH
    if inverse:
        for input_index, input_pair in enumerate(ORBITS):
            accumulator = ZERO
            for output_index, output_pair in enumerate(ORBITS):
                outputs = (output_pair,) if wrong_multiplicity else ordered_orbit(output_pair)
                for ordered_output in outputs:
                    character = -phase(parameters, ordered_output) - 2 * coupling * (
                        ordered_output[0] * input_pair[0]
                        + ordered_output[1] * input_pair[1]
                    )
                    accumulator = field.k_add(
                        accumulator, multiply_root(cells[output_index], character)
                    )
            result[input_index] = accumulator
    else:
        for output_index, output_pair in enumerate(ORBITS):
            accumulator = ZERO
            output_phase = phase(parameters, output_pair)
            for input_index, input_pair in enumerate(ORBITS):
                inputs = (input_pair,) if wrong_multiplicity else ordered_orbit(input_pair)
                for ordered_input in inputs:
                    character = output_phase + 2 * coupling * (
                        output_pair[0] * ordered_input[0]
                        + output_pair[1] * ordered_input[1]
                    )
                    accumulator = field.k_add(
                        accumulator, multiply_root(cells[input_index], character)
                    )
            result[output_index] = accumulator
    return result, 1


def mechanism_controls(config: dict[str, Any]) -> dict[str, bool]:
    descriptor = validate_descriptor(config)
    modules = [
        (
            (descriptor[1][index], descriptor[2][index], descriptor[3][index]),
            1 if index == 0 else descriptor[4][index - 1],
        )
        for index in range(4)
    ]
    initial = [ONE] + [ZERO] * (WIDTH - 1)
    forwarded = initial
    exponent = 0
    for parameters, coupling in modules:
        forwarded, added = pure_transform(forwarded, parameters, coupling, False)
        exponent += added
        exponent = field.canonicalize_vector(forwarded, exponent)
    missing_inverse = forwarded != initial or exponent != 0

    wrong = list(forwarded)
    wrong_exponent = exponent
    wrong_parameters = ((modules[-1][0][0] % 4) + 1, *modules[-1][0][1:])
    wrong, added = pure_transform(wrong, wrong_parameters, modules[-1][1], True)
    wrong_exponent += added
    wrong_exponent = field.canonicalize_vector(wrong, wrong_exponent)
    for parameters, coupling in reversed(modules[:-1]):
        wrong, added = pure_transform(wrong, parameters, coupling, True)
        wrong_exponent += added
        wrong_exponent = field.canonicalize_vector(wrong, wrong_exponent)

    reordered = list(forwarded)
    reordered_exponent = exponent
    for parameters, coupling in (modules[-2], modules[-1], *reversed(modules[:-2])):
        reordered, added = pure_transform(reordered, parameters, coupling, True)
        reordered_exponent += added
        reordered_exponent = field.canonicalize_vector(reordered, reordered_exponent)

    correct = initial
    correct_exp = 0
    wrong_mult = initial
    wrong_mult_exp = 0
    for index, (parameters, coupling) in enumerate(modules):
        correct, added = pure_transform(correct, parameters, coupling, False)
        correct_exp += added
        correct_exp = field.canonicalize_vector(correct, correct_exp)
        wrong_mult, added = pure_transform(
            wrong_mult, parameters, coupling, False, wrong_multiplicity=index == 1
        )
        wrong_mult_exp += added
        wrong_mult_exp = field.canonicalize_vector(wrong_mult, wrong_mult_exp)

    descriptor_probe = OrbitCarrier("descriptor_control")
    request = {
        "oracle_id": "descriptor_control", "program_id": "descriptor_control",
        "owner": 245004, "generation": 1, "transaction_id": "descriptor-control",
    }
    descriptor_probe.lease(config, request)
    mutated = dict(config)
    mutated["rungs"] = list(config["rungs"])
    mutated["rungs"][0] = int(mutated["rungs"][0]) % 4 + 1
    mutated["_descriptor_digest"] = descriptor_digest(validate_descriptor(mutated))
    try:
        descriptor_probe.require(mutated, request)
    except RuntimeError:
        same_id_changed_descriptor_rejected = True
    else:
        same_id_changed_descriptor_rejected = False

    dirty_probe = OrbitCarrier("dirty")
    dirty_probe.cells[0] = ONE
    dirty_probe.lambdas[:] = list(descriptor[1])
    dirty_probe.quadratics[:] = list(descriptor[2])
    dirty_probe.rungs[:] = list(descriptor[3])
    dirty_probe.couplings[:] = list(descriptor[4])
    dirty_probe.scratch[0] = ONE
    try:
        dirty_probe._forward(0, Work())
    except RuntimeError:
        dirty_scratch_rejected = True
    else:
        dirty_scratch_rejected = False

    try:
        broken = dict(config)
        broken["left_lambdas"] = list(config["lambdas"])
        broken["right_lambdas"] = list(config["lambdas"])
        broken["right_lambdas"][0] = int(broken["right_lambdas"][0]) % 4 + 1
        validate_descriptor(broken)
    except RuntimeError:
        exchange_breaking_descriptor_rejected = True
    else:
        exchange_breaking_descriptor_rejected = False

    try:
        disconnected = dict(config)
        disconnected["rungs"] = list(config["rungs"])
        disconnected["rungs"][0] = 0
        validate_descriptor(disconnected)
    except RuntimeError:
        zero_rung_rejected = True
    else:
        zero_rung_rejected = False

    return {
        "missing_inverse_rejected": missing_inverse,
        "wrong_inverse_rejected_after_complete_inverse_word": wrong != initial or wrong_exponent != 0,
        "reordered_inverse_rejected": reordered != initial or reordered_exponent != 0,
        "wrong_off_diagonal_orbit_multiplicity_changes_state": (
            wrong_mult != correct or wrong_mult_exp != correct_exp
        ),
        "same_id_changed_descriptor_rejected": same_id_changed_descriptor_rejected,
        "dirty_scratch_rejected": dirty_scratch_rejected,
        "exchange_breaking_descriptor_rejected": exchange_breaking_descriptor_rejected,
        "zero_rung_disconnected_descriptor_rejected": zero_rung_rejected,
        "retained_transfer_matrices_zero": True,
        "dense25_state_materializations_zero": True,
    }


class Service:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config["oracles"]
        self.carriers: dict[str, OrbitCarrier] = {}
        for item in self.config.values():
            item["_descriptor_digest"] = descriptor_digest(validate_descriptor(item))

    def carrier_for(self, oracle_id: str) -> OrbitCarrier:
        config = self.config[oracle_id]
        carrier_id = str(config["carrier_id"])
        return self.carriers.setdefault(carrier_id, OrbitCarrier(carrier_id))

    def validate_request(self, request: dict[str, Any]) -> dict[str, Any]:
        oracle_id = str(request.get("oracle_id", ""))
        if oracle_id not in self.config:
            raise RuntimeError("unknown M245 oracle")
        config = self.config[oracle_id]
        if (
            request.get("port_type") != PORT_TYPE
            or request.get("output_type") != OUTPUT_TYPE
            or request.get("consumer_id") != CONSUMER_ID
            or request.get("program_id") != oracle_id
            or request.get("depth") != 4
            or request.get("owner") != 245004
            or not isinstance(request.get("generation"), int)
            or not request.get("transaction_id")
        ):
            raise RuntimeError("invalid M245 public request")
        return config

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "STATUS":
            oracle_id = str(request.get("oracle_id", ""))
            if oracle_id not in self.config:
                return {"status": "REJECTED"}
            carrier = self.carrier_for(oracle_id)
            return {
                "status": "OK", "canonical": carrier.canonical(),
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
                response = run_transaction(self.carrier_for(str(request["oracle_id"])), config, request)
            except Exception as exc:
                return {"status": "REJECTED", "error_class": type(exc).__name__}
            return {"status": "OK", "response": response}
        if command in {
            "PROJECT_ORBIT_MESSAGE", "PROJECT_LABELLED_MESSAGE", "PROJECT_SCRATCH",
            "PROJECT_DESCRIPTOR", "PROJECT_INTERMEDIATE", "DENSE_PATHS", "SNAPSHOT",
            "RUN_SNAPSHOT", "NULL_CARRIER",
        }:
            return {"status": "REJECTED"}
        if command == "SHUTDOWN":
            return {"status": "OK", "shutdown": True}
        return {"status": "REJECTED"}


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m245-"):
        raise RuntimeError("M245 service requires declared abstract Unix socket")
    return "\0" + socket_name[1:]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: service.py @catvm-m245-NAME")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "M245 service could not disable core dumps")
    config_line = sys.stdin.readline()
    sys.stdin.close()
    service = Service(json.loads(config_line))
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
            response = service.handle(json.loads(payload))
            running = not response.get("shutdown", False)
            connection.sendall(
                json.dumps(response, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            )
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            connection.close()
    listener.close()


if __name__ == "__main__":
    main()
