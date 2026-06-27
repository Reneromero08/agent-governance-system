from __future__ import annotations

import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
HISTORICAL = (
    HERE.parent.parent
    / "10_cross_core_wormhole"
    / "slot2_pdn"
    / "slot2_pdn_lockin.c"
)
V2 = HERE / "combined_pdn_hardware.c"


def function_source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    marker = "static double alu_burst(uint64_t *iseed) {"
    start = source.index(marker)
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1].replace("\r\n", "\n")
    raise AssertionError(f"unterminated alu_burst in {path}")


class Slot2PrimitiveIdentityTests(unittest.TestCase):
    def test_v2_primitive_is_verbatim_historical_slot2(self) -> None:
        self.assertEqual(function_source(V2), function_source(HISTORICAL))


if __name__ == "__main__":
    unittest.main()
