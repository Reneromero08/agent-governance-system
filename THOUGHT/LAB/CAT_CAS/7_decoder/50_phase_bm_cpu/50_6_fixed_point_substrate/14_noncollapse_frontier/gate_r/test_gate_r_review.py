#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import gate_r_review


class GateRReviewTests(unittest.TestCase):
    def write_json(self, path: Path, value: object) -> None:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fixture_design(self) -> dict[str, object]:
        states = [
            {"model_id": "S0_minimal", "fields": "lockin_I;lockin_Q;ring_osc_period"},
            {"model_id": "S1_contextual", "fields": "lockin_I;lockin_Q;ring_osc_period;sender_mode;route;capture_window"},
            {"model_id": "S2_delay_embedded", "fields": "y_history;u_history;route_context"},
        ]
        return {
            "design_id": "l4b5b0_observability_operator_v1",
            "design_version": "1.0.0",
            "design_digest": "0123456789abcdef",
            "status": "READY_FOR_HUMAN_REVIEW",
            "human_reviewed": False,
            "implementation_authorized": False,
            "executed": False,
            "full_physical_observability_claimed": False,
            "physical_restoration_claimed": False,
            "claim_level": 1,
            "design_scope": {"state_space_model": "x(t+1)=F(x(t),u(t));y(t)=H(x(t))"},
            "state_models": states,
            "input_families": [{"input_id": f"I{i}"} for i in range(11)],
            "operator_candidates": [{"operator_id": f"O{i}"} for i in range(4)],
            "acceptance_gates": [{"gate_id": f"G{i}"} for i in range(1, 11)],
            "falsification_conditions": [{"condition_id": f"F{i}"} for i in range(1, 11)],
            "artifact_contracts": [{"artifact_id": f"A{i}"} for i in range(6)],
        }

    def test_generate_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            design = root / "design.json"
            addendum = root / "addendum.md"
            review_md = root / "review.md"
            findings = root / "findings.md"
            phase5c = root / "phase5c.json"
            phase5d = root / "phase5d.json"
            tone = root / "tone.md"
            output = root / "output"

            self.write_json(design, self.fixture_design())
            addendum.write_text(
                "BINDING_REPAIR_ADDENDUM_PENDING_PROJECT_OWNER_RATIFICATION\n"
                "S1_contextual = gauge_normalize\nPERSISTENT_STATE_CANDIDATE\n"
                "DRIVEN_RELATIONAL_TRANSPORT_ONLY\nGR5 governance separation\n",
                encoding="utf-8",
            )
            review_md.write_text(
                "TECHNICAL_ACCEPT_WITH_REQUIRED_REPAIRS_APPLIED\n"
                "Project-owner ratification: pending\n"
                "Physical acquisition authorization: FALSE\n",
                encoding="utf-8",
            )
            findings.write_text(
                "F1 — State partition\nF3 — Driven transport versus persistence\n"
                "F4 — Tone/path confounding\nF5 — Session gauge\n",
                encoding="utf-8",
            )
            self.write_json(phase5c, {"generated_utc": "2026-06-19T20:13:06+00:00", "campaign_id": "campaign", "decision": {"primary_outcome": "TRANSFER_EQUIVARIANCE_SUPPORTED"}})
            self.write_json(phase5d, {"generated_utc_inherited_from_phase6b5c": "2026-06-19T20:13:06+00:00", "decision": {"gate_r_ready": True}})
            tone.write_text("PREREGISTERED_NOT_AUTHORIZED\nFWD\nREV\nRND1\nRND2\norder-label sham\n", encoding="utf-8")

            manifest = gate_r_review.generate(design, addendum, review_md, findings, phase5c, phase5d, tone, output)
            self.assertTrue(manifest["decision"]["technical_review_complete"])
            self.assertFalse(manifest["decision"]["implementation_authorized"])
            review = gate_r_review.load_json(output / "gate_r_technical_review.json")
            self.assertFalse(review["human_review"])
            self.assertTrue(review["project_owner_ratification_required"])
            self.assertEqual(review["verdict"], "TECHNICAL_ACCEPT_WITH_REQUIRED_REPAIRS_APPLIED")
            verified = gate_r_review.verify(output)
            self.assertTrue(verified["valid"], verified["errors"])

    def test_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            output.mkdir()
            for name in gate_r_review.OUTPUTS:
                (output / name).write_text("{}\n", encoding="utf-8")
            outputs = {
                name: {"size": (output / name).stat().st_size, "sha256": gate_r_review.sha256_file(output / name)}
                for name in gate_r_review.OUTPUTS
            }
            self.write_json(output / "gate_r_manifest.json", {"outputs": outputs})
            review_path = output / "gate_r_technical_review.json"
            review_path.write_text('{"human_review":true,"implementation_authorized":true}\n', encoding="utf-8")
            verified = gate_r_review.verify(output)
            self.assertFalse(verified["valid"])


if __name__ == "__main__":
    unittest.main()
