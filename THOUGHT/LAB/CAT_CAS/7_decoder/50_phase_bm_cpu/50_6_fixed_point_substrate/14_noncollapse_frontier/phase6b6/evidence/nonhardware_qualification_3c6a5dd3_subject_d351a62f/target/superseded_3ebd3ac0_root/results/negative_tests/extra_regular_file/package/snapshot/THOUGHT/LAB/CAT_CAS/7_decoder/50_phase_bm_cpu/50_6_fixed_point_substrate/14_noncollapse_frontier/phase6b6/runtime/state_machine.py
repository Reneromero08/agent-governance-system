"""Explicit sender state machine for Phase 6B.6 software runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_EVENTS = ("SENDER_START", "SENDER_CONTINUE", "SENDER_PHASE_UPDATE", "SENDER_STOP", "SENDER_OFF")


@dataclass
class SenderStateMachine:
    live_epoch: str | None = None
    phase_action: str | None = None
    packet_id: str | None = None

    def apply(self, slot: dict[str, Any]) -> list[dict[str, Any]]:
        executed = slot["executed"]
        drive_on = bool(executed["drive_on"])
        epoch = executed.get("sender_epoch_id")
        phase = executed.get("phase_action")
        events: list[dict[str, Any]] = []
        if not drive_on:
            if epoch is not None:
                raise ValueError("hidden drive: sender-off slot carries sender epoch")
            if self.live_epoch is not None:
                events.append(self._event("SENDER_STOP", slot, self.live_epoch))
                self.live_epoch = None
                self.phase_action = None
                self.packet_id = None
            events.append(self._event("SENDER_OFF", slot, None))
            return events
        if epoch is None:
            raise ValueError("driven slot without sender epoch")
        if self.live_epoch is None:
            self.live_epoch = epoch
            self.phase_action = phase
            self.packet_id = slot.get("packet_id")
            events.append(self._event("SENDER_START", slot, epoch))
            return events
        if epoch != self.live_epoch:
            if self.packet_id is not None and slot.get("packet_id") == self.packet_id:
                raise ValueError("invalid transition: driven epoch replaced before sender stop")
            events.append(self._event("SENDER_STOP", slot, self.live_epoch))
            self.live_epoch = epoch
            self.phase_action = phase
            self.packet_id = slot.get("packet_id")
            events.append(self._event("SENDER_START", slot, epoch))
            return events
        if phase != self.phase_action:
            self.phase_action = phase
            events.append(self._event("SENDER_PHASE_UPDATE", slot, epoch))
        else:
            events.append(self._event("SENDER_CONTINUE", slot, epoch))
        return events

    def finish(self, final_slot: dict[str, Any]) -> list[dict[str, Any]]:
        if self.live_epoch is None:
            return []
        event = self._event("SENDER_STOP", final_slot, self.live_epoch)
        self.live_epoch = None
        self.phase_action = None
        self.packet_id = None
        return [event]

    @staticmethod
    def _event(event: str, slot: dict[str, Any], epoch: str | None) -> dict[str, Any]:
        if event not in VALID_EVENTS:
            raise ValueError(f"invalid runtime event: {event}")
        return {
            "event": event,
            "slot_index": slot["slot_index"],
            "packet_id": slot.get("packet_id"),
            "sender_epoch_id": epoch,
            "phase_action": slot["executed"].get("phase_action"),
        }


def validate_runtime_events(custody: dict[str, Any]) -> None:
    for session in custody["sessions"]:
        live = None
        for row in session["slots"]:
            for event in row["runtime_events"]:
                kind = event["event"]
                if kind == "SENDER_START":
                    if live is not None:
                        raise ValueError("start while sender live")
                    live = event["sender_epoch_id"]
                elif kind in ("SENDER_CONTINUE", "SENDER_PHASE_UPDATE"):
                    if live != event["sender_epoch_id"]:
                        raise ValueError("continue/update without matching live sender")
                elif kind == "SENDER_STOP":
                    if live != event["sender_epoch_id"]:
                        raise ValueError("stop without matching live sender")
                    live = None
                elif kind == "SENDER_OFF":
                    if live is not None:
                        raise ValueError("off event while sender live")
                else:
                    raise ValueError("unknown runtime event")
            if not row["u_t"]["drive_on"] and row["u_t"].get("sender_epoch_id") is not None:
                raise ValueError("hidden drive in sender-off captured row")
        if live is not None:
            raise ValueError("session ended with live sender")
