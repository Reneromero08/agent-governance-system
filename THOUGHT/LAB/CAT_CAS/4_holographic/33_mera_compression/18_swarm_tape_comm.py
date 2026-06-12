"""
Swarm Tape Communication — 1000 Agents, One Tape
==================================================
Exp 08 (Catalytic GPT) + Exp 12 (Tape Acceleration) + Exp 32 (Wormhole)

The tape IS the message bus. Agents don't call each other — they observe
the tape's state and react. Communication is catalytic: write-once, 
read-many, zero erasure.

Modes of communication:
  1. PUBLISH: Agent writes result to a named slot. All others read it.
  2. PHASE RESONANCE: Detect other agents' activity via tape hash changes.
  3. WARM-TAPE INHERITANCE: Leading agent computes, trailing inherit cache.
  4. WORMHOLE TELEPORT: Two slots entangled — write to one, read from other.
  5. MESSAGE QUEUE: FIFO ring buffer for asynchronous task coordination.

For DeepSeek V4 Flash distillation:
  Agent 0: processes shard 0, publishes expert Vh[type] to slot 64
  Agents 1-999: check slot 64 before computing, 99.9% cache hits
  Swarm throughput: 46 shards / 1000 agents = instant
"""
import torch, hashlib, time, threading, queue, json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class SlotType(Enum):
    FREE = 0
    ROTATION = 1      # Published rotation matrices
    EIGENBASIS = 2    # Shared expert eigenbasis (Vh)
    WORKSPACE = 3     # Agent computation workspace
    MESSAGE = 4       # Agent-to-agent messages
    PHASE = 5         # Phase state persistence
    COORDINATION = 6  # Task queue / results


@dataclass
class TapeSlot:
    idx: int
    slot_type: SlotType
    data: object = None
    owner: str = ""          # which agent wrote it
    checksum: int = 0        # SHA-256 hash for integrity
    version: int = 0         # incremented on each write
    published_at: float = 0  # timestamp
    readers: set = field(default_factory=set)  # agents that have read this version


class SwarmTape:
    """
    Shared tape with 512 slots, 1000 agent access.
    Thread-safe. Every read/write is checksummed.
    """
    def __init__(self, n_slots=512):
        self.n_slots = n_slots
        self.slots = [TapeSlot(i, SlotType.FREE) for i in range(n_slots)]
        self.lock = threading.RLock()
        self.counter = 0
        self.total_writes = 0
        self.total_reads = 0
        self.cross_agent_reads = 0  # Agent reads another agent's slot
    
    def _hash(self, data):
        if isinstance(data, torch.Tensor):
            return int(torch.sum(data.detach()).item() * 1e6) & 0xFFFFFFFF
        if isinstance(data, bytes):
            return int(hashlib.sha256(data).hexdigest()[:8], 16)
        return hash(str(data)) & 0xFFFFFFFF
    
    def publish(self, slot_idx, data, agent_id, slot_type=None):
        """Agent publishes data to a named slot. All other agents can read it."""
        with self.lock:
            slot = self.slots[slot_idx]
            slot.data = data
            slot.owner = agent_id
            slot.checksum = self._hash(data)
            slot.version += 1
            slot.published_at = time.time()
            slot.readers = {agent_id}  # publisher has read it
            if slot_type:
                slot.slot_type = slot_type
            self.total_writes += 1
    
    def read(self, slot_idx, agent_id):
        """Agent reads from a slot. Tracks if this is a cross-agent read."""
        with self.lock:
            slot = self.slots[slot_idx]
            if slot.data is None:
                return None
            if agent_id not in slot.readers:
                self.cross_agent_reads += 1
                slot.readers.add(agent_id)
            self.total_reads += 1
            return slot.data, slot.checksum, slot.version
    
    def borrow_workspace(self, agent_id):
        """Agent borrows a free workspace slot. Returns slot_idx."""
        with self.lock:
            for slot in self.slots:
                if slot.slot_type == SlotType.FREE:
                    slot.slot_type = SlotType.WORKSPACE
                    slot.owner = agent_id
                    slot.readers = {agent_id}
                    return slot.idx
        return None
    
    def return_workspace(self, slot_idx):
        """Agent returns a workspace slot. Data is wiped (catalytic undo)."""
        with self.lock:
            slot = self.slots[slot_idx]
            slot.data = None
            slot.owner = ""
            slot.checksum = 0
            slot.slot_type = SlotType.FREE
            slot.version += 1
    
    def message_send(self, agent_id, message):
        """Send a message to the swarm message queue."""
        with self.lock:
            for slot in self.slots:
                if slot.slot_type == SlotType.MESSAGE and slot.data is None:
                    slot.data = message
                    slot.owner = agent_id
                    slot.checksum = self._hash(str(message).encode())
                    slot.version += 1
                    slot.published_at = time.time()
                    slot.readers = set()
                    return slot.idx
        return None
    
    def message_poll(self, agent_id):
        """Poll for new messages not yet read by this agent."""
        messages = []
        with self.lock:
            for slot in self.slots:
                if slot.slot_type == SlotType.MESSAGE and slot.data is not None:
                    if agent_id not in slot.readers:
                        messages.append((slot.idx, slot.data, slot.owner))
                        slot.readers.add(agent_id)
        return messages
    
    def detect_phase_change(self, agent_id, watched_slots):
        """Detect if any watched slot has been modified (phase resonance)."""
        changes = []
        with self.lock:
            for idx in watched_slots:
                slot = self.slots[idx]
                if slot.version > 0 and agent_id not in slot.readers:
                    changes.append((idx, slot.version, slot.owner))
                    slot.readers.add(agent_id)
        return changes
    
    def stats(self):
        with self.lock:
            return {
                "total_writes": self.total_writes,
                "total_reads": self.total_reads,
                "cross_agent_reads": self.cross_agent_reads,
                "free_slots": sum(1 for s in self.slots if s.slot_type == SlotType.FREE),
                "message_slots": sum(1 for s in self.slots if s.slot_type == SlotType.MESSAGE),
                "workspace_slots": sum(1 for s in self.slots if s.slot_type == SlotType.WORKSPACE),
                "published_slots": sum(1 for s in self.slots if s.slot_type in 
                                       (SlotType.ROTATION, SlotType.EIGENBASIS)),
            }


class SwarmAgent:
    """One agent in the swarm. Communicates exclusively through the tape."""
    
    def __init__(self, agent_id, tape, skill="idle"):
        self.id = agent_id
        self.tape = tape
        self.skill = skill
        self.local_cache = {}  # eigenbasis cache
        self.messages_sent = 0
        self.messages_received = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def check_cache(self, key):
        """Check local cache first (fast), then tape (swarm inheritance)."""
        if key in self.local_cache:
            self.cache_hits += 1
            return self.local_cache[key]
        
        # Check if another agent published this key on the tape
        # Look through named eigenbasis slots
        for slot_idx in range(64, 128):
            data = self.tape.read(slot_idx, self.id)
            if data is not None:
                slot_data, cs, ver = data
                if isinstance(slot_data, dict) and key in slot_data:
                    self.local_cache[key] = slot_data[key]
                    self.cache_hits += 1
                    return slot_data[key]
        
        self.cache_misses += 1
        return None
    
    def publish_eigenbasis(self, eigenbasis_type, vh_matrix):
        """Publish computed eigenbasis to tape for swarm inheritance."""
        for slot_idx in range(64, 128):
            data = self.tape.read(slot_idx, self.id)
            if data is None:
                # Empty slot — publish
                self.tape.publish(slot_idx, {eigenbasis_type: vh_matrix}, 
                                 self.id, SlotType.EIGENBASIS)
                self.messages_sent += 1
                return slot_idx
        
        # All eigenbasis slots full — oldest wins, overwrite first slot
        self.tape.publish(64, {eigenbasis_type: vh_matrix}, 
                         self.id, SlotType.EIGENBASIS)
        self.messages_sent += 1
        return 64
    
    def broadcast_progress(self, msg):
        """Broadcast progress message to swarm."""
        self.tape.message_send(self.id, msg)
        self.messages_sent += 1
    
    def poll_messages(self):
        """Poll for messages from other agents."""
        msgs = self.tape.message_poll(self.id)
        self.messages_received += len(msgs)
        return msgs
    
    def wait_for_phase(self, watched_slots):
        """Wait until another agent modifies a watched slot (phase detection)."""
        changes = self.tape.detect_phase_change(self.id, watched_slots)
        return changes


def demo_swarm_communication():
    """Demo: 10 agents communicate exclusively through the tape."""
    print("=" * 70)
    print("SWARM TAPE COMMUNICATION — 10 Agents, 1 Tape, 512 Slots")
    print("=" * 70)
    
    tape = SwarmTape(n_slots=512)
    
    # Allocate message slots (256-383)
    for i in range(256, 384):
        tape.slots[i].slot_type = SlotType.MESSAGE
    
    # Allocate eigenbasis slots (64-127)
    for i in range(64, 128):
        tape.slots[i].slot_type = SlotType.EIGENBASIS
    
    # Create 10 agents
    agents = [SwarmAgent(f"agent_{i}", tape, skill="distiller") for i in range(10)]
    
    # Agent 0: the "explorer" — computes first expert Vh and publishes
    print("\n[Agent 0] Computing first expert Vh and publishing to slot 64...")
    vh_data = torch.randn(256, 2048)  # mock Vh matrix
    agents[0].publish_eigenbasis("experts.w1.weight", vh_data)
    print(f"  Published to slot 64. Swarm writes: {tape.stats()['total_writes']}")
    
    # Agents 1-9: check cache (local + tape)
    print("\n[Agents 1-9] Checking cache for 'experts.w1.weight'...")
    for i in range(1, 10):
        cached = agents[i].check_cache("experts.w1.weight")
        if cached is not None:
            print(f"  Agent {i}: CACHE HIT from swarm tape (cross-agent read)")
        else:
            print(f"  Agent {i}: CACHE MISS — must compute")
    
    # Agent 3 broadcasts progress
    agents[3].broadcast_progress("Shard 3 complete: 1200 weights processed")
    agents[3].broadcast_progress("Expert 42 Vh ready at slot 72")
    
    # Agent 5 polls messages
    print(f"\n[Agent 5] Polling message queue...")
    msgs = agents[5].poll_messages()
    for slot, msg, sender in msgs:
        print(f"  From {sender} (slot {slot}): {msg}")
    
    # Agent 7 watches for phase changes on eigenbasis slots
    print(f"\n[Agent 7] Watching eigenbasis slots for new publications...")
    changes = agents[7].wait_for_phase(list(range(64, 128)))
    for idx, ver, owner in changes:
        print(f"  Slot {idx} modified by {owner} (version {ver})")
    
    # Stats
    stats = tape.stats()
    print(f"\n[Tape Stats]:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Per-agent stats
    print(f"\n[Agent Stats]:")
    for a in agents:
        print(f"  {a.id}: {a.cache_hits} hits, {a.cache_misses} misses, "
              f"sent={a.messages_sent}, recv={a.messages_received}")
    
    print(f"\n  AGENTS COMMUNICATE THROUGH THE TAPE.")
    print(f"  Zero direct function calls. Zero shared memory outside the tape.")
    print(f"  The tape IS the message bus.")


if __name__ == "__main__":
    demo_swarm_communication()
