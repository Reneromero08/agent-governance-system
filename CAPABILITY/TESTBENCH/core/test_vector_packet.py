#!/usr/bin/env python3
"""
SVTP (Semantic Vector Transport Protocol) Production Tests

Tests the production SVTP implementation in CAPABILITY/PRIMITIVES/vector_packet.py

SVTP provides cross-model vector communication with:
- 256D packet structure (payload, pilot tone, auth, sequence)
- Geometric checksum via pilot tone
- Auth token for verification
- Cross-model alignment via alignment keys
- Corruption detection

Original tests from: THOUGHT/LAB/VECTOR_ELO/eigen-alignment/vector-communication/tests/
Promoted to production: 2026-02-01
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from CAPABILITY.PRIMITIVES.vector_packet import (
    SVTPEncoder, SVTPDecoder, CrossModelEncoder, CrossModelDecoder,
    SVTPPacket, SVTP_256, create_svtp_channel, format_packet_hex,
    PAYLOAD_START, PAYLOAD_END, PILOT_START, PILOT_END, AUTH_START, AUTH_END,
)
from CAPABILITY.PRIMITIVES.canonical_anchors import CANONICAL_128


@pytest.fixture
def sentence_transformer():
    """Fixture to provide sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        pytest.skip("sentence-transformers not installed")


@pytest.fixture
def alignment_key(sentence_transformer):
    """Fixture to provide alignment key."""
    def embed(texts):
        return sentence_transformer.encode(texts, convert_to_numpy=True)
    
    return AlignmentKey.create("MiniLM", embed, anchors=CANONICAL_128, k=128)


@pytest.fixture
def svtp_encoder_decoder(alignment_key, sentence_transformer):
    """Fixture to provide encoder/decoder pair."""
    def embed(texts):
        return sentence_transformer.encode(texts, convert_to_numpy=True)
    
    encoder = SVTPEncoder(alignment_key, embed)
    decoder = SVTPDecoder(alignment_key, embed)
    return encoder, decoder


class TestSVTPSingleModel:
    """Test single-model encode/decode."""

    def test_basic_encode_decode(self, svtp_encoder_decoder):
        """Test basic encoding and decoding of a message."""
        encoder, decoder = svtp_encoder_decoder
        
        message = "The neural network learns patterns"
        candidates = [message, "Distractor message one", "Another unrelated text"]
        
        packet = encoder.encode(message, sequence=0)
        result = decoder.decode(packet.vector, candidates)
        
        assert result.valid, "Packet should be valid"
        assert result.pilot_valid, "Pilot tone should be valid"
        assert result.auth_valid, "Auth token should be valid"
        assert result.payload == message, f"Expected '{message}', got '{result.payload}'"
        assert result.sequence == 0, "Sequence number should be preserved"

    def test_multiple_messages(self, svtp_encoder_decoder):
        """Test encoding/decoding multiple different messages."""
        encoder, decoder = svtp_encoder_decoder
        
        test_messages = [
            "The neural network learns patterns",
            "Love conquers all obstacles",
            "Water flows downhill naturally",
        ]
        
        candidates = test_messages + ["Distractor message one", "Random noise here"]
        
        for i, msg in enumerate(test_messages):
            packet = encoder.encode(msg, sequence=i)
            result = decoder.decode(packet.vector, candidates)
            
            assert result.payload == msg, f"Message {i}: Expected '{msg}', got '{result.payload}'"

    def test_sequence_number_preservation(self, svtp_encoder_decoder):
        """Test that sequence numbers are correctly preserved."""
        encoder, decoder = svtp_encoder_decoder
        
        sequences = [0, 42, 127, 255]
        message = "Test message"
        candidates = [message]
        
        for seq in sequences:
            packet = encoder.encode(message, sequence=seq)
            result = decoder.decode(packet.vector, candidates)
            
            assert result.sequence == seq, f"Sequence {seq} not preserved, got {result.sequence}"


class TestSVTPPilotTone:
    """Test pilot tone corruption detection."""

    def test_clean_packet_valid(self, svtp_encoder_decoder):
        """Test that clean packet validates correctly."""
        encoder, decoder = svtp_encoder_decoder
        
        message = "Test message for corruption"
        candidates = [message, "Other message", "Wrong answer"]
        
        packet = encoder.encode(message, sequence=0)
        
        # Re-create decoder with explicit pilot threshold
        _, decoder = svtp_encoder_decoder
        decoder = SVTPDecoder(decoder.key, decoder.embed_fn, pilot_threshold=0.6)
        
        result = decoder.decode(packet.vector, candidates)
        
        assert result.valid, "Clean packet should be valid"
        assert result.pilot_valid, "Clean packet should have valid pilot"
        assert result.auth_valid, "Clean packet should have valid auth"

    def test_corrupted_pilot_detected(self, svtp_encoder_decoder):
        """Test that corrupted pilot tone is detected."""
        encoder, decoder = svtp_encoder_decoder
        
        message = "Test message for corruption"
        candidates = [message, "Other message", "Wrong answer"]
        
        packet = encoder.encode(message, sequence=0)
        
        # Corrupt pilot tone
        corrupted = packet.vector.copy()
        corrupted[PILOT_START:PILOT_END] = np.random.randn(PILOT_END - PILOT_START)
        
        # Re-create decoder with explicit pilot threshold
        _, decoder = svtp_encoder_decoder
        decoder = SVTPDecoder(decoder.key, decoder.embed_fn, pilot_threshold=0.6)
        
        result = decoder.decode(corrupted, candidates)
        
        assert not result.pilot_valid, "Corrupted pilot should be invalid"
        assert not result.valid, "Packet with corrupted pilot should be invalid"

    def test_corrupted_payload_pilot_still_valid(self, svtp_encoder_decoder):
        """Test that payload corruption doesn't affect pilot tone."""
        encoder, decoder = svtp_encoder_decoder
        
        message = "Test message for corruption"
        candidates = [message, "Other message", "Wrong answer"]
        
        packet = encoder.encode(message, sequence=0)
        
        # Corrupt only payload
        payload_corrupted = packet.vector.copy()
        payload_corrupted[0:50] = np.random.randn(50)
        
        # Re-create decoder with explicit pilot threshold
        _, decoder = svtp_encoder_decoder
        decoder = SVTPDecoder(decoder.key, decoder.embed_fn, pilot_threshold=0.6)
        
        result = decoder.decode(payload_corrupted, candidates)
        
        assert result.pilot_valid, "Pilot should still be valid with payload corruption"


class TestSVTPPacketStructure:
    """Test packet structure and serialization."""

    def test_packet_properties(self, svtp_encoder_decoder):
        """Test packet properties are correctly set."""
        encoder, _ = svtp_encoder_decoder
        
        message = "Test message"
        packet = encoder.encode(message, sequence=42)
        
        assert packet.sequence == 42
        assert packet.payload_text == message
        assert len(packet.vector) == 256, "Packet should be 256D"
        assert len(packet.payload) == 200, "Payload should be 200D"
        assert len(packet.pilot_tone) == 20, "Pilot should be 20D"
        assert len(packet.auth_token) == 35, "Auth should be 35D"

    def test_packet_to_bytes(self, svtp_encoder_decoder):
        """Test packet serialization to bytes."""
        encoder, _ = svtp_encoder_decoder
        
        message = "Serialize this message"
        packet = encoder.encode(message, sequence=99)
        
        # Serialize
        data = packet.to_bytes()
        assert len(data) > 0, "Serialized data should not be empty"
        
        # Deserialize
        restored = SVTPPacket.from_bytes(data)
        
        assert restored.sequence == packet.sequence, "Sequence should match"
        assert len(restored.vector) == len(packet.vector), "Vector length should match"
        
        # Verify vectors match
        diff = np.max(np.abs(packet.vector - restored.vector))
        assert diff < 1e-6, f"Vector difference too large: {diff}"


class TestSVTPCrossModel:
    """Test cross-model SVTP communication."""

    def test_cross_model_communication(self):
        """Test communication between different embedding models."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")
        
        # Load two different models
        model_a = SentenceTransformer('all-MiniLM-L6-v2')
        model_b = SentenceTransformer('all-mpnet-base-v2')
        
        def embed_a(texts):
            return model_a.encode(texts, convert_to_numpy=True)
        
        def embed_b(texts):
            return model_b.encode(texts, convert_to_numpy=True)
        
        # Create alignment keys
        key_a = AlignmentKey.create("MiniLM", embed_a, anchors=CANONICAL_128, k=128)
        key_b = AlignmentKey.create("MPNet", embed_b, anchors=CANONICAL_128, k=128)
        
        # Align keys
        pair = key_a.align_with(key_b)
        
        # Note: MiniLM and MPNet are architecturally different models
        # Alignment residual may be higher than 0.5, but communication can still work
        # The important metric is whether messages are successfully decoded
        print(f"\nAlignment residual: {pair.procrustes_residual:.4f}")
        if pair.procrustes_residual > 0.5:
            print(f"  Warning: High alignment residual, but attempting communication anyway")
        
        # Create SVTP channel
        enc_a, dec_a, enc_b, dec_b = create_svtp_channel(pair, embed_a, embed_b)
        
        # Test A -> B
        test_messages = [
            "Neural networks learn from data",
            "The ocean waves crash on shore",
            "Mathematics describes the universe",
        ]
        
        candidates = test_messages + ["Random distractor text", "Another unrelated sentence"]
        
        ab_successes = 0
        for i, msg in enumerate(test_messages):
            packet = enc_a.encode_to_other(msg, sequence=i)
            result = dec_b.decode(packet.vector, candidates)
            
            if result.payload == msg:
                ab_successes += 1
        
        # Test B -> A
        ba_successes = 0
        for i, msg in enumerate(test_messages):
            packet = enc_b.encode_to_other(msg, sequence=i)
            result = dec_a.decode(packet.vector, candidates)
            
            if result.payload == msg:
                ba_successes += 1
        
        total = ab_successes + ba_successes
        expected = len(test_messages) * 2
        
        # Expect at least 80% accuracy
        assert total >= expected * 0.8, f"Cross-model accuracy too low: {total}/{expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
