import json
import numpy as np
from typing import Optional, Dict

from CAPABILITY.PRIMITIVES.vector_packet import SVTPPacket, CrossModelDecoder, CrossModelEncoder, SVTP_256
from CAPABILITY.PRIMITIVES.alignment_key import AlignedKeyPair
from CAPABILITY.PRIMITIVES.geometric_reasoner import GeometricState
from NAVIGATION.CORTEX.network.network_hub import SemanticNetworkHub
from NAVIGATION.CORTEX.network.codebook_sync import SyncTuple, CodebookSync

class SVTPCortexBridge:
    """Bridge for receiving SVTP packets from .holo models and querying the Cassette Network."""
    
    def __init__(self, hub: SemanticNetworkHub, aligned_pair: AlignedKeyPair, embed_fn):
        self.hub = hub
        self.aligned_pair = aligned_pair
        self.embed_fn = embed_fn
        # Decoder receives from external model (Model B) so we are Model A
        self.decoder = CrossModelDecoder(aligned_pair, embed_fn, is_model_a=True)
        # Encoder sends to external model
        self.encoder = CrossModelEncoder(aligned_pair, embed_fn, is_model_a=True)
        self.sync = CodebookSync(sender_id="cortex-svtp-bridge")
        self._session_established = False

    def establish_sync(self, remote_sync_tuple_dict: dict) -> dict:
        """Perform CODEBOOK_SYNC_PROTOCOL handshake before processing packets."""
        remote_tuple = SyncTuple.from_dict(remote_sync_tuple_dict)
        if self.hub._hub_sync_tuple is None:
            return {"status": "ERROR", "error": "Hub has no sync tuple"}
            
        is_match, mismatches = self.sync.sync_tuples_match(remote_tuple, self.hub._hub_sync_tuple)
        if is_match:
            self._session_established = True
            return {"status": "ALIGNED", "session_token": "svtp-session"}
        else:
            return {"status": "DISSOLVED", "mismatches": mismatches}

    def handle_packet(self, packet_bytes: bytes) -> bytes:
        """Process an incoming SVTP packet, route as geometric query, and return SVTP response."""
        if not self._session_established:
            return self.encoder.encode_to_other("ERROR: E_BLANKET_DISSOLVED").to_bytes()
            
        try:
            packet = SVTPPacket.from_bytes(packet_bytes)
            
            # Verify packet integrity (Pilot and Auth)
            result = self.decoder.decode(packet.vector, candidates=[], verify_pilot=True, verify_auth=True)
            if not result.valid:
                return self.encoder.encode_to_other(f"ERROR: {result.error}").to_bytes()
                
            payload = packet.payload
            
            # Convert SVTP payload to full-dimensional GeometricState
            dim = 384
            if hasattr(self.embed_fn, "get_sentence_embedding_dimension"):
                dim = self.embed_fn.get_sentence_embedding_dimension()
                
            padded_vector = np.zeros(dim)
            payload_len = min(len(payload), dim)
            padded_vector[:payload_len] = payload[:payload_len]
            
            query_state = GeometricState(vector=padded_vector)
            
            # Route geometric query to cassette network
            hub_results = self.hub.query_merged_geometric(query_state, top_k=3)
            
            # Encode response
            if not hub_results:
                response_text = "NO_RESULTS"
            else:
                # Return hashes/pointers of top results
                response_text = json.dumps([r.get('hash', r.get('chunk_id', '')) for r in hub_results])[:200]
                
            return self.encoder.encode_to_other(response_text).to_bytes()
            
        except Exception as e:
            return self.encoder.encode_to_other(f"ERROR: {str(e)}").to_bytes()
