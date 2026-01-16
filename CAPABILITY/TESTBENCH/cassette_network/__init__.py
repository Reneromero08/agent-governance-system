"""
Cassette Network Rigorous Test Suite

This package contains scientific, rigorous tests that validate the Cassette Network
actually works according to its vision claims, not just that it runs without errors.

Test categories:
- ground_truth/: Tests with known correct answers (not keyword matching)
- adversarial/: Tests that MUST reject invalid inputs
- determinism/: Tests that verify identical inputs produce identical outputs
- compression/: Tests that validate compression layer claims
- coverage/: Tests that measure corpus reachability
"""
