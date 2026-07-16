# Catalytic Wave Loop Restoration Law

**Status:** `FROZEN_PROSPECTIVE_CONTINUOUS_EQUIVALENCE_LAW`

## Two independent restoration channels

```text
phase carrier:
    max absolute complex sample error <= 1e-12

R1 ancestry:
    exact canonical committed T0 bytes
```

The phase carrier is a 6000-sample `complex128` numerical state. It is not defined as a
byte tape. IEEE-754 multiplication by unit-modulus beams and their conjugates may return
the same continuous state within the frozen numerical region while producing different
serialized bytes.

The accepted carrier language is therefore:

```text
software phase-carrier equivalence restoration
```

The forbidden language is:

```text
byte restoration
exact carrier restoration
hash restoration
```

unless the before and restored hashes actually match.

## Prospective envelope

The following values are source constants and loop-contract fields, not thresholds
chosen after observing the result:

```text
forward displacement L2 minimum   1.0
restoration metric                max_abs_complex_sample_error
restoration tolerance             1e-12
wrong-restoration minimum         0.05
wrong-query separation minimum    1e-6
```

## Mechanical acceptance conditions

The equivalence law is lawful only when all of these remain true:

```text
the carrier is continuous/numeric
the exact raw format is frozen
forward displacement is at least 1.0
the correct reverse schedule enters the 1e-12 region
wrong, reordered, omitted, duplicated, shifted, and no-restore arms remain outside 0.05
before, displaced, and restored hashes are reported separately
carrier_byte_exact equals the actual hash comparison
the claim uses equivalence language
T0 ancestry restores exact canonical bytes
```

## Qualified observation

```text
before SHA-256              b907b0c948cf7929353816771bc3c5916911e5f0240f17eb923af65ac4d79605
displaced SHA-256           ddf312eac86edad3f160048f06b5efa5e0346c8d83737acfe2c55136147f0157
restored SHA-256            dcdd7ecc904f435e5fe7ef9410872f4c117a95d001e99e90b008395af1d37917
carrier byte exact          false
forward displacement L2    73.1576613427
restore max error           4.74287484027e-16
equivalence restored        true
T0 ancestry byte exact      true
minimum wrong-arm error     0.959034213823
```

The hashes differ, so no byte-exact carrier claim is authorized. The correct error is
over three orders of magnitude below the frozen tolerance, while the closest declared
wrong arm is over nineteen times the frozen rejection minimum.

## Claim boundary

This law establishes no physical restoration, noise robustness, long-lived state,
hardware catalysis, Ising computation, or advantage over ordinary software. It applies
only to the deterministic committed software packet.
