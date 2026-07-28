# CATVM Cyclotomic Cubic TT Boundary

## Enforced custody

One Linux userspace service owns a persistent width-four exact
`Q(zeta_5)` tensor-train carrier. The accepted service is non-dumpable,
listens on a mode-`0600` Unix `SOCK_SEQPACKET` socket, validates same-UID peer
credentials, and accepts fixed 1,024-byte requests. Every response is exactly
4,096 bytes.

The controller imports only a protocol/framing module; it does not import or
load the service or phase engine. A structural source/runtime module gate
enforces that separation. The controller can select only public primary or
reuse programs. It receives:

```text
one final four-coefficient cyclotomic amplitude
restoration generation
actual-inverse flag
snapshot flag
one-way custody receipt
```

It cannot request tensors, bonds, ranks, pivots, Fourier intermediates, or
factorization state. Intermediate projection and null-carrier commands fail
closed.

## Restoration and reuse

The service latches the final amplitude internally, applies the actual
inverse to the resident wave TT, verifies exact restoration, and only then
returns the boundary. The unrelated second request consumes the same
restored carrier; generations advance from one to two.

The accepted in-place process contains no snapshot image. A separate
snapshot-mode service owns the saved baseline, matches the primary boundary,
and cannot mint an actual-inverse restoration generation. It charges all
three 160-byte logical-payload copies: image creation, execution load, and
restoration reload (480 logical copy bytes total). It separately reports the
measured Python resident size of the image, working copy, and restored copy;
the logical figures are not labeled as allocator traffic.

## Ceiling

The accepted machine claim is bounded to Linux userspace, width four,
primary rounds four, reuse rounds three, and exact software arithmetic. It
does not establish distinct phase resources, advantage, Small Wall crossing,
physical waveform execution, or protection against a privileged host.
