# PHASE2_AGESA_VID_SOURCES

## Verdict

VID_SOURCES_IDENTIFIED_AS_RUNTIME_MSR_AND_CONSTRUCTOR_PATHS_NO_P4_ONLY_SOURCE_PROVEN

## Immediate Source For Compared Fields

The normalizer reads each P-state MSR through wrapper `0xFFF44E76`.

```text
fff66df6: push     dword ptr [ebp + 8]
fff66df9: lea      eax, [ebp - 0xc]
fff66dfc: push     eax
fff66dfd: push     dword ptr [ebp - 4]
fff66e00: call     0xfff44e76
```

The compared fields are extracted from the just-read qword at `[ebp-0xc]` / `[ebp-8]`:

```text
fff66e1b: mov      ecx, dword ptr [ebp - 0xc]
fff66e1e: mov      eax, ecx
fff66e20: mov      edx, esi
fff66e22: shrd     eax, edx, 0x19
fff66e26: shr      edx, 0x19
fff66e29: mov      edx, ecx
fff66e2b: mov      edi, esi
fff66e2d: shrd     edx, edi, 9
fff66e31: and      eax, 0x7f
fff66e34: and      edx, 0x7f
fff66e37: shr      edi, 9
fff66e3a: cmp      eax, edx
```

Interpretation:

- `eax` is a VID-like field extracted through a 25-bit right shift of the MSR qword.
- `edx` is a VID-like field extracted through a 9-bit right shift of the same MSR qword.
- Both fields are masked with `0x7F`.
- Stock selection keeps the lower numeric VID unless the fields are equal.

## Nearby Writers And Constructors

### Normalizer Writer

The normalizer writes only when the selected field changes the qword:

```text
fff66e67: push     dword ptr [ebp - 4]
...
fff66e71: mov      dword ptr [ebp - 0xc], edi
fff66e74: mov      dword ptr [ebp - 8], ebx
fff66e77: call     0xfff44e91
```

This writer loops over `0xC0010064-0xC0010068`. It is not P4-only.

### P-state MSR Field Update Helper

`0xFFF6788D-0xFFF6791A` loops over the same MSR range:

```text
fff678a7: mov      edi, 0xc0010064
...
fff678db: mov      al, byte ptr [esi]
fff678e3: and      al, 0x7f
...
fff678f0: and      ecx, 0x1ffffff
fff678f6: or       eax, ecx
fff678f8: mov      dword ptr [ebp - 0xc], eax
fff678ff: or       edx, ebx
fff67901: push     edi
fff67902: mov      dword ptr [ebp - 8], edx
fff67905: call     0xfff44e91
fff6790d: inc      edi
fff6790e: cmp      edi, 0xc0010068
fff67914: jbe      0xfff678ac
```

This helper reads bytes from a caller-supplied structure at `esi`. It can feed VID-like values, but no specific P4-only record address is proven from the current artifacts.

### Main P-state Write Loop

`0xFFF737A3-0xFFF73A90` builds and writes P-state MSRs from a structure whose entries are spaced by `0x18` bytes:

```text
fff7397d: movzx    eax, byte ptr [ecx + 0xb]
fff73981: mov      dword ptr [ebp - 0x20], edi
fff73984: cmp      edi, eax
fff73986: ja       0xfff73a8d
fff7398c: lea      eax, [edi - 0x3ffeff9c]
fff73992: imul     edi, edi, 0x18
...
fff739ba: cmp      dword ptr [edi + 0x10], 1
...
fff73a1f: movzx    eax, byte ptr [edi + 0x1c]
...
fff73a34: mov      ecx, dword ptr [edi + 0x20]
...
fff73a68: push     dword ptr [ebp + 0xc]
fff73a6b: call     0xfff44e91
fff73a7e: inc      dword ptr [ebp - 0x20]
fff73a81: inc      dword ptr [ebp + 0xc]
fff73a84: cmp      dword ptr [ebp - 0x20], eax
fff73a87: jbe      0xfff739a2
```

This is the strongest constructor-like source found in the existing disassembly. It implies an AGESA table or runtime structure with per-P-state entries, but the backing table bytes have not been located in the BIOS dump as an editable static record.

### Other P-state Writers

Additional MSR writer loops in the same artifact:

| Range | Evidence | Status |
|---|---|---|
| `0xFFF67468-0xFFF67557` | Writes low FID/DID-like fields via `0xC0010064` and later paths | Not a VID-only source |
| `0xFFF7CD86-0xFFF7CDE5` | Sequential P-state MSR writer | Not enough context to select P4 VID only |
| DXE64 P-state loops | Similar high-level logic in 64-bit code paths | Not the target PEIM normalizer |

## VID Source Decision

The current artifacts identify the runtime MSR as the immediate source and a constructor path that likely consumes per-P-state entries. They do not prove the static location of a P4-only table record. Without that record, a table edit cannot be made byte-ready.

