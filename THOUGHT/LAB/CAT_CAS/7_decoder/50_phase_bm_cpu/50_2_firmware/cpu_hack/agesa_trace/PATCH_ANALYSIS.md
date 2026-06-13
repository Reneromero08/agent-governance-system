# AGESA Voltage Normalizer – Surgical P4-Only Patch Analysis

## Function: `AmdProcessorInitPeim` voltage normalizer
- BIOS raw offset: `0x00366de6 – 0x00366e93`
- PEIM VA: `0xfff66de6 – 0xfff66e93` (ImageBase `0xfff4008c`, body at `0x0034008c`)
- Uncompressed — direct hex edit possible with FFS checksum fix

## Exact Disassembly (corrected offsets)

```
00366E35: 3B C2             cmp eax, edx           ; compare field_a vs field_b
00366E37: C1 EF 09          shr edi, 9             ; shift field_b
00366E3A: 83 E0 7F          and eax, 0x7f          ; mask field_a
00366E3D: 83 E2 7F          and edx, 0x7f          ; mask field_b
00366E40: 3B C2             cmp eax, edx           ; compare masked values
00366E42: 74 41             je  +0x41              ; if equal, skip to 0x00366E85 (inc loop)
00366E44: 76 02             jbe +0x02              ; if field_a <= field_b, KEEP field_a
00366E46: 8B C2             mov eax, edx           ; else use field_b (higher voltage default)

... value reconstruction ...
00366E7D: E8 15 E0 FD FF    call WRMSR@0xfff44e91

00366E82: 83 C4 0C          add esp, 0xc
00366E85: FF 45 FC          inc dword [ebp-4]      ; next MSR (0xC0010064 → 0xC0010068)
00366E88: 81 7D FC 68 00 01 C0  cmp [ebp-4], 0xC0010068
00366E8F: 0F 86 67 FF FF FF     jbe back_to_loop

00366E95: 5F                pop edi
00366E96: 5E                pop esi
00366E97: 5B                pop ebx
00366E98: C9                leave
00366E99: C3                ret
```

Wait, let me cross-check with the hex dump bytes.

OK from the actual hex at offset 0x00366E26:
```
C1 EA 19 | 8B D1 | 8B FE | 0F AC FA 09 | 83 E0 7F | 83 E2 7F | C1 EF 09 | 3B C2 | 74 41 | 76 02 | 8B C2
```

Let me trace instruction boundaries from 0x00366E26:

0x00366E26: C1 EA 19       shr edx, 0x19          (3B) → 0x00366E29
0x00366E29: 8B D1          mov edx, ecx           (2B) → 0x00366E2B  
0x00366E2B: 8B FE          mov edi, esi           (2B) → 0x00366E2D
0x00366E2D: 0F AC FA 09    shrd edx, edi, 9       (4B) → 0x00366E31
0x00366E31: 83 E0 7F       and eax, 0x7f          (3B) → 0x00366E34
0x00366E34: 83 E2 7F       and edx, 0x7f          (3B) → 0x00366E37
0x00366E37: C1 EF 09       shr edi, 9             (3B) → 0x00366E3A
0x00366E3A: 3B C2          cmp eax, edx           (2B) → 0x00366E3C
0x00366E3C: 74 41          je  +0x41              (2B) → target = 0x00366E3E + 0x41 = 0x00366E7F
0x00366E3E: 76 02          jbe +0x02              (2B) → target = 0x00366E42
0x00366E40: 8B C2          mov eax, edx           (2B) → 0x00366E42

...then at 0x00366E42:
FF 75 08 | 83 E0 7F | 33 D2 | 8B F8 | 8B DA | 0F A4 FB 10 | C1 E7 10 | 0B F8 | 0B DA | 0F A4 FB 09 | 8D 45 F4 | 81 E1 FF 01 FF 01 | 50 | FF 75 FC | C1 E7 09 | 0B F9 | 0B DE | 89 7D F4 | 89 5D F8 | E8 15 E0 FD FF

0x00366E42: FF 75 08       push [ebp+8]           (3B) → 0x00366E45
0x00366E45: 83 E0 7F       and eax, 0x7f          (3B) → 0x00366E48
0x00366E48: 33 D2          xor edx, edx           (2B) → 0x00366E4A
0x00366E4A: 8B F8          mov edi, eax           (2B) → 0x00366E4C
0x00366E4C: 8B DA          mov ebx, edx           (2B) → 0x00366E4E
0x00366E4E: 0F A4 FB 10    shld ebx, edi, 0x10    (4B) → 0x00366E52
0x00366E52: C1 E7 10       shl edi, 0x10           (3B) → 0x00366E55
0x00366E55: 0B F8          or edi, eax            (2B) → 0x00366E57
0x00366E57: 0B DA          or ebx, edx            (2B) → 0x00366E59
0x00366E59: 0F A4 FB 09    shld ebx, edi, 9       (4B) → 0x00366E5D
0x00366E5D: 8D 45 F4       lea eax, [ebp-0xc]     (3B) → 0x00366E60
0x00366E60: 81 E1 FF 01 FF 01  and ecx, 0x1ff01ff (6B) → 0x00366E66
0x00366E66: 50             push eax               (1B) → 0x00366E67
0x00366E67: FF 75 FC       push [ebp-4]           (3B) → 0x00366E6A
0x00366E6A: C1 E7 09       shl edi, 9             (3B) → 0x00366E6D
0x00366E6D: 0B F9          or edi, ecx            (2B) → 0x00366E6F
0x00366E6F: 0B DE          or ebx, esi            (2B) → 0x00366E71
0x00366E71: 89 7D F4       mov [ebp-0xc], edi     (3B) → 0x00366E74
0x00366E74: 89 5D F8       mov [ebp-8], ebx       (3B) → 0x00366E77
0x00366E77: E8 15 E0 FD FF call WRMSR             (5B) → 0x00366E7C

0x00366E7C: 83 C4 0C       add esp, 0xc           (3B) → 0x00366E7F
0x00366E7F: FF 45 FC       inc dword [ebp-4]      (3B) → 0x00366E82
0x00366E82: 81 7D FC 68 00 01 C0  cmp [ebp-4], 0xC0010068  (7B) → 0x00366E89
0x00366E89: 0F 86 78 FF FF FF  jbe back_to_loop   (6B) → 0x00366E8F

Wait, let me check the jbe offset. The hex dump shows:
0F 86 67 FF FF FF

That's jbe with a signed 32-bit offset: 0xFFFFFF67 = -0x99. Target = 0x00366E8F + (-0x99) = 0x00366DF6.

0x00366DF6: 8B 75 F8       mov esi, [ebp-8]  ← this is inside the loop body, makes sense!

Then:
0x00366E8F: 5F             pop edi
0x00366E90: 5E             pop esi  
0x00366E91: 5B             pop ebx
0x00366E92: C9             leave
0x00366E93: C3             ret

Now everything makes sense! The `je +0x41` at 0x00366E3C targets 0x00366E3E + 0x41 = 0x00366E7F. That's the `inc dword [ebp-4]` at 0x00366E7F — skip the normalization and WRMSR, go straight to the next P-state. 

## Key Observation

For ALL stock P-states, field_a < field_b (0x06,0x12,0x14,0x16,0x1A < 0x20). The `jbe` at 0x00366E3E ALWAYS takes the branch (keeps field_a). The `mov eax, edx` at 0x00366E40 NEVER executes with stock values. 

The normalizer is effectively a NO-OP in the stock configuration. It only matters when field_a > field_b (which would happen if someone writes a higher-numeric VID to field_a at runtime).

## RECOMMENDED APPROACH: Runtime MSR Test First (NO BIOS FLASH)

Before patching the BIOS, test at RUNTIME whether the SVI will accept VID=0x20 for P4. The AGESA normalizer only runs at boot. After boot, we can write P-state definitions via /dev/cpu/*/msr. We already proved DID changes work — the question is whether the SVI hardware will clamp VID=0x20.

Test procedure:
1. Write P4 definition with VID=0x20, DID=3 (200MHz) to Core 4 via /dev/cpu/4/msr
2. P-state cycle (P0→P4) to force hardware reload
3. Read COFVID_STATUS to see if VID=0x20 was accepted or clamped

If SVI accepts VID=0x20 at runtime → no BIOS patch needed, proceed directly to sub-threshold tests
If SVI clamps VID back to 0x12 → then we need the BIOS patch approach

## BIOS PATCH OPTION (if runtime write fails)

The surgical BIOS patch requires fitting a P4-specific check in limited space.
The only viable approach without adding code elsewhere:

**Option A: Hardcode the post-normalizer VID for all P-states**
Replace the value construction/reconstruction logic with a hardcoded table lookup.
Too invasive — changes too many bytes, high risk.

**Option B: Skip P4 in loop, handle post-loop**
Change `cmp [ebp-4], 0xC0010068` at 0x00366E82 to `cmp [ebp-4], 0xC0010067`.
This makes the loop process P0-P3 only. P4 would need separate handling.
Problem: no space after the function epilogue for P4 handler.

**Option C: The "magic constant" trick**
Since the normalizer picks min(field_a, field_b), and we want P4 picked 0x20:
Replace one of the instruction bytes to make the comparison behave differently for P4.
Not feasible with existing byte budget.

**Conclusion: Test runtime MSR write first.** BIOS patch only if SVI clamps the runtime write.
