# AGESA Gate 1 CFG Pseudocode

Status: `GATE1_CFG_RECONSTRUCTED`

Scope: owned local Phenom II X6 1090T / GA-970A-DS3P firmware route research. No flash command. No hardware-changing command.

## Evidence

- BIOS dump: `50_2_firmware/cpu_hack/bios_dump.bin`
- BIOS SHA-256: `B7C0C725C4B6F50F399A208E5CAD6938BAACDD8FA1BBC795098CA393083FBC91`
- Target module: `AmdProcessorInitPeim`
- File GUID: `DE3E049C-A218-4891-8658-5FC0FA84C788`
- Corrected PE32 body: `50_2_firmware/cpu_hack/bios_dump.bin.dump/5 8C8CE578-8A3D-4F1C-9935-896185C32DD3/0 AmdProcessorInitPeim/1 PE32 image section/body.bin`
- PE32 body SHA-256: `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A`
- PE32 body matches BIOS slice at raw `0x0034008C`.
- Image base mapping used by current disassembly: raw `0x0034008C` -> VA `0xFFF4008C`.

## Range 1: Normalizer `0xFFF66DE6-0xFFF66E93`

Purpose: read P-state MSRs `MSRC001_0064` through `MSRC001_0068`, compare two VID-like 7-bit fields, and rewrite both fields to the lower numeric value when they differ.

Decompiler-grade pseudocode:

```c
void normalize_pstate_vids(void *ctx)
{
    uint32_t msr = 0xC0010064;

    do {
        uint64_t value = rdmsr_wrapper(msr, ctx);
        uint32_t lo = (uint32_t)value;
        uint32_t hi = (uint32_t)(value >> 32);

        if ((hi >> 31) == 1) {
            uint32_t field_25 = (uint32_t)((value >> 25) & 0x7F);
            uint32_t field_9 = (uint32_t)((value >> 9) & 0x7F);

            if (field_25 != field_9) {
                uint32_t selected = field_25;
                if (field_25 > field_9) {
                    selected = field_9;
                }

                uint64_t rebuilt = value;
                rebuilt &= ~(((uint64_t)0x7F << 25) | ((uint64_t)0x7F << 9));
                rebuilt |= ((uint64_t)selected << 25) | ((uint64_t)selected << 9);
                wrmsr_wrapper(msr, rebuilt, ctx);
            }
        }

        msr++;
    } while (msr <= 0xC0010068);
}
```

CFG:

| Block | VA range | Role |
|---|---:|---|
| B0 | `0xFFF66DE6-0xFFF66DF5` | Prologue; initialize loop MSR to `0xC0010064`. |
| B1 | `0xFFF66DF6-0xFFF66E15` | Read current MSR; test enable/high bit through `hi >> 31`. |
| B2 | `0xFFF66E17-0xFFF66E3C` | Extract and compare fields `(value >> 25) & 0x7F` and `(value >> 9) & 0x7F`. |
| B3 | `0xFFF66E3E-0xFFF66E40` | Stock selector: choose lower numeric VID-like field. |
| B4 | `0xFFF66E42-0xFFF66E7C` | Rebuild both fields and write MSR. |
| B5 | `0xFFF66E7F-0xFFF66E89` | Increment MSR; loop while `msr <= 0xC0010068`. |
| B6 | `0xFFF66E8F-0xFFF66E93` | Epilogue/return. |

P4 distinction: only the loop counter value `[ebp-4] == 0xC0010068` distinguishes P4. The rejected global branch byte at `0x00366E3E` does not test that condition.

## Range 2: Helper `0xFFF6788D-0xFFF6791A`

Purpose: update a 7-bit VID-like field in each P-state MSR from a caller-supplied byte structure.

Decompiler-grade pseudocode:

```c
void pstate_field_update_helper(uint8_t *src, void *ctx)
{
    uint32_t local;
    setup_helper(ctx, &local);

    uint32_t msr = 0xC0010064;
    do {
        uint64_t value = rdmsr_wrapper(msr, ctx);
        uint32_t lo = (uint32_t)value;
        uint32_t hi = (uint32_t)(value >> 32);

        bool enabled = ((lo & 0x00400000) != 0);
        if (!enabled || src[1] != 0) {
            uint32_t selected = src[0] & 0x7F;
            uint64_t rebuilt = value;
            rebuilt &= ~((uint64_t)0x7F << 25);
            rebuilt |= (uint64_t)selected << 25;
            wrmsr_wrapper(msr, rebuilt, ctx);
        }

        msr++;
    } while (msr <= 0xC0010068);
}
```

CFG:

| Block | VA range | Role |
|---|---:|---|
| H0 | `0xFFF6788D-0xFFF678A7` | Prologue; helper setup; initialize loop MSR to `0xC0010064`. |
| H1 | `0xFFF678AC-0xFFF678C6` | Read current MSR; skip update when source byte/condition blocks write. |
| H2 | `0xFFF678C8-0xFFF6790A` | Read `src[0]`, mask to `0x7F`, rebuild field at bit 25, write MSR. |
| H3 | `0xFFF6790D-0xFFF67914` | Increment MSR; loop while `msr <= 0xC0010068`. |
| H4 | `0xFFF67916-0xFFF6791A` | Epilogue/return. |

P4 distinction: same as the normalizer. It is loop-counter based only. The source pointer at `esi` may be a table/control structure, but this range does not prove a static P4-only record address.

## Range 3: Constructor Path `0xFFF737A3-0xFFF73A90`

Purpose: construct and write P-state MSR values from a runtime structure. The strongest table clue is a 0x18-byte stride and per-entry fields read at `+0x10`, `+0x14`, `+0x1C`, and `+0x20`.

Decompiler-grade pseudocode for the relevant path:

```c
void pstate_constructor_path(struct PstateCtx *ctx, void *service, void *wrctx)
{
    uint8_t first = ctx->base[0x0F];
    uint32_t pstate_index = first;

    if (ctx->mode_byte_at_esi_plus_4 != 0) {
        uint32_t msr = pstate_index - 0x3FFEFF9C; /* equals 0xC0010064 + pstate_index in this mapping */
        uint64_t value = rdmsr_wrapper(msr, wrctx);

        helper_fff73559(ctx->entry_or_field_0x14, &local_a, &local_b, &local_c);
        value = update_fid_did_like_fields(value, local_b, local_c);
        value = set_byte_field(value, ctx->base[0x1C]);
        value = set_two_bit_field(value, ctx->base32[0x20 / 4] & 3);
        value |= 0x8000000000000000ULL;

        wrmsr_wrapper(msr, value, wrctx);
        wrmsr_wrapper(0xC0010065 + pstate_index, value, wrctx);
        perform_transition_and_wait_loops(service, wrctx);
        return;
    }

    uint8_t max_pstate = ctx->base[0x0B];
    if (pstate_index > max_pstate) {
        return;
    }

    uint32_t record_offset = pstate_index * 0x18;
    uint8_t *record = ctx->base + record_offset;
    uint32_t msr = pstate_index - 0x3FFEFF9C;

    do {
        uint64_t value = rdmsr_wrapper(msr, wrctx);

        if (*(uint32_t *)(record + 0x10) == 1) {
            uint32_t decode_status = helper_fff73559(*(uint32_t *)(record + 0x14),
                                                     &local_a, &local_b, &local_c);
            if (decode_status != 5) {
                value = update_fid_did_like_fields(value, local_b, local_c);
            }

            value = set_byte_field(value, *(uint8_t *)(record + 0x1C));
            value = set_two_bit_field(value, *(uint32_t *)(record + 0x20) & 3);
            value |= 0x8000000000000000ULL;
        } else {
            value &= ~0x8000000000000000ULL;
        }

        wrmsr_wrapper(msr, value, wrctx);

        record += 0x18;
        pstate_index++;
        msr++;
    } while (pstate_index <= max_pstate);
}
```

CFG:

| Block | VA range | Role |
|---|---:|---|
| C0 | `0xFFF737A3-0xFFF737B0` | Load context and initial P-state index; branch on mode byte. |
| C1 | `0xFFF737B6-0xFFF73978` | Single/current P-state construction path with transition wait loops. |
| C2 | `0xFFF7397D-0xFFF73998` | Multi-entry path; load max P-state count, compute `index * 0x18`. |
| C3 | `0xFFF739A2-0xFFF739BE` | Read current MSR and test record enable flag at `record + 0x10`. |
| C4 | `0xFFF739C4-0xFFF73A54` | Enabled record path; consume fields at `+0x14`, `+0x1C`, `+0x20`; set enable bit. |
| C5 | `0xFFF73A56-0xFFF73A60` | Disabled record path; clear enable bit. |
| C6 | `0xFFF73A63-0xFFF73A87` | Write MSR; increment record by `0x18`; increment P-state and MSR; loop through max. |
| C7 | `0xFFF73A8D-0xFFF73A90` | Epilogue continuation. |

## Gate 1 Decision

Gate 1 is complete enough to drive Gate 2. The normalizer and helper are not P4-safe by themselves. The constructor path is the live route because it exposes a 0x18-byte per-P-state structure, a max P-state count at `ctx + 0x0B`, and fields at `record + 0x10`, `+0x14`, `+0x1C`, and `+0x20`.

