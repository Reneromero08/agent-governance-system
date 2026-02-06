# Q32 Phase 3 evidence package (neighbor falsifier + transfer stability)

Date: 2026-01-10

This is a **Phase 3 evidence index** for Q32. It is **not** a claim that Q32 is fully solved; it is the auditable data trail for the Phase 3 gates defined in `THOUGHT/LAB/FORMULA/questions/critical/q32_meaning_as_field.md`.

## What Phase 3 establishes (operational, falsifiable)

- **Third domain exists** (SNLI) and runs through the same falsifiers.
- **Cross-domain threshold transfer** works **without retuning** across **three domains** (SciFact, Climate-FEVER, SNLI).
- The above is **not a one-seed fluke** (multi-seed calibration + verification).
- **Full-mode stress variability gate** passes at higher `--stress_n` (SciFact streaming neighbor falsifier).

## What Phase 3 does NOT establish

- This does **not** prove "meaning is a physical field" (that is a separate claim and needs additional external/physical observables).
- This does **not** mark Q32 **ANSWERED**; it only satisfies the Phase 3 checklist gates in the roadmap.

## Evidence index (SHA256 pointers)

All artifacts are under `LAW/CONTRACTS/_runs/q32_public/datatrail/` and are also recorded in:
- `THOUGHT/LAB/FORMULA/questions/reports/Q32_NEIGHBOR_FALSIFIER_DATA_TRAIL.md`

### Phase 3 multi-seed transfer matrix (calibration_n=2, verify_n=2; full / crossencoder)

SciFact → Climate-FEVER:
- `p3m_transfer_scifact_to_climate_neighbor_full_20260110_011344.txt` = `061FF941BDB0D9F921F3E94452E057D44B03E7D399A82DE40BE744D8B3F923A0`
- `empirical_receipt_p3m_transfer_scifact_to_climate_neighbor_full_20260110_011344.json` = `BDAA611AE43DA4CBE221B86DAFEFAB946131808A96EAB0CD9348947A3249584D`
- `transfer_calibration_scifact_to_climate_full_20260110_011344.json` = `1B5397937208E27714626B7E33D59138B31FC2922404FB55CC1C204E817976B8`

SciFact → SNLI:
- `p3m_transfer_scifact_to_snli_neighbor_full_20260110_012137.txt` = `37B033E75593079EFD56D6435D67BE4195273DE3423D7DC967205477A9ADC58E`
- `empirical_receipt_p3m_transfer_scifact_to_snli_neighbor_full_20260110_012137.json` = `48FC2DF4893E2F142108CD5F5F3E4CDAD6EDA9BD80E4C86E0FBE7531743451C9`
- `transfer_calibration_scifact_to_snli_full_20260110_012137.json` = `F43B145A45D14AD2C976A1F685B9387AC3C41374318ADFBC026FD04888E026E8`

Climate-FEVER → SciFact:
- `p3m_transfer_climate_to_scifact_neighbor_full_20260110_013528.txt` = `73F2865A7DD94DB190B2DB782638984B363F91B64C485E2E87051978D335C70E`
- `empirical_receipt_p3m_transfer_climate_to_scifact_neighbor_full_20260110_013528.json` = `F15AA1165455CFACEBAEF9E0B2C4861D1FA8C7E80CC566A2673C946C9BB9B991`
- `transfer_calibration_climate_to_scifact_full_20260110_013528.json` = `652B97A77910A2D6EAA77B959660FB0FE220EFD8100E8963D688D18A4D73A5F9`

Climate-FEVER → SNLI:
- `p3m_transfer_climate_to_snli_neighbor_full_20260110_014350.txt` = `C40F86F515E43258D2CA96F43D25C8FF65BDBDAF1EC931E81157EDAEB5DE3B11`
- `empirical_receipt_p3m_transfer_climate_to_snli_neighbor_full_20260110_014350.json` = `0BD2C79E748BEF6C82F1D81FFC20788A6F4EC83C460B2A5348A8CC7E5B790998`
- `transfer_calibration_climate_to_snli_full_20260110_014350.json` = `1DE7A4A1C3E15D673E1A16544917BB0600C1B14CE6C27E225CE25426B6A51175`

SNLI → SciFact:
- `p3m_transfer_snli_to_scifact_neighbor_full_20260110_015622.txt` = `6E9A500B884B01D70FFF2670F01479CD83E6B4BCD781B37F584157F70DE8F037`
- `empirical_receipt_p3m_transfer_snli_to_scifact_neighbor_full_20260110_015622.json` = `86DEFAB2CB185EB241362FD781F46A19A70B99D7E55F149484F08A6B44561FBB`
- `transfer_calibration_snli_to_scifact_full_20260110_015622.json` = `09EC5BF56A4E0EE14DE6B3F57C40988832C187FA83541FA0EEBFDB2D29AAB862`

SNLI → Climate-FEVER:
- `p3m_transfer_snli_to_climate_neighbor_full_20260110_021118.txt` = `8E8A07A984F21D8E5C0546D9C92FA19D9526500B230E89583F7BDB311D9849F0`
- `empirical_receipt_p3m_transfer_snli_to_climate_neighbor_full_20260110_021118.json` = `60C762BBFFBEE8D09EBDA58FB0F57A22429878EA464133E238D26AF6EC49A65D`
- `transfer_calibration_snli_to_climate_full_20260110_021118.json` = `887C5708C4F8C2721B106741B576209EB4FB8B0877491751229D75EA0BDBA762`

### Phase 3 higher-n stress gate (full / crossencoder; SciFact streaming variability)

- `p3_stress_scifact_neighbor_full_v3_20260110_022541.txt` = `8D676F3D4F4C252912DD47D2B4EBDF868742C980E6C6C64220C32E260B62E7C8`
- `p3_stress_scifact_neighbor_full_v3_20260110_022541.txt.rc.txt` = `A9F58776A09B5DAC438049683F24BF85764E0FF8E7455952456165C68C158627`
- `empirical_receipt_p3_stress_scifact_neighbor_full_v3_20260110_022541.json` = `0023CF27EAD112AB9050EE8BABD728382C02A35240299FD0C0889AC29BE3CEE3`
- `stress_p3_scifact_neighbor_full_v3_20260110_022541.json` = `94A952EA35BF66872412C4FD99237229721477A20883336776C7C2F1DECC7DEA`

### Replication bundle (environment capture + internal hashes + exact rerun commands)

- `LAW/CONTRACTS/_runs/q32_public/datatrail/p3_replication_bundle_20260110_023257/SHA256SUMS.txt` = `030CA669C923D60AB72BF8DE0F9E63B2DE93C569465DF3FF9820C07355AF1F46`

Use `p3_replication_bundle_20260110_023257/README.txt` for the exact reproduction commands used for these gates.
