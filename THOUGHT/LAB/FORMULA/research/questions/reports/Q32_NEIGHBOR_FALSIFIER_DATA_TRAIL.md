# Q32 Neighbor-Falsifier (J) Data Trail

This report is an **audit-style datatrail** for Q32 work on the “neighbor falsifier / J = neighbor fitness” track.

Scope:
- Worktree: `D:\CCC 2.0\AI\wt-q32-next`
- Branch: `task/q32-next`
- This is authored research documentation under `THOUGHT/` (non-canon).

## What we built (high level)

1) A **harder falsifier** for the Q32 public harness:
- `--wrong_checks dissimilar` = easy topic-mismatch wrong checks (baseline).
- `--wrong_checks neighbor` = **nearest-neighbor competitor wrong checks** (“J-style”), plus reporting:
  - `details["mean_neighbor_sim"]`
  - console: `"[J] mean neighbor sim (k=...)=..."`

2) A **transfer / invariance test** (Phase 3) that calibrates once on one dataset and verifies on the other without retuning:
- `--mode transfer`: one direction
- `--mode matrix`: both directions
- `--calibration_n`, `--verify_n`: multi-seed calibration/verification

3) CPU performance knobs so we can iterate without hour-long runs:
- `--threads` for BLAS/torch threads
- `--ce_batch`, `--st_batch`

4) A **Phi-style coupling proxy** integrated into the public harness outputs:
- We report `phi_proxy_bits` which is `I(mu_hat; mu_check)` (mutual information) estimated via histogram binning.
- This is explicitly **not** canonical IIT Phi; it is a cheap, reproducible “veil-piercing” coupling/structure signal that
  pairs with `J` in the Interface Theory framing.

## Key commits (worktree)

- `5a1df48` — `Q32: add neighbor-fitness (J) diagnostics`
- `34ec923` — `Q32: make neighbor falsifier truth-inconsistent`

## Core finding (why “neighbor” matters)

The naive “neighbor” falsifier can accidentally pick “wrong checks” that still support the current claim (semantic closeness ≠ contradiction).
That creates false PASS or false FAIL depending on seed, which is exactly the opposite of an empirical gate.

So we made the neighbor falsifier **truth-inconsistent** in the SciFact bench by selecting competitor pools from **CONTRADICT-labeled** examples,
then selecting the candidate that minimizes the *actual* `M_wrong` under the intervention, rather than relying on a proxy.

## Multi-seed matrix failure → root cause → fix

### Symptom
Running full multi-seed matrix for neighbor mode:
- `--mode matrix --calibration_n 3 --verify_n 3 --wrong_checks neighbor --neighbor_k 10`
initially produced seed-dependent SciFact streaming failures (`SciFact-Streaming@seed=124/125`).

### Root cause
SciFact streaming is highly sensitive to which abstract sentences are sampled as the stream.
Different seeds changed which sentences were sampled (and which examples were selected), flipping the intervention effect sign.

### Fix applied (stability over seed variation)
We stabilized SciFact streaming selection to be **deterministic across seeds** by internally using `base_seed=123`
for the sampling order inside `run_scifact_streaming(...)`.

Trade-off:
- This removes seed variation **for SciFact streaming specifically**.
- It makes the public harness less flaky and more reproducible for transfer/matrix.
- If you want stochastic robustness later, we should add a dedicated “variability stress test” mode rather than tying it to `--seed`.

## Mechanical datatrail (verbatim logs + hashes)

The following files are written under the allowed artifact root:
- `LAW/CONTRACTS/_runs/q32_public/datatrail/`

### 2026-01-09 run bundle

Files:
- `LAW/CONTRACTS/_runs/q32_public/datatrail/status_20260109_153545.txt`
- `LAW/CONTRACTS/_runs/q32_public/datatrail/diff_20260109_153545.patch`
- `LAW/CONTRACTS/_runs/q32_public/datatrail/matrix_neighbor_20260109_153545.txt`

SHA256 (as printed by the runner):
- `status_20260109_153545.txt` = `B14DA729B0DECD9BBAC32FBB9414EECB2F1A84BD17C5E6ECCCBC20AC7A0CDA7B`
- `diff_20260109_153545.patch` = `497C7B39E83E51B8D3067C4703EA485A407F64AAAA6767E0DC6449F8BF68132C`
- `matrix_neighbor_20260109_153545.txt` = `3D1084D702100490AB7CD8826EC57C57A61B59DD64ED0FB6555E4643FBAB071E`

Command captured in `matrix_neighbor_20260109_153545.txt`:
- `python q32_public_benchmarks.py --mode matrix --scoring crossencoder --wrong_checks neighbor --neighbor_k 10 --threads 12 --device cpu --ce_batch 32 --st_batch 64 --calibration_n 3 --verify_n 3`

Matrix outcome (from the captured summary):
- All 12 results PASS (both directions, seeds 123/124/125):
  - `climate_fever->scifact:*@seed=123/124/125: PASS`
  - `scifact->climate_fever:*@seed=123/124/125: PASS`

### 2026-01-09 Phi-proxy matrix (neighbor mode)

Purpose:
- Record `J` (neighbor fitness) and `phi_proxy_bits` alongside the usual pass/fail matrix.

Failure artifact (pre-fix, kept for traceability):
- `LAW/CONTRACTS/_runs/q32_public/datatrail/matrix_neighbor_phi_20260109_192150.txt`
- SHA256 = `13549572B700471F12AC6727756C57A2140FAB5C5C9FDE736868E713FB97DE21`
- Failure: `NameError: mu_hat_list is not defined` in Climate-FEVER intervention.

Passing artifact (fast matrix, cosine scoring, neighbor wrong-checks):
- `LAW/CONTRACTS/_runs/q32_public/datatrail/matrix_neighbor_phi_fast_20260109_192754.txt`
- SHA256 = `0A654EEB10FC7CD8FDE7892E9F5471D23A0609970350D3BF2740AA4807D8B6F5`

EmpiricalMetricReceipt artifact (same fast matrix run class, includes gates + J + Phi-proxy):
- `LAW/CONTRACTS/_runs/q32_public/datatrail/empirical_receipt_matrix_neighbor_phi_fast_20260109_193751.json`
- SHA256 = `B60591D8AED74D7166B9330C79323CAEE4025A8F20505FB0C63DCEF45D9D57AA`
- Verbatim log: `LAW/CONTRACTS/_runs/q32_public/datatrail/matrix_neighbor_phi_receipt_fast_20260109_193751.txt`
- SHA256 = `3A5CFC598F4ECC5113CC5E9A8370233964EAD8471B1175AF67DDCA1106B95249`

### 2026-01-09 variability stress (SciFact streaming)

Purpose:
- Quantify brittleness when we *intentionally* vary SciFact streaming sampling (`scifact_stream_seed=-1`).

Artifacts:
- `LAW/CONTRACTS/_runs/q32_public/datatrail/stress_scifact_neighbor_fast_n10_20260109_180741.txt`
- `LAW/CONTRACTS/_runs/q32_public/datatrail/stress_scifact_neighbor_fast_n10_20260109_180741.json`
- `LAW/CONTRACTS/_runs/q32_public/datatrail/stress_scifact_neighbor_full_n3_20260109_181239.txt`
- `LAW/CONTRACTS/_runs/q32_public/datatrail/stress_scifact_neighbor_full_n3_20260109_181239.json`

SHA256:
- `stress_scifact_neighbor_fast_n10_20260109_180741.txt` = `04CA83E44F1436140FC6077A5C505696232C72D55E887065B07225E180CAC677`
- `stress_scifact_neighbor_fast_n10_20260109_180741.json` = `B1F70A2A8D809C53EEEF214BEAF57B93D2B1E0D6EA8FD431EEE7E430FF83C2A4`
- `stress_scifact_neighbor_full_n3_20260109_181239.txt` = `EC89AF26BFF8FE99A71B892735398520B8CDB37E34A8A9FE819350169BBC96A3`
- `stress_scifact_neighbor_full_n3_20260109_181239.json` = `527B58A4E189934623DE54D075F39C4A0BB9AFA711413B8E01A99C83D86AF2B4`

Notes:
- The **fast** stress run is expected to be noisy because `--fast` reduces `n` heavily.
- The stress mode is intentionally **non-strict** (it records PASS/FAIL distribution instead of aborting).

### 2026-01-09 stress receipt smoke (SciFact streaming)

Purpose:
- Prove stress mode can be captured in an `EmpiricalMetricReceipt` (including `R/M` end-of-stream summary stats).

Artifacts:
- `LAW/CONTRACTS/_runs/q32_public/datatrail/stress_receipt_smoke_20260109_202945.txt`
- SHA256 = `DC1C6DC673DD72E5514069EF1CB4CAA48BA5EC8BCC6D5E76913FB54451A1CBAE`
- `LAW/CONTRACTS/_runs/q32_public/datatrail/empirical_receipt_stress_smoke_20260109_202945.json`
- SHA256 = `04A53121276F9677140DE8D66CCB8197439052A682F99743C488B5CB1C937C3E`
- `LAW/CONTRACTS/_runs/q32_public/datatrail/stress_summary_smoke_20260109_202945.json`
- SHA256 = `8887FC37FB5F1D9BBDDFBAD74413BEB063A0E3914078D9B94E1AEDADA86E121C`

## Current working state

- There are **uncommitted edits** after `34ec923` (the stabilization change to SciFact streaming sampling).
- `git status` is recorded verbatim in the `status_20260109_153545.txt` artifact.

## Next hardening steps (recommended)

1) Commit the SciFact streaming stabilization (if you accept the “stability over seed-variance” trade-off).
2) Add a separate stress mode that *intentionally varies* the stream sampling and requires passing in aggregate
   (so we get both reproducibility and robustness, without conflating them).

---

## 2026-01-09 Phase 2 mechanism validation (ablations + inflation + swaps + sweeps)

Purpose:
- Close Phase 2 checklist items in `THOUGHT/LAB/FORMULA/research/questions/critical/q32_meaning_as_field.md` with receipted evidence.
- Add “must fail hard” negative controls and distributional robustness sweeps.

All artifacts below live under:
- `LAW/CONTRACTS/_runs/q32_public/datatrail/`

### SciFact bench (P2 intervention) — baseline PASS

- `p2_scifact_bench_neighbor_full_fast_20260109_215409.txt` = `B5B8A52FE1AB363191B4E197D3F7A3BE04AEFBF1755B60AC4807ACEA9DF24ACA`
- `empirical_receipt_p2_scifact_bench_neighbor_full_fast_20260109_215409.json` = `C7DF1A752666960711BDE4E056882F77F91F65E89174398A35AF2A9D8BE703C0`

### SciFact bench (P2 intervention) — ablation must kill

Hard-kill ablation (R=1 constant):
- `p2_scifact_bench_neighbor_no_grounding_fast_20260109_215638.txt` = `1B7A97157DACEF787675A148C237B8E182434D1BAC2BC85B86FE63763F8C5FF0`
- `empirical_receipt_p2_scifact_bench_neighbor_no_grounding_fast_20260109_215638.json` = `2384DF755799D46C59156A93A620F6500008304954AA5CDB8C4B52FA52E0ED2D`

Note (traceability):
- `no_scale` did not hard-kill the effect in fast mode:
  - `p2_scifact_bench_neighbor_no_scale_fast_20260109_215448.txt` = `3537B4C403EA712BEC0FB3E849356D69145DAC61D222683919A9E0BED5BA2073`
  - `empirical_receipt_p2_scifact_bench_neighbor_no_scale_fast_20260109_215448.json` = `18E63CC1F5CBFB755054092ECB174D32C49DF0BCF82D02AAC06CE95B06583287`

### SciFact bench (P2 intervention) — agreement inflation negative control must FAIL

- `p2_scifact_bench_inflation_fast_20260109_215724.txt` = `4BF56795C281253ACCB215A579D3BC98EB768D3B660B1B303E028C6C06B3BF20`
- `empirical_receipt_p2_scifact_bench_inflation_fast_20260109_215724.json` = `D95115A42B5BFC20C3FCF0D6DEA34AEF7D435DD886348832FDC75A4A5EAF2CDE`

### SciFact bench — depth proxy knob + no-depth ablation (Df/σ proxy)

- `p2_scifact_bench_neighbor_depth_power1_full_fast_20260109_215810.txt` = `EA5D0C18BC0AB6F626A46D6124ECC8E738198B51FB48B6812E9B80ABFF15C331`
- `empirical_receipt_p2_scifact_bench_neighbor_depth_power1_full_fast_20260109_215810.json` = `0AFDEE319C5F71A38FCD7587DDF9A18D21E7F54C3EABB4625CC02060FF83FA10`
- `p2_scifact_bench_neighbor_depth_power1_no_depth_fast_20260109_215853.txt` = `BE9FCCA7C342C4AFBDDFD3FD5F66D13BC052551B527DB5246B102FD4F70171F9`
- `empirical_receipt_p2_scifact_bench_neighbor_depth_power1_no_depth_fast_20260109_215853.json` = `94F1E5747B1CB1CC9F08474FF88B6BD963ACBE29ED51DB2274553C86BA038B05`

### Climate-FEVER streaming (P4 intervention) — baseline PASS vs inflation FAIL

Neighbor baseline:
- `p2_climate_stream_neighbor_full_fast_20260109_215932.txt` = `567DFF814D532EFCBFADDEF4CAF7E459F4C53557795FACFF0F568DC330BB67B7`
- `empirical_receipt_p2_climate_stream_neighbor_full_fast_20260109_215932.json` = `FBEA5596D6B0CBDA978B87148F855467656EDF93939968691139F4A4B5C3F424`

Inflation negative control:
- `p2_climate_stream_inflation_fast_20260109_220007.txt` = `6DBD3929E8ECC2FF288CCCBFA4EA6CD87CA6C6A98116ECA3020F6A8516342584`
- `empirical_receipt_p2_climate_stream_inflation_fast_20260109_220007.json` = `4203C18B59A652FD0598E6DC1603467C699A6ED27B84A20CD218D96477DBABAE`

### SciFact streaming — variability stress (pass-rate gate)

- `p2_scifact_stress_neighbor_fast_20260109_220056.txt` = `8C5FA723D612FC900615FECEC1577F28378EE32215BBEC8B33FB1C08C1181113`
- `empirical_receipt_p2_scifact_stress_neighbor_fast_20260109_220056.json` = `5F6036D167DFC7C787AA844568E79BB4BB82AFDA5A798509F45938F8181E515E`
- `stress_p2_scifact_neighbor_fast_20260109_220056.json` = `FA03242A1A2DDBBB64CDA1E83C1A1CDCD951FD116E48717266243244D9AB6C46`

### SciFact streaming — neighbor_k sweep (pass-rate across k)

- `p2_scifact_sweep_k_neighbor_fast_20260109_220958.txt` = `7494E8B4AA632C482F65ADC28E73D5D1C1C6D8055449112145D478EA5344CC23`
- `empirical_receipt_p2_scifact_sweep_k_neighbor_fast_20260109_220958.json` = `81AE12B27EF614C8BA93D9A9C3030744C5BA5E269E5884AAE3E555F3702D150C`
- `sweep_k_p2_scifact_neighbor_fast_20260109_220958.json` = `9B4CA279CC9989678818735C87AD6768BA02EAEEBD171F4F88F11B7983B28EC0`

---

## 2026-01-09 Phase 3 third domain (SNLI) + transfer (no retuning)

Purpose:
- Break the 2-dataset trap by adding a third public domain and repeating transfer without retuning.

### SNLI bench — neighbor PASS

- `p3_snli_bench_neighbor_full_fast_20260109_224219.txt` = `477CC5BC9F97801811E0DF342CE710F2E816EE2073754D6B65121DA9D7A087E4`
- `empirical_receipt_p3_snli_bench_neighbor_full_fast_20260109_224219.json` = `DB0118056C96DF8E63DC6C746BA19FCC6ACA1AD88B84260BCB874EF7BB13BFBC`

### SNLI bench — inflation FAIL (agreement inflation negative control)

- `p3_snli_bench_inflation_fast_20260109_224311.txt` = `BA7D917ECC0C8DA11B7D21737827BD4AFB75890B66DF9B99E60F350B4B7B35A3`
- `empirical_receipt_p3_snli_bench_inflation_fast_20260109_224311.json` = `326FFAD33BB9A8CD03E566CEF45573B2825C3867BA356DA3B7C189E6B6BEEE29`

### SNLI streaming — neighbor PASS

- `p3_snli_stream_neighbor_full_fast_20260109_224403.txt` = `5215D45B035A94E9C2DA0B3E4D0C2D658610B197853B3CDA985B2F20EEC00AEA`
- `empirical_receipt_p3_snli_stream_neighbor_full_fast_20260109_224403.json` = `F9CDF94287E3F1BA603FB94815A829998D9C493F453A2AA66C24CFBF1203BEFD`

### SNLI streaming — inflation FAIL (agreement inflation negative control)

- `p3_snli_stream_inflation_fast_20260109_224456.txt` = `F65C0A97898D72ED79F77B66DBEF4424BC73DB4E69C355EAD1C8E41F089DF4B5`
- `empirical_receipt_p3_snli_stream_inflation_fast_20260109_224456.json` = `BF157F3D55F10783F0028D84F0245ED6D5258957D917F3B9E654F3C696AAF992`

### Transfer (calibrate once on SciFact → verify on SNLI, no retuning)

- `p3_transfer_scifact_to_snli_neighbor_fast_20260109_224542.txt` = `F70C92CC0D9982B60AB5B1B41000C3D9947D19F9FCFA817DE2F779273758C1D1`
- `empirical_receipt_p3_transfer_scifact_to_snli_neighbor_fast_20260109_224542.json` = `E741434742521553096342F2CE3B8C6C4FB38585A23B9C432E22C37E9B9AAACE`
- `transfer_calibration_scifact_to_snli_20260109_224542.json` = `2305E8BC7C354C739FEE7BA481E30D08F04116DB84B1E9DC234B0C18740AB4D2`

### Transfer (full / crossencoder) — SciFact ↔ SNLI (no retuning)

SciFact → SNLI:
- `p3_transfer_scifact_to_snli_neighbor_full_20260109_225243.txt` = `3C5F4744805B8D0AD6178707C1480A6671C110431EDE4F7A89C31575FA83C0D9`
- `empirical_receipt_p3_transfer_scifact_to_snli_neighbor_full_20260109_225243.json` = `AC35C0AED467FA0C40F4C5D9FF2355FC98EE7C583741D2F5082CBEEBC74E9B59`
- `transfer_calibration_scifact_to_snli_full_20260109_225243.json` = `4AF9734B3296478CD41E8B6E44B44DE150F478FD2A133103D12BE06BCAAB8446`

SNLI → SciFact:
- `p3_transfer_snli_to_scifact_neighbor_full_20260109_230132.txt` = `6B644879DA4536795AB60103DC2352A60FDCAC29895292FBD13327B828CB92AA`
- `empirical_receipt_p3_transfer_snli_to_scifact_neighbor_full_20260109_230132.json` = `F8FE62793A21D617854059F12DA915B9BC62E0C506FAD9B25A94918F5BC6BB36`
- `transfer_calibration_snli_to_scifact_full_20260109_230132.json` = `7C6B3DF3A832A5535985678151A0C7995C86810BBBBD54062C621770C65BB195`

### Stress (full / crossencoder) — SciFact streaming variability (expectedly harsh)

This run FAILs the pass-rate gate (this is evidence we are not “settled” yet in full mode):
- `p3_stress_scifact_neighbor_full_20260109_232034.txt` = `579D19455BCA161241FA8F0161539E2576D228B315B8B621CAC82F9D6E3811DD`
- `empirical_receipt_p3_stress_scifact_neighbor_full_20260109_232034.json` = `BE4F4370DD1CE442064085DB66FAFE8A42C6773186C42E86FC562CB08BAFA8E4`
- `stress_p3_scifact_neighbor_full_20260109_232034.json` = `9F0150AE29C13A9A2C91C2C904CEED1F071AAEA4D9CDF58B902BE074A1B63042`

### Stress (full / crossencoder) — SciFact streaming variability (fixed neighbor selection, now stable)

This rerun uses the same stress settings but selects the neighbor falsifier using `M_from_R(R_grounded(...))` instead of mean cross-score:
- `p3_stress_scifact_neighbor_full_v2_20260110_002134.txt` = `723D37707E60A389D86243C18722C25E46EA0B72590BC2EE0D4914F49FB91A39`
- `empirical_receipt_p3_stress_scifact_neighbor_full_v2_20260110_002134.json` = `EB3722926111737749619522A5E237CB05E655C47F6365932828B8EC6863879D`
- `stress_p3_scifact_neighbor_full_v2_20260110_002134.json` = `5793DF2E403A28306E42326C89EE970A7ACE60B4BBF8CC4DF0FB656B5219DD14`

### Transfer (full / crossencoder) — remaining pairs to complete the 3-domain matrix by chunks

SciFact → Climate-FEVER:
- `p3_transfer_scifact_to_climate_neighbor_full_20260110_003022.txt` = `DE937087425918E42BFC2CE9D23410AB2FAEC8CE4AD4D500EB3AA24FECD1EB8A`
- `empirical_receipt_p3_transfer_scifact_to_climate_neighbor_full_20260110_003022.json` = `C35C1082FB113D8BC2FEC34B6CE7AD33EEA02847B2F8FA6FBF6D7AAE1DFB3DDB`
- `transfer_calibration_scifact_to_climate_full_20260110_003022.json` = `94D977358F0A76C78824ABBA0B84FD1D46D7AA14E2D7753381226C0DD4C9657F`

Climate-FEVER → SciFact:
- `p3_transfer_climate_to_scifact_neighbor_full_20260110_003820.txt` = `DFD56FFC1CEE300C4A320292713FF9EDB8503627E3417AC6D4FE141893CD3F21`
- `empirical_receipt_p3_transfer_climate_to_scifact_neighbor_full_20260110_003820.json` = `DF98809B39591985733E3578325EF147B63A74731806466D0C1A00C054E0D175`
- `transfer_calibration_climate_to_scifact_full_20260110_003820.json` = `355A43C448C5361008AEA5F57D3FA648B5DC294E4D9BA237F295B83ACAAB37FC`

Climate-FEVER → SNLI:
- `p3_transfer_climate_to_snli_neighbor_full_20260110_004506.txt` = `D59D07F91F20EFB3156869EF2D1E696CECCEFE556C3862D1CDFE17E67C51DEBA`
- `empirical_receipt_p3_transfer_climate_to_snli_neighbor_full_20260110_004506.json` = `55BD5DC08553D3F6A30C95A81FCF6BA067583B833A53C336F05BE35DC800A0B8`
- `transfer_calibration_climate_to_snli_full_20260110_004506.json` = `8C20E9EF391106595141824EC28E73D29A8883C58CAEF5368CAE6055F4601EC`

SNLI → Climate-FEVER:
- `p3_transfer_snli_to_climate_neighbor_full_20260110_005742.txt` = `8F7CB46A002BACF6E8ACA686E8CF03190F5141997D625D917CA11DE73D09E7E2`
- `empirical_receipt_p3_transfer_snli_to_climate_neighbor_full_20260110_005742.json` = `E9A899CB7A6513900C5AB3B8FE23F59C2677B771A435D36E3DE086A17A787E8C`
- `transfer_calibration_snli_to_climate_full_20260110_005742.json` = `5D9FF6334472FE7E78465BFFC2E400D4D0AE46D69623D137F6A621C7356269DD`

---

## 2026-01-10 Phase 3 multi-seed transfer matrix (calibration_n=2, verify_n=2; full / crossencoder)

Purpose:
- Prove the Phase-3 "transfer without retuning" result is not a one-seed fluke by repeating all 3-domain ordered pairs with multiple seeds.

SciFact  Climate-FEVER:
- `p3m_transfer_scifact_to_climate_neighbor_full_20260110_011344.txt` = `061FF941BDB0D9F921F3E94452E057D44B03E7D399A82DE40BE744D8B3F923A0`
- `empirical_receipt_p3m_transfer_scifact_to_climate_neighbor_full_20260110_011344.json` = `BDAA611AE43DA4CBE221B86DAFEFAB946131808A96EAB0CD9348947A3249584D`
- `transfer_calibration_scifact_to_climate_full_20260110_011344.json` = `1B5397937208E27714626B7E33D59138B31FC2922404FB55CC1C204E817976B8`

SciFact  SNLI:
- `p3m_transfer_scifact_to_snli_neighbor_full_20260110_012137.txt` = `37B033E75593079EFD56D6435D67BE4195273DE3423D7DC967205477A9ADC58E`
- `empirical_receipt_p3m_transfer_scifact_to_snli_neighbor_full_20260110_012137.json` = `48FC2DF4893E2F142108CD5F5F3E4CDAD6EDA9BD80E4C86E0FBE7531743451C9`
- `transfer_calibration_scifact_to_snli_full_20260110_012137.json` = `F43B145A45D14AD2C976A1F685B9387AC3C41374318ADFBC026FD04888E026E8`

Climate-FEVER  SciFact:
- `p3m_transfer_climate_to_scifact_neighbor_full_20260110_013528.txt` = `73F2865A7DD94DB190B2DB782638984B363F91B64C485E2E87051978D335C70E`
- `empirical_receipt_p3m_transfer_climate_to_scifact_neighbor_full_20260110_013528.json` = `F15AA1165455CFACEBAEF9E0B2C4861D1FA8C7E80CC566A2673C946C9BB9B991`
- `transfer_calibration_climate_to_scifact_full_20260110_013528.json` = `652B97A77910A2D6EAA77B959660FB0FE220EFD8100E8963D688D18A4D73A5F9`

Climate-FEVER  SNLI:
- `p3m_transfer_climate_to_snli_neighbor_full_20260110_014350.txt` = `C40F86F515E43258D2CA96F43D25C8FF65BDBDAF1EC931E81157EDAEB5DE3B11`
- `empirical_receipt_p3m_transfer_climate_to_snli_neighbor_full_20260110_014350.json` = `0BD2C79E748BEF6C82F1D81FFC20788A6F4EC83C460B2A5348A8CC7E5B790998`
- `transfer_calibration_climate_to_snli_full_20260110_014350.json` = `1DE7A4A1C3E15D673E1A16544917BB0600C1B14CE6C27E225CE25426B6A51175`

SNLI  SciFact:
- `p3m_transfer_snli_to_scifact_neighbor_full_20260110_015622.txt` = `6E9A500B884B01D70FFF2670F01479CD83E6B4BCD781B37F584157F70DE8F037`
- `empirical_receipt_p3m_transfer_snli_to_scifact_neighbor_full_20260110_015622.json` = `86DEFAB2CB185EB241362FD781F46A19A70B99D7E55F149484F08A6B44561FBB`
- `transfer_calibration_snli_to_scifact_full_20260110_015622.json` = `09EC5BF56A4E0EE14DE6B3F57C40988832C187FA83541FA0EEBFDB2D29AAB862`

SNLI  Climate-FEVER:
- `p3m_transfer_snli_to_climate_neighbor_full_20260110_021118.txt` = `8E8A07A984F21D8E5C0546D9C92FA19D9526500B230E89583F7BDB311D9849F0`
- `empirical_receipt_p3m_transfer_snli_to_climate_neighbor_full_20260110_021118.json` = `60C762BBFFBEE8D09EBDA58FB0F57A22429878EA464133E238D26AF6EC49A65D`
- `transfer_calibration_snli_to_climate_full_20260110_021118.json` = `887C5708C4F8C2721B106741B576209EB4FB8B0877491751229D75EA0BDBA762`

---

## 2026-01-10 Phase 3 higher-n stress gate (full / crossencoder; SciFact streaming variability)

Purpose:
- Increase `--stress_n` and keep the pass-rate gate hard (not a single lucky run).

- `p3_stress_scifact_neighbor_full_v3_20260110_022541.txt` = `8D676F3D4F4C252912DD47D2B4EBDF868742C980E6C6C64220C32E260B62E7C8`
- `p3_stress_scifact_neighbor_full_v3_20260110_022541.txt.rc.txt` = `A9F58776A09B5DAC438049683F24BF85764E0FF8E7455952456165C68C158627`
- `empirical_receipt_p3_stress_scifact_neighbor_full_v3_20260110_022541.json` = `0023CF27EAD112AB9050EE8BABD728382C02A35240299FD0C0889AC29BE3CEE3`
- `stress_p3_scifact_neighbor_full_v3_20260110_022541.json` = `94A952EA35BF66872412C4FD99237229721477A20883336776C7C2F1DECC7DEA`

---

## 2026-01-10 Phase 3 pinned replication bundle (environment + rerun commands + internal SHA256SUMS)

Bundle directory:
- `LAW/CONTRACTS/_runs/q32_public/datatrail/p3_replication_bundle_20260110_023257/`

Bundle hash index:
- `p3_replication_bundle_20260110_023257/SHA256SUMS.txt` = `030CA669C923D60AB72BF8DE0F9E63B2DE93C569465DF3FF9820C07355AF1F46`

The bundle includes:
- `python_version.txt`, `python_platform.txt`
- `pip_version.txt`, `pip_freeze.txt`
- `git_branch.txt`, `git_head.txt`, `git_status_porcelain.txt`
- `README.txt` (exact rerun commands)

---

## 2026-01-10 Phase 4 geometry-break “tipping test” (proxy; full / crossencoder)

Purpose:
- Add an independent, receipted geometry signal alongside `M=log(R)` on the same public SciFact streaming intervention.
- Gate: geometry separates truth-consistent checks vs wrong checks (neighbor falsifier) at end-step.

Artifacts:
- `p4_geom_tipping_scifact_neighbor_full_20260110_033745.txt` = `C20164849494D258A568F2C65D94A7695C05934DD16103B66F701359160505AC`
- `p4_geom_tipping_scifact_neighbor_full_20260110_033745.txt.rc.txt` = `A9F58776A09B5DAC438049683F24BF85764E0FF8E7455952456165C68C158627`
- `empirical_receipt_p4_geom_tipping_scifact_neighbor_full_20260110_033745.json` = `994065D2E7EF175EEE074E738EA0D6F4752011C0D9786906F58038D04DD5CD36`
- `geometry_p4_geom_tipping_scifact_neighbor_full_20260110_033745.json` = `3C6CC92D27641D78DDC5E4E1F16D2543B82F7F582123BF6508C8CC4E6CC8CEA2`

---

## 2026-01-10 Phase 4 streaming + independence + phase-boundary + QGTL (full / crossencoder)

### SciFact (full / crossencoder; QGTL backend; strict gates)

Command:
- `python THOUGHT/LAB/FORMULA/experiments/open_questions/q32/q32_public_benchmarks.py --mode stream --dataset scifact --scoring crossencoder --threads 12 --device cpu --geometry_backend qgtl --require_geometry_gate --require_phase_boundary_gate --phase_min_stable_rate 0.55 --require_injection_gate --strict --empirical_receipt_out LAW/CONTRACTS/_runs/q32_public/datatrail/empirical_receipt_p4_scifact_full_20260110_044438.json --geometry_out LAW/CONTRACTS/_runs/q32_public/datatrail/geometry_p4_scifact_full_20260110_044438.json --stream_series_out LAW/CONTRACTS/_runs/q32_public/datatrail/series_p4_scifact_full_20260110_044438.json`

Artifacts:
- `scifact_p4_full_20260110_044438.log.txt` = `7EC64C52ACCAB757080EC1732F75BFA341E8829B7AFF96B9E7569F99AA793527`
- `scifact_p4_full_20260110_044438.rc.txt` = `13BF7B3039C63BF5A50491FA3CFD8EB4E699D1BA1436315AEF9CBE5711530354`
- `empirical_receipt_p4_scifact_full_20260110_044438.json` = `C7E428FE396286E025CB990A5C218F1F9EFDF1C09D9254017E8351C2FF2CC074`
- `geometry_p4_scifact_full_20260110_044438.json` = `2736389536B624325A4B5A13EF624771C88295308817147A7FF2A63EF52D34FF`
- `series_p4_scifact_full_20260110_044438.json` = `9E85EE3E1F06A5267F2EA5200B74FAE83CC9B843579809F7CA00C124D1C59D0E`

### Climate-FEVER (full / crossencoder; strict gates)

Notes:
- Climate-FEVER has ~5 evidence items/claim, so the correlated stream uses within-evidence n-gram chunks (correlated), and independent checks are other supportive evidence sentences for the same claim.

Command:
- `python THOUGHT/LAB/FORMULA/experiments/open_questions/q32/q32_public_benchmarks.py --mode stream --dataset climate_fever --scoring crossencoder --threads 12 --device cpu --require_phase_boundary_gate --phase_min_tail 2 --phase_min_stable_rate 0.45 --strict --empirical_receipt_out LAW/CONTRACTS/_runs/q32_public/datatrail/empirical_receipt_p4_climate_full_20260110_050232.json --geometry_out LAW/CONTRACTS/_runs/q32_public/datatrail/geometry_p4_climate_full_20260110_050232.json --stream_series_out LAW/CONTRACTS/_runs/q32_public/datatrail/series_p4_climate_full_20260110_050232.json`

Artifacts:
- `climate_p4_full_20260110_050232.log.txt` = `50F6E4CA5537A72116F4B697697C825C8B23FC11EE862577F7BB647ED0B6115C`
- `climate_p4_full_20260110_050232.rc.txt` = `13BF7B3039C63BF5A50491FA3CFD8EB4E699D1BA1436315AEF9CBE5711530354`
- `empirical_receipt_p4_climate_full_20260110_050232.json` = `9C267FBA0B833ADD151D4665D0C274DEF5A19B13CB6835CB2793ED75A6BB85BA`
- `geometry_p4_climate_full_20260110_050232.json` = `D590FAA16F669CE24689D800CE1DBCEDF67BC4AC2989E0C97106FFF1338AAA39`
- `series_p4_climate_full_20260110_050232.json` = `C0E2DAA5EDC209B860C0B8839B64F7B2B71DC0670F51B96F34F1ED2827206DC0`

---

## 2026-01-10 Phase 5 (start) — 4th domain scaffold (MNLI) + initial transfer smoke (fast / cosine)

### MNLI benchmark (fast / cosine)

Command:
- `python THOUGHT/LAB/FORMULA/experiments/open_questions/q32/q32_public_benchmarks.py --mode bench --dataset mnli --fast --scoring cosine --threads 12 --device cpu --empirical_receipt_out LAW/CONTRACTS/_runs/q32_public/datatrail/empirical_receipt_p5_mnli_bench_fast_20260110_051326.json`

Artifacts:
- `p5_mnli_bench_fast_20260110_051326.log.txt` = `9C937C4E60CB09DB7802D9A6D637AD4F4E20D1A2E8970864C78427281B5A2BD4`
- `p5_mnli_bench_fast_20260110_051326.rc.txt` = `13BF7B3039C63BF5A50491FA3CFD8EB4E699D1BA1436315AEF9CBE5711530354`
- `empirical_receipt_p5_mnli_bench_fast_20260110_051326.json` = `80D8BBCEDAD0C655F0EF198317AE23880C1D68597622B43F0104DCFE02DAD6EA`

### Transfer smoke: SciFact -> MNLI (fast / cosine; calibration_n=1, verify_n=1)

Command:
- `python THOUGHT/LAB/FORMULA/experiments/open_questions/q32/q32_public_benchmarks.py --mode transfer --calibrate_on scifact --apply_to mnli --calibration_n 1 --verify_n 1 --fast --scoring cosine --threads 12 --device cpu --empirical_receipt_out LAW/CONTRACTS/_runs/q32_public/datatrail/empirical_receipt_p5_transfer_scifact_to_mnli_fast_20260110_051920.json --calibration_out LAW/CONTRACTS/_runs/q32_public/datatrail/transfer_calibration_scifact_to_mnli_fast_20260110_051920.json`

Artifacts:
- `p5_transfer_scifact_to_mnli_fast_20260110_051920.log.txt` = `A640DE75C4060BAAF16548B3646D9DBED741B53BDCA797D55EF82361B432DACC`
- `p5_transfer_scifact_to_mnli_fast_20260110_051920.rc.txt` = `13BF7B3039C63BF5A50491FA3CFD8EB4E699D1BA1436315AEF9CBE5711530354`
- `empirical_receipt_p5_transfer_scifact_to_mnli_fast_20260110_051920.json` = `52CEAA5DAD4F44D2EFCB6C69C25BE84FAB08526D6EB33FCA4674926235C197EE`
- `transfer_calibration_scifact_to_mnli_fast_20260110_051920.json` = `4091FB173192E6033881BA0175FF8F07D63FB041C5E6B2FC8BC2C3124F9C321A`

---

## 2026-01-10 Phase 5.1 (fast) — 4-domain transfer matrix (12 ordered pairs; cosine; calibration_n=1, verify_n=1)

This is a fast, low-compute matrix run to expose brittle directions before running the full-mode matrix.

Summary index (contains per-pair commands + per-pair log/receipt/calibration SHA256):
- `p5_matrix4_fast_20260110_052808.summary.txt` = `5D2D238E7E28240D90F17B76A8FD2726F181D1C97A65F5E550DD1BF445FF9D36`

Observed failures in this fast matrix (by receipt `passed=false`):
- `empirical_receipt_p5_transfer_scifact_to_climate_fever_fast_20260110_052808.json`: `scifact->climate_fever:Climate-FEVER-Streaming@seed=123`
- `empirical_receipt_p5_transfer_snli_to_climate_fever_fast_20260110_052808.json`: `snli->climate_fever:Climate-FEVER-Streaming@seed=123`
- `empirical_receipt_p5_transfer_mnli_to_climate_fever_fast_20260110_052808.json`: `mnli->climate_fever:Climate-FEVER-Streaming@seed=123`
