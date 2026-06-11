# PHASE2_MASTER_C_BIOS_MOD_DONORS

## Verdict

`PUBLIC_MOD_DONOR_DIFFED`

Route C found public GA-970A-DS3P BIOS-mod donor workflows and acquired a local diffable stock/donor pair. The donor workflow is useful for rebuild/checksum structure only; it does not create a voltage/P4 candidate.

## Local Search Result

Local Exp44 now contains a diffable donor BIOS image pair for GA-970A-DS3P rev. 1.0 F2j.

Found local categories:

- Stock/local BIOS dump and UEFIExtract parse tree in `cpu_hack/`
- Official F2j stock package and image in `cpu_hack/mod_donors/`
- Public F2j NVMe donor package and image in `cpu_hack/mod_donors/`
- UEFIExtract parse reports for both stock and donor images
- Coreboot/AGESA source drops in ignored temporary trees
- Existing AGESA/P-state reports

Not found locally:

- GA-970A-DS3P SLIC donor image
- GA-970A-DS3P hidden-menu donor image
- GA-970A-DS3P voltage/P-state donor image
- Any voltage/P-state donor-vs-stock binary pair suitable for P4 edit derivation

## Public Donor Leads

| Lead | Classification | Use |
|---|---|---|
| Win-Raid / Level1Techs GA-970A-DS3P rev. 1.0 NVMe-mod thread | NVMe workflow donor | Study successful Aptio/AMI module insertion, checksum handling, and parse/rebuild workflow. Do not treat as flash candidate. |
| Gigabyte official GA-970A-DS3P support BIOS pages | Stock baseline source | Compare official version lineage and board revision boundaries. |
| BIOS-mod forum patterns for SLIC/hidden menu mods | Workflow donor only | Useful for rebuild/checksum behavior; not relevant as voltage target without exact board/revision match. |

Public pages consulted:

- `https://winraid.level1techs.com/t/offer-gigabyte-ga-970a-ds3p-rev-1-0-nvme-mod/33089`
- `https://www.gigabyte.com/Motherboard/GA-970A-DS3P-rev-2x/support`

## Classification

| Mod class | Found public lead | Local diffable image | Relevance |
|---|---:|---:|---|
| NVMe | Yes | Yes | Diffed workflow donor; not voltage-specific |
| SLIC | Query lead only | No | Rebuild donor only |
| Hidden menu | Query lead only | No | UI/NVRAM donor only |
| CPU support | Query lead only | No | Potential AGESA donor if exact revision found |
| AGESA | No exact local donor | No | Still needs exact board/revision pair |
| Voltage/P-state | No | No | No donor target found |

## Diffed Artifacts

Full report:

`cpu_sing_3/PHASE2_DONOR_DIFF_REPORT.md`

Local artifacts:

- `cpu_hack/mod_donors/gigabyte_rev1_stock_F2j.zip`
- `cpu_hack/mod_donors/gigabyte_rev1_stock_F2j_extracted/970ADS3P.F2j`
- `cpu_hack/mod_donors/gigabyte_rev1_stock_F2j_extracted/970ADS3P.F2j.report.txt`
- `cpu_hack/mod_donors/winraid_970ADS3PNVME.zip`
- `cpu_hack/mod_donors/winraid_970ADS3PNVME_extracted/970ADS3PNVME.F2j`
- `cpu_hack/mod_donors/winraid_970ADS3PNVME_extracted/970ADS3PNVME.F2j.report.txt`

Structural result:

- first changed offset `0x002C58A0`
- last changed offset `0x002CA9FF`
- donor inserts `NvmExpressDxe_4` into existing free space
- later volumes remain byte-identical

## Route C Outcome

Public donor workflow is now locally diffed. The next safe firmware action is not donor acquisition. It is still P4-only edit-source proof in the owned `AmdProcessorInitPeim` route.

No donor image is authorized for boot or flash by this report.
