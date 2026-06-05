# PHASE2_MASTER_C_BIOS_MOD_DONORS

## Verdict

`PUBLIC_MOD_DONOR_FOUND`

Route C found public GA-970A-DS3P BIOS-mod donor workflows, but no donor image exists locally for byte diffing in the current lab folder.

## Local Search Result

Local Exp44 search did not find a diffable donor BIOS image for GA-970A-DS3P.

Found local categories:

- Stock/local BIOS dump and UEFIExtract parse tree in `cpu_hack/`
- Coreboot/AGESA source drops in ignored temporary trees
- Existing AGESA/P-state reports

Not found locally:

- GA-970A-DS3P NVMe donor image
- GA-970A-DS3P SLIC donor image
- GA-970A-DS3P hidden-menu donor image
- GA-970A-DS3P voltage/P-state donor image
- Any local donor-vs-stock binary pair suitable for structural diff

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
| NVMe | Yes | No | Good rebuild/checksum donor, not voltage-specific |
| SLIC | Query lead only | No | Rebuild donor only |
| Hidden menu | Query lead only | No | UI/NVRAM donor only |
| CPU support | Query lead only | No | Potential AGESA donor if exact revision found |
| AGESA | No exact local donor | No | Still needs exact board/revision pair |
| Voltage/P-state | No | No | No donor target found |

## Exact Artifacts Needed For A Real Donor Diff

1. Exact board revision and stock BIOS version matching the lab board.
2. Matching official stock image from Gigabyte for that revision.
3. Public mod image built from the same stock version.
4. Mod thread notes stating tool and operation used.
5. Hashes for stock and donor.
6. UEFIExtract parse reports for stock and donor.
7. Structural diff report limited to donor workflow and checksums.

## Route C Outcome

Public donor workflow exists, but no local donor image pair is available.

The next safe action is acquisition/review, not flashing:

- collect the exact donor image and stock-matching image into `cpu_hack/mod_donors/`
- record hashes and URLs
- parse both
- diff structure only

No donor image is authorized for boot or flash by this report.
