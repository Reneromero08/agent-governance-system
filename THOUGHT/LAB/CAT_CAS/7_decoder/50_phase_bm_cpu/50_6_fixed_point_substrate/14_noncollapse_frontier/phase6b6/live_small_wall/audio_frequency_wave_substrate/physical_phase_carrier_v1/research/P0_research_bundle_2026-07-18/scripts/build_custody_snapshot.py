#!/usr/bin/env python3
"""Build a timestamp-free metadata snapshot of the private source cache.

The output contains hashes and document metadata, never third-party bytes.  It is
safe to bind into the P0 candidate while the downloaded files remain ignored.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"
LINKS = ROOT / "DOWNLOAD_LINKS.md"
RECEIPT = ROOT / "DOWNLOAD_RECEIPT.json"
OUTPUT = ROOT / "SOURCE_CUSTODY.json"
SOURCE_COMMIT = "cb53976612cbe83bec82df826a9889418f7e0b89"

PUBLISHERS = {
    "analog.com": "Analog Devices",
    "arxiv.org": "arXiv",
    "components.omron.com": "Omron",
    "doi.org": "Elsevier DOI record",
    "download.epsondevice.com": "Epson",
    "epsondevice.com": "Epson",
    "littelfuse.com": "Littelfuse",
    "murata.com": "Murata",
    "nexperia.com": "Nexperia",
    "pmc.ncbi.nlm.nih.gov": "U.S. National Library of Medicine PubMed Central",
    "sensirion.com": "Sensirion",
    "siglent.com": "SIGLENT",
    "siglentna.com": "SIGLENT",
    "spectrum-instrumentation.com": "Spectrum Instrumentation",
    "st.com": "STMicroelectronics",
    "ti.com": "Texas Instruments",
    "vishay.com": "Vishay",
    "yageo.com": "Yageo / KEMET",
}

REVISION_IDENTITIES = {
    "ADG1419_REV_A": "Rev. A; publication date not asserted",
    "ADR45XX_REV_G": "Rev. G; publication date not asserted",
    "ADUM140D_REV_K": "Rev. K; publication date not asserted",
    "ADXL354_REV_D": "Rev. D; publication date not asserted",
    "EPSON_FC135": "Q13FC13500004 exact-product brief; publication date not asserted",
    "KEMET_C0G_CURRENT": "current exact-product data; revision/date not asserted",
    "LITTELFUSE_0467_CURRENT": "current 467-series exact-product page; revision/date not asserted",
    "MURATA_GJM_CURRENT": "current exact-product data; revision/date not asserted",
    "MURATA_GRM_CURRENT": "current exact-product data; revision/date not asserted",
    "MURATA_MEV1_CURRENT": "current exact-product data; revision/date not asserted",
    "NEXPERIA_1N4148": "legacy 1N4148W,115 source record; exact current first-party page unavailable; candidate substitution not authorized",
    "NEXPERIA_2N7002": "captured 2N7002PW data sheet; Nexperia marks the type Not for Design In; replacement not yet selected",
    "NEXPERIA_BZT52": "current official PDF captured 2026-07-18; internal revision/date not asserted",
    "OMRON_G6K_2026_06_01": "official product-page update 2026-06-01",
    "OPA810": "SBOS799E; August 2019, revised August 2024",
    "SHT4X_2025": "PDF Version 7.1, March 2025; product-page label 04/2025",
    "SIGLENT_DATASHEET": "current global listing EN01I, 2025-03-18; North America marks SDG1032X obsolete; lifecycle is region-dependent; retained legacy hash is historical",
    "SIGLENT_MANUAL": "current global listing EN01J, 2025-09-22; North America marks SDG1032X obsolete; lifecycle is region-dependent; retained legacy hash is historical",
    "SIGLENT_PROGRAMMING": "current global listing E05C, 2026-06-30; North America marks SDG1032X obsolete; lifecycle is region-dependent; retained legacy hash is historical",
    "SPECTRUM_DATASHEET": "product-page listing 2026-05-19; official direct URL yields bytes matching the retained legacy hash; no distinct newer byte revision asserted",
    "SPECTRUM_MANUAL": "current product-page listing 2026-05-19; driver 7.010; retained legacy hash is historical until recaptured",
    "ST_UM2591_REV2": "Rev. 2, April 2026",
    "VISHAY_CRHV": "document 68002, 2024-11-12",
    "VISHAY_TNPW": "TNPW e3 document 28758, revision 10-Apr-2026; document 31006 is a different lead-bearing family",
    "ST_NUCLEO_G031K8_D01_RESOURCES": "MB1455-G031K8-D01 current resource set; date not asserted",
    "GHZ_OPTOMECH_AIR_2011": "2011 open-access preprint",
    "QTF_PASSIVE_ELECTRICAL_DAMPING_2021": "2021 open-access journal article",
    "QTF_RESONANCE_TRACKING_2019": "2019 open-access journal article",
    "QTF_VOLTAGE_INDUCED_SHIFT_2014": "2014 open-access journal article",
    "QTF_VOLTAGE_MODE_READOUT_2023": "2023 open-access journal article",
    "SILICON_PHONONIC_SLAB_2011": "2011 DOI record; full text not captured",
    "THIN_FILM_QUARTZ_PHONONIC_2024": "2024 open-access preprint",
    "UNIFIED_QTF_THEORY_2026": "2026 arXiv preprint; peer review and independent replication not asserted",
    "ADG1419_SIMULATION_MODELS": "current official model resources; an Analog Devices support thread reports an LTspice 26.0.1 issue; exact model/simulator version and data-sheet conformance must be qualified",
    "OPA810_SIMULATION_MODELS": "current official model resources; revision/date not asserted",
}


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def destination(record: dict[str, object]) -> Path:
    group = "official" if record["collection"] == "core_component_document" else "supplemental"
    return ROOT / "sources" / group / str(record["local_filename"])


def publisher(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    for domain, name in PUBLISHERS.items():
        if host == domain or host.endswith("." + domain):
            return name
    return host or "publisher not asserted"


def link_metadata() -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in LINKS.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"### ([A-Z0-9_]+): (.+)", line)
        if match:
            current = {"title": match.group(2)}
            records[match.group(1)] = current
        elif current is not None and line.startswith("- Role: "):
            current["role"] = line.removeprefix("- Role: ")
        elif current is not None and line.startswith("- Why it matters: "):
            current["relevance"] = line.removeprefix("- Why it matters: ")
    return records


def main() -> int:
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    outcomes = {item["id"]: item for item in receipt["results"]}
    details = link_metadata()
    records: list[dict[str, object]] = []
    for source in manifest["records"]:
        identity = source["id"]
        outcome = outcomes[identity]
        local_bytes = outcome.get("bytes") if outcome["status"] in {"DOWNLOADED", "ALREADY_PRESENT"} else None
        local_hash = outcome.get("sha256") if local_bytes is not None else None
        if local_bytes is not None:
            path = destination(source)
            if not path.is_file() or path.stat().st_size != local_bytes or file_sha256(path) != local_hash:
                raise ValueError(f"private-cache custody mismatch: {identity}")
            if source.get("legacy_expected_sha256") and local_hash != source["legacy_expected_sha256"]:
                custody = "LOCAL_CURRENT_BYTES_CAPTURED__LEGACY_DIFFERS"
            else:
                custody = "LOCAL_BYTES_CAPTURED_AND_HASH_VERIFIED"
        elif source.get("legacy_expected_sha256"):
            custody = "URL_AND_LEGACY_HASH_RECORDED__BYTES_NOT_LOCAL"
        elif outcome["status"] in {"MANUAL_REQUIRED", "FAILED", "FAILED_NOT_PDF"}:
            custody = "MANUAL_CAPTURE_REQUIRED"
        else:
            custody = "PROSPECTIVE_IDENTITY_ONLY"
        url = source.get("official_product_page") or source.get("direct_download_url")
        reason = outcome.get("reason")
        if reason is None and outcome["status"] == "FAILED_NOT_PDF":
            reason = "direct URL resolved to non-PDF content"
        if reason is None and outcome["status"] == "MANUAL_REQUIRED":
            reason = "manual product-page or acceptance-gated capture required"
        if reason is None:
            reason = "none"
        meta = details.get(identity, {})
        records.append({
            "collection": source["collection"],
            "current_bytes": local_bytes,
            "current_sha256": local_hash,
            "custody_state": custody,
            "direct_download_url": source.get("direct_download_url"),
            "download_result": outcome["status"],
            "download_result_detail": reason,
            "final_redirect_url": outcome.get("final_url"),
            "legacy_expected_bytes": source.get("legacy_expected_bytes"),
            "legacy_expected_sha256": source.get("legacy_expected_sha256"),
            "license_or_redistribution_note": (
                "open-access location recorded; license terms not independently verified"
                if source["collection"] == "scientific_background"
                else "public metadata recorded; redistribution permission for document/model bytes not asserted"
            ),
            "local_filename": source["local_filename"],
            "official_product_page": source.get("official_product_page"),
            "publisher": publisher(str(url or "")),
            "relevance_to_p0": meta.get("relevance", "relevance not asserted beyond manifest collection"),
            "revision_and_date": REVISION_IDENTITIES[identity],
            "role": meta.get("role", "role not separately asserted"),
            "source_id": identity,
            "title": meta.get("title", identity),
        })
    counts = dict(sorted(Counter(item["custody_state"] for item in records).items()))
    snapshot = {
        "custody_state_counts": counts,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "records": records,
        "schema": "p0.research-source-custody.v1",
        "source_commit": SOURCE_COMMIT,
        "source_record_count": len(records),
        "third_party_byte_policy": "metadata and hashes may be committed; downloaded document/model bytes remain private and uncommitted",
    }
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_bytes(canonical(snapshot))
    temporary.replace(OUTPUT)
    print(canonical({"output": OUTPUT.name, "records": len(records), "state_counts": counts}).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
