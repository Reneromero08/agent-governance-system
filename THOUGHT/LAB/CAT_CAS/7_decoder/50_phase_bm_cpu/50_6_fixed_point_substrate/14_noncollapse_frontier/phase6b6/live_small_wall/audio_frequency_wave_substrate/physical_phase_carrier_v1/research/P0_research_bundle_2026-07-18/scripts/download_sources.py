#!/usr/bin/env python3
"""Download direct official/open-access P0 research sources and write a receipt."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, ssl
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"
RECEIPT = ROOT / "DOWNLOAD_RECEIPT.json"
UA = "P0ResearchCustody/1.0 (+local reproducibility archive)"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def destination(record: dict) -> Path:
    group = "supplemental" if record["collection"] != "core_component_document" else "official"
    return ROOT / "sources" / group / record["local_filename"]

def download(record: dict, force: bool, timeout: int) -> dict:
    dest = destination(record)
    result = {"id": record["id"], "url": record.get("direct_download_url"), "path": str(dest.relative_to(ROOT))}
    if not result["url"]:
        result.update(status="MANUAL_REQUIRED", reason="No direct download URL in manifest")
        return result
    if record.get("download_mode") != "direct_pdf":
        result.update(status="MANUAL_REQUIRED", reason=f"download_mode={record.get('download_mode')}")
        return result
    if dest.exists() and not force:
        result.update(status="ALREADY_PRESENT", bytes=dest.stat().st_size, sha256=sha256(dest))
        return result
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(result["url"], headers={"User-Agent": UA, "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.5"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        context = ssl.create_default_context()
        with urlopen(req, timeout=timeout, context=context) as response, tmp.open("wb") as out:
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        prefix = tmp.read_bytes()[:5]
        if dest.suffix.lower() == ".pdf" and prefix != b"%PDF-":
            tmp.unlink(missing_ok=True)
            result.update(status="FAILED_NOT_PDF", final_url=final_url, content_type=content_type)
            return result
        os.replace(tmp, dest)
        actual = sha256(dest)
        legacy = record.get("legacy_expected_sha256")
        result.update(status="DOWNLOADED", final_url=final_url, content_type=content_type,
                      bytes=dest.stat().st_size, sha256=actual,
                      legacy_comparison=("MATCH" if legacy and actual == legacy else "DIFFERS" if legacy else "NO_LEGACY_HASH"))
        return result
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        result.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        return result

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="Download all direct-PDF records")
    p.add_argument("--id", action="append", default=[], help="Download one or more manifest IDs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--strict-legacy", action="store_true", help="Return failure if a downloaded file differs from a legacy hash")
    args = p.parse_args()
    if not args.all and not args.id:
        p.error("use --all or --id ID")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = [r for r in manifest["records"] if args.all or r["id"] in set(args.id)]
    unknown = sorted(set(args.id) - {r["id"] for r in selected})
    if unknown:
        p.error("unknown IDs: " + ", ".join(unknown))
    results = []
    for index, record in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {record['id']}", flush=True)
        results.append(download(record, args.force, args.timeout))
    receipt = {
        "schema": "p0.research-download-receipt.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "results": results,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = [r for r in results if r["status"].startswith("FAILED")]
    mismatches = [r for r in results if r.get("legacy_comparison") == "DIFFERS"]
    print(json.dumps({"downloaded": sum(r["status"] == "DOWNLOADED" for r in results),
                      "manual_required": sum(r["status"] == "MANUAL_REQUIRED" for r in results),
                      "failed": len(failed), "legacy_mismatches": len(mismatches),
                      "receipt": str(RECEIPT)}, indent=2))
    if failed or (args.strict_legacy and mismatches):
        return 2
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
