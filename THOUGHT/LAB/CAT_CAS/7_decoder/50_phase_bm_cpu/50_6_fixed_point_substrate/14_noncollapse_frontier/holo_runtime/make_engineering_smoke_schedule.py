#!/usr/bin/env python3
"""Create a non-scientific three-window executor smoke schedule."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("output",type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 sid="ENGINEERING_SMOKE_TEST";common={"session_id":sid,"block_id":"ENGINEERING_SMOKE_TEST","family":"engineering_smoke","executed_tone_order":"ENGINEERING","declared_tone_order":"ENGINEERING"}
 rows=[]
 for i,mode in enumerate(("basis","rotation")):
  rows.append({**common,"window_index":i,"stage":"ENGINEERING_SMOKE_DRIVEN","actual_mode":mode,"declared_mode":mode,"measurement_mode":"lockin_and_raw_ring","drive_on":True,"sender_off_required":False,"physical_tone_index":i,"codeword_source_index":i,"theta_idx":i,"amplitude_level":1})
 rows.append({**common,"window_index":2,"stage":"ENGINEERING_SMOKE_SENDER_OFF","actual_mode":None,"declared_mode":None,"measurement_mode":"raw_ring_sender_off","drive_on":False,"sender_off_required":True,"physical_tone_index":None,"codeword_source_index":None,"theta_idx":None,"amplitude_level":0})
 header={"schema_id":"CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V1","campaign_source_commit":"0"*40,"campaign_plan_sha256":"0"*64,"session_id":sid,"route":"v4s5","seed":-1,"partition":"ENGINEERING_SMOKE_TEST_NOT_SCIENTIFIC_ACQUISITION","window_count":len(rows),"restoration_authorized":False}
 (a.output/"session.json").write_text(json.dumps(header,indent=2,sort_keys=True)+"\n")
 (a.output/"windows.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows))
 manifest={"schema_id":"CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1","session_id":sid,"files":{n:{"size":(a.output/n).stat().st_size,"sha256":sha(a.output/n)} for n in ("session.json","windows.jsonl")}}
 (a.output/"session_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");print("ENGINEERING_SMOKE_TEST\nNOT_SCIENTIFIC_ACQUISITION");return 0
if __name__=="__main__":raise SystemExit(main())
