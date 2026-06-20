#!/usr/bin/env python3
"""Read-only acquisition preflight for the authorized combined campaign."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil
from pathlib import Path
from typing import Any
from generate_campaign_plan import verify
ROUTES={"v4s5":(4,5),"v2s3":(2,3)}
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def first_k10temp()->str|None:
 for name in sorted(Path("/sys/class/hwmon").glob("hwmon*/name")):
  try:
   if name.read_text().strip()=="k10temp" and (name.parent/"temp1_input").is_file():return str(name.parent/"temp1_input")
  except OSError:pass
 return None
def cpu_flags()->set[str]:
 try:
  for line in Path("/proc/cpuinfo").read_text().splitlines():
   if line.startswith("flags"):return set(line.split(":",1)[1].split())
 except OSError:pass
 return set()
def session_bundles_valid(root:Path)->bool:
 dirs=sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
 if len(dirs)!=12:return False
 for directory in dirs:
  try:
   m=json.loads((directory/"session_manifest.json").read_text())
   for name,b in m["files"].items():
    p=directory/name
    if not p.is_file() or p.stat().st_size!=b["size"] or sha(p)!=b["sha256"]:return False
  except (OSError,KeyError,ValueError,json.JSONDecodeError):return False
 return True
def inspect(plan_dir:Path,repo_root:Path,output_root:Path,min_free_gb:float)->dict[str,Any]:
 manifest=json.loads((plan_dir/"campaign_manifest.json").read_text());plan=json.loads((plan_dir/"campaign_plan.json").read_text());source=manifest.get("source_commit")
 bundle_path=repo_root/"source_bundle.json"
 if not bundle_path.is_file():raise ValueError("preflight requires hash-bound source_bundle.json")
 bundle=json.loads(bundle_path.read_text())
 runner=repo_root/"combined_pdn_runner";binding_path=repo_root/"COMBINED_CAMPAIGN_BINDING.json";schedules=repo_root/"compiled_sessions"
 binding=json.loads(binding_path.read_text()) if binding_path.is_file() else {}
 head=bundle.get("executor_commit");source_ok=bundle.get("plan_source_verified") is True;clean=True;package_ok=bundle.get("frozen_inputs_verified") is True
 flags=cpu_flags();cpus=os.cpu_count() or 0
 msr={str(c):os.access(f"/dev/cpu/{c}/msr",os.R_OK) for c in range(min(cpus,6))}
 cpufreq={str(c):all(os.access(f"/sys/devices/system/cpu/cpu{c}/cpufreq/{n}",os.R_OK|os.W_OK) for n in ("scaling_min_freq","scaling_max_freq")) for c in range(min(cpus,6))}
 usage=shutil.disk_usage(output_root.parent if output_root.parent.exists() else repo_root);plan_errors=verify(plan_dir)
 plan_hash=sha(plan_dir/"campaign_plan.json");manifest_hash=sha(plan_dir/"campaign_manifest.json")
 route_ok=all(v!=s and v<cpus and s<cpus for v,s in ROUTES.values()) and {x.get("route") for x in plan.get("sessions",[])}==set(ROUTES)
 checks={
  "running_as_root":os.geteuid()==0,"cpu_count_at_least_6":cpus>=6,"route_cores_online_and_distinct":route_ok,
  "constant_tsc":"constant_tsc" in flags,"nonstop_tsc":"nonstop_tsc" in flags,"k10temp_available":first_k10temp() is not None,
  "msr_readable_cores_0_5":len(msr)==6 and all(msr.values()),"cpufreq_controls_readable_writable_cores_0_5":len(cpufreq)==6 and all(cpufreq.values()),
  "free_space_sufficient":usage.free>=int(min_free_gb*1024**3),"working_tree_or_source_bundle_acceptable":clean,
  "plan_source_verified":source_ok,"authorized_package_unchanged_since_plan":package_ok,"plan_sources_agree":source==plan.get("source_commit"),
  "plan_manifest_valid":not plan_errors,"canonical_plan_binding_valid":plan_hash==binding.get("campaign_plan",{}).get("sha256") and manifest_hash==binding.get("campaign_manifest_sha256"),
  "executor_exists_and_executable":runner.is_file() and os.access(runner,os.X_OK),
  "executor_commit_recorded":isinstance(head,str) and len(head)==40,
  "executor_binary_hash_valid":runner.is_file() and sha(runner)==bundle.get("executor_sha256"),
  "required_tests_pass":bundle.get("strict_tests_pass") is True and bundle.get("sanitizers_pass") is True,
  "all_twelve_schedules_compiled":session_bundles_valid(schedules) and bundle.get("validation_sessions_passed")==12,
  "output_path_unused":not output_root.exists(),"restoration_not_authorized":plan.get("restoration_authorized") is False and manifest.get("restoration_authorized") is False,
 }
 return {"schema_id":"CAT_CAS_PHASE6_COMBINED_PREFLIGHT_V2","host":os.uname().nodename,"repo_root":str(repo_root),"source_bundle_mode":bool(bundle),"executor_commit":head,"plan_source_commit":source,"plan_dir":str(plan_dir),"plan_sha256":plan_hash,"campaign_manifest_sha256":manifest_hash,"output_root":str(output_root),"cpu_count":cpus,"k10temp_path":first_k10temp(),"msr_readable":msr,"cpufreq_controls":cpufreq,"free_bytes":usage.free,"minimum_free_gb":min_free_gb,"plan_validation_errors":plan_errors,"checks":checks,"acquisition_ready":all(checks.values())}
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--plan-dir",type=Path,required=True);p.add_argument("--repo-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--report",type=Path,required=True);p.add_argument("--min-free-gb",type=float,default=20.0);a=p.parse_args()
 report=inspect(a.plan_dir.resolve(),a.repo_root.resolve(),a.output_root.resolve(),a.min_free_gb);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True));return 0 if report["acquisition_ready"] else 2
if __name__=="__main__":raise SystemExit(main())
