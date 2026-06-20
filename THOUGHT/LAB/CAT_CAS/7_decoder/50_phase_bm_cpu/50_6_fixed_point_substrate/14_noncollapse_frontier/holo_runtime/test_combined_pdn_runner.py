#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, os, subprocess, tempfile, unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent; RUNNER=HERE/"combined_pdn_runner"
def sha(p:Path)->str:
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def dump(p:Path,v:object)->None:p.write_text(json.dumps(v,sort_keys=True,indent=2)+"\n",encoding="utf-8")
def base_window(i:int,off:bool=False)->dict:
 return {"window_index":i,"session_id":"v4s5_seed4","stage":"C_PERSISTENCE_OFF" if off else "B_TONE_ORDER","block_id":"block","family":"silent" if off else "real","actual_mode":"basis","declared_mode":"basis","executed_tone_order":"FWD","declared_tone_order":"FWD","measurement_mode":"raw_ring_sender_off" if off else "lockin_and_raw_ring","drive_on":not off,"sender_off_required":off,"physical_tone_index":None if off else i,"codeword_source_index":None if off else i,"theta_idx":None if off else 0,"amplitude_level":0 if off else 3}
def write_session(root:Path,mutate=None,count=3,manifest_schema="CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1")->Path:
 d=root/"session";d.mkdir(); rows=[base_window(0),base_window(1),base_window(2,True)][:count]
 h={"schema_id":"CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V1","campaign_source_commit":"f5b6079a5748bb6138ab19d1c22d79c74734dddf","campaign_plan_sha256":"e"*64,"session_id":"v4s5_seed4","route":"v4s5","seed":4,"partition":"stress","window_count":len(rows),"restoration_authorized":False}
 if mutate: mutate(h,rows)
 dump(d/"session.json",h);(d/"windows.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows),encoding="utf-8")
 m={"schema_id":manifest_schema,"session_id":"v4s5_seed4","files":{n:{"size":(d/n).stat().st_size,"sha256":sha(d/n)} for n in ("session.json","windows.jsonl")}};dump(d/"session_manifest.json",m);return d
class Tests(unittest.TestCase):
 def exec_runner(self,session:Path,out:Path,*args:str,fail:str|None=None):
  e=os.environ.copy();
  if fail:e["COMBINED_PDN_MOCK_FAIL"]=fail
  return subprocess.run([str(RUNNER),"--session-dir",str(session),"--output-dir",str(out),"--victim","4","--sender","5","--executor-commit","0"*40,*args],text=True,capture_output=True,env=e)
 def assert_reject(self,mutate,text,manifest_schema=None):
  with tempfile.TemporaryDirectory() as t:
   d=write_session(Path(t),mutate=mutate,manifest_schema=manifest_schema or "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1");p=self.exec_runner(d,Path(t)/"o","--validate-only");self.assertNotEqual(p.returncode,0);self.assertIn(text,p.stderr)
 def test_validate_contract_files_and_hashes(self):
  with tempfile.TemporaryDirectory() as t:
   d=write_session(Path(t));o=Path(t)/"o";p=self.exec_runner(d,o,"--validate-only");self.assertEqual(p.returncode,0,p.stderr);names=("run.json","session.json","windows.jsonl","window_results.csv","raw_samples.bin","telemetry.csv","stdout.log","stderr.log","run_manifest.json");self.assertTrue(all((o/n).is_file() for n in names));m=json.loads((o/"run_manifest.json").read_text());self.assertNotIn("run_manifest.json",m["files"]);self.assertTrue(all(sha(o/n)==v["sha256"] for n,v in m["files"].items()))
 def test_output_collision(self):
  with tempfile.TemporaryDirectory() as t:d=write_session(Path(t));o=Path(t)/"o";o.mkdir();self.assertIn("refusing existing output",self.exec_runner(d,o,"--validate-only").stderr)
 def test_manifest_schema(self):self.assert_reject(None,"unexpected session manifest schema","BAD")
 def test_manifest_size(self):
  with tempfile.TemporaryDirectory() as t:d=write_session(Path(t));(d/"windows.jsonl").write_text((d/"windows.jsonl").read_text()+" ");self.assertIn("size mismatch",self.exec_runner(d,Path(t)/"o","--validate-only").stderr)
 def test_manifest_sha(self):
  def m(h,r):r[0]["family"]="wrong"
  with tempfile.TemporaryDirectory() as t:
   d=write_session(Path(t));x=json.loads((d/"session_manifest.json").read_text());x["files"]["windows.jsonl"]["sha256"]="0"*64;dump(d/"session_manifest.json",x);self.assertIn("sha256 mismatch",self.exec_runner(d,Path(t)/"o","--validate-only").stderr)
 def test_noncontiguous_duplicate(self):self.assert_reject(lambda h,r:r[1].update(window_index=0),"not contiguous")
 def test_session_id_mismatch(self):self.assert_reject(lambda h,r:r[0].update(session_id="bad"),"session ID mismatch")
 def test_unsupported_mode(self):self.assert_reject(lambda h,r:r[0].update(measurement_mode="bad"),"unsupported measurement")
 def test_sender_off_drive(self):self.assert_reject(lambda h,r:r[2].update(drive_on=True),"sender_off_required + drive_on")
 def test_raw_ring_requires_off(self):self.assert_reject(lambda h,r:r[2].update(sender_off_required=False),"raw_ring_sender_off requires")
 def test_driven_requires_tone(self):self.assert_reject(lambda h,r:r[0].update(physical_tone_index=None),"missing physical tone")
 def test_driven_requires_codeword(self):self.assert_reject(lambda h,r:r[0].update(codeword_source_index=None),"codeword source")
 def test_extra_rows(self):self.assert_reject(lambda h,r:h.update(window_count=2),"extra schedule rows")
 def test_short_count(self):self.assert_reject(lambda h,r:h.update(window_count=4),"short schedule row count")
 def test_unsafe_path(self):
  p=subprocess.run([str(RUNNER),"--session-dir","../x","--output-dir","/tmp/o","--victim","4","--sender","5","--validate-only"],text=True,capture_output=True);self.assertIn("unsafe path",p.stderr)
 def test_validation_never_hardware(self):
  with tempfile.TemporaryDirectory() as t:d=write_session(Path(t));p=self.exec_runner(d,Path(t)/"o","--validate-only",fail="thermal");self.assertEqual(p.returncode,0,p.stderr)
 def test_mock_sender_lifecycle_and_sender_off(self):
  with tempfile.TemporaryDirectory() as t:
   d=write_session(Path(t));o=Path(t)/"o";p=self.exec_runner(d,o,"--mock-hardware");self.assertEqual(p.returncode,0,p.stderr)
   with (o/"window_results.csv").open() as f:rows=list(csv.DictReader(f))
   self.assertEqual(rows[0]["sender_started"],"1");self.assertEqual(rows[0]["sender_stopped"],"1");self.assertEqual(rows[2]["sender_alive_at_capture"],"0");self.assertEqual(rows[2]["computed_I"],"null")
 def test_failure_cleanup_matrix(self):
  for failure in ("thermal","cpufreq","sender_create","sender_stop","capture","raw"):
   with self.subTest(failure=failure),tempfile.TemporaryDirectory() as t:
    d=write_session(Path(t));o=Path(t)/"o";p=self.exec_runner(d,o,"--mock-hardware",fail=failure);self.assertNotEqual(p.returncode,0);run=json.loads((o/"run.json").read_text());self.assertTrue(run["machine_state_restored"]);self.assertEqual(run["exit_status"],"FAILED");self.assertTrue((o/"run_manifest.json").is_file())
 def test_restoration_failure_fatal(self):
  with tempfile.TemporaryDirectory() as t:d=write_session(Path(t));o=Path(t)/"o";p=self.exec_runner(d,o,"--mock-hardware",fail="restore");self.assertEqual(p.returncode,6);self.assertEqual(json.loads((o/"run.json").read_text())["failure_reason"],"RESTORATION_FAILURE")
if __name__=="__main__":unittest.main(verbosity=2)
