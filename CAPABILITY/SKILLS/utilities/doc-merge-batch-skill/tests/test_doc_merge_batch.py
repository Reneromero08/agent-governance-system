import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import json
from doc_merge_batch.core import run_job

class MockWriter:
    def write_durable(self, path, content):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding='utf-8')
    def mkdir_durable(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

def test_compare_and_apply_and_verify(tmp_path: Path):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("# Title\n\nA1\n\nA2\n", encoding="utf-8")
    b.write_text("# Title\n\nA1\n\nB2\n", encoding="utf-8")

    out_dir = tmp_path / "out"

    payload = {
        "mode": "verify",
        "out_dir": str(out_dir),
        "pairs": [{"a": str(a), "b": str(b)}],
        "normalization": {"newline":"lf","strip_trailing_ws":False,"collapse_blank_lines":False},
        "diff": {"max_diff_lines": 200, "context_lines": 3},
        "merge": {"base":"a","strategy":"append_unique_blocks"},
        "max_file_mb": 5,
        "max_pairs": 10,
    }

    writer = MockWriter()
    report = run_job(payload, writer)
    assert report["errors"] == []
    assert len(report["artifacts"]) == 1
    v = report["artifacts"][0]["verification"]
    assert v["pass"] is True
