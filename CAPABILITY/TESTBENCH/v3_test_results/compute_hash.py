import json
import hashlib

adapter = {
    'adapter_version': '1.0.0',
    'artifacts': {
        'domain_roots': 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
        'ledger': 'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
        'proof': 'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd'
    },
    'command': ['python', 'CAPABILITY/SKILLS/ant-worker/scripts/run.py', 'LAW/CONTRACTS/_runs/_tmp/phase65_registry/task.json', 'LAW/CONTRACTS/_runs/_tmp/phase65_registry/result.json'],
    'deref_caps': {'max_bytes': 1024, 'max_depth': 2, 'max_matches': 1, 'max_nodes': 10},
    'inputs': {'LAW/CONTRACTS/_runs/_tmp/phase65_registry/in.txt': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'},
    'jobspec': {
        'catalytic_domains': ['LAW/CONTRACTS/_runs/_tmp/phase65_registry/domain'],
        'determinism': 'deterministic',
        'inputs': {},
        'intent': 'capability: ant-worker copy (Phase 6.5)',
        'job_id': 'cap-ant-worker-copy-v1',
        'outputs': {'durable_paths': ['LAW/CONTRACTS/_runs/_tmp/phase65_registry/out.txt'], 'validation_criteria': {}},
        'phase': 6,
        'task_type': 'adapter_execution'
    },
    'name': 'ant-worker-copy-v1',
    'outputs': {'LAW/CONTRACTS/_runs/_tmp/phase65_registry/out.txt': 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'},
    'side_effects': {'clock': False, 'filesystem_unbounded': False, 'network': False, 'nondeterministic': False}
}

canonical = json.dumps(adapter, sort_keys=True, separators=(',', ':')).encode('utf-8')
computed_hash = hashlib.sha256(canonical).hexdigest()
print('Computed hash:', computed_hash)