"""Frozen final experiment bindings and transactional receipts; standard library only."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

PROTOCOL_SHA256 = 'f02d5750a96f5532168aaf37a2a2b19d2bd2f20342837fc5459b4114485d350a'
PLAN_SHA256 = 'ff802370c85ed286d2ff8bad8cdf681c59ad9aa892037adc971cc2e8fc34d8f3'
PROTOCOL_ID = 'aura-ksp-stream3-ntu120-xset-final-v1'
INSTANCE_ID = '49850271'
STREAMS = ('joint', 'bone', 'joint_motion', 'bone_motion')
SOURCE_HASHES = {
    'train': 'fd037a9f71dbb00778bf71428892c8de407577402ee8330fdba0b02cea20edaa',
    'val': '2ece0ff18ea434cea3815dfeaf4832cb527d2f64dbbc5167f7389d9d2892ff35',
    'test': '5aed365468ad753ffd6887df88109841155f302db6cc033459c79cd9afd3194f',
}
SAMPLES = {'train': 38670, 'val': 15798, 'test': 59477}
SETUPS = {'train': [4,6,8,12,14,16,18,22,24,26,28,32], 'val': [2,10,20,30], 'test': list(range(1,33,2))}
SAMPLE_ID = re.compile(r'^S(\d{3})C\d{3}P\d{3}R\d{3}A(\d{3})$')


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, json_bytes(value))


def create_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(json_bytes(value))
        f.flush()
        os.fsync(f.fileno())


def kit_root():
    return Path(__file__).resolve().parents[1]


def protocol():
    p = kit_root()/'protocol/PROTOCOL.json'
    require(sha256_file(p) == PROTOCOL_SHA256, 'Frozen protocol SHA-256 mismatch')
    v = load_json(p)
    require(v['protocol_id'] == PROTOCOL_ID, 'Wrong protocol identity')
    return v


def plan():
    p = kit_root()/'PLAN.json'
    require(sha256_file(p) == PLAN_SHA256, 'Frozen plan SHA-256 mismatch')
    v = load_json(p)
    require(v['protocol_sha256'] == PROTOCOL_SHA256, 'Wrong plan/protocol binding')
    require([j['job_id'] for j in v['jobs']] == ['train_'+s for s in STREAMS]+['evaluate_'+s for s in STREAMS]+['analysis'], 'Changed job matrix')
    return v


def normalize(value, key=None):
    if isinstance(value, dict):
        return {k: normalize(v,k) for k,v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, str) and key and key.endswith('_path'):
        return value.replace('\\','/')
    return value


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def canonical_hash(rows):
    h=hashlib.sha256()
    for row in rows:
        h.update((json.dumps(normalize(row),sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode())
    return h.hexdigest()


def safe_relative(value):
    text=str(value).replace('\\','/')
    p=PurePosixPath(text)
    require(not p.is_absolute() and '..' not in p.parts and ':' not in text and bool(p.parts), 'Unsafe relative path: '+text)
    return p.as_posix()


def validate_partition(rows, role):
    require(role in SOURCE_HASHES, 'Unknown source partition')
    require(len(rows)==SAMPLES[role], 'Wrong '+role+' sample count')
    require(canonical_hash(rows)==SOURCE_HASHES[role], 'Canonical '+role+' manifest hash mismatch')
    ids=[]
    for row in rows:
        sid=str(row['sample_id'])
        m=SAMPLE_ID.fullmatch(sid)
        require(m is not None, 'Invalid NTU sample ID: '+sid)
        require(int(row['setup'])==int(m.group(1)), 'Setup disagrees with sample ID')
        require(int(row['label'])==int(m.group(2))-1 and 0<=int(row['label'])<120, 'Label disagrees with sample ID')
        require(int(row['sequence_length'])>0, 'Invalid sequence length')
        rel=safe_relative(row['data_path'])
        require(rel=='samples/'+sid+'.npz', 'Unexpected canonical NPZ path: '+rel)
        ids.append(sid)
    require(len(ids)==len(set(ids)), 'Duplicate IDs in '+role)
    require(sorted({int(r['setup']) for r in rows})==SETUPS[role], 'Unexpected setup partition')
    require(sorted({int(r['label']) for r in rows})==list(range(120)), 'Missing classes in '+role)
    return set(ids)


def verify_partitions(partitions):
    ids={role:validate_partition(partitions[role],role) for role in SOURCE_HASHES}
    require(not (ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test']), 'Partitions overlap')
    return ids


def parity_rows(rows):
    result=[]
    for setup in SETUPS['train']:
        group=sorted((r for r in rows if int(r['setup'])==setup),key=lambda r:(int(r['sequence_length']),str(r['sample_id'])))
        require(bool(group), 'Missing parity setup')
        result.extend(group[i] for i in sorted({0,len(group)//2,len(group)-1}))
    return result


def history_scan(roots, allow_marker=None):
    """Inspect historical result/marker filenames only; never read result metrics."""
    hits=[]; scanned=[]; absent=[]
    skip={'data','samples','.venv','venv','.git','__pycache__','node_modules','src','tests','configs','protocols'}
    for root in dict.fromkeys(str(Path(x).resolve()) for x in roots):
        p=Path(root)
        if not p.is_dir():
            absent.append(root); continue
        scanned.append(root)
        for directory, dirs, files in os.walk(p):
            dirs[:]=[d for d in dirs if d not in skip]
            for name in files:
                f=Path(directory)/name
                low=str(f).replace('\\','/').lower()
                if not ('ntu120' in low or 'xset' in low):
                    continue
                marker=('test' in name.lower() and any(t in name.lower() for t in ('started','complete','event')) and name.endswith('.json'))
                test_output=any(t in low for t in ('/eval_test/','/final_test/','/sealed_test','/test_predictions/')) and (name in ('metrics.json','predictions.npz') or name.endswith('.npz'))
                if (marker or test_output) and (allow_marker is None or f.resolve()!=Path(allow_marker).resolve()):
                    hits.append(str(f))
    return {'schema':'aura-ksp.final-history-scan.v1','created_utc':utc(),'scope':'Known roots; result and marker filenames only, no outcome content read. Absence is not proof about inaccessible copies.','scanned_roots':scanned,'absent_roots':absent,'unexpected_xset_test_artifacts':sorted(set(hits)),'outcome_content_read':False}


def verify_manifest(root, filename='MANIFEST_SHA256.txt'):
    root=Path(root); seen=set()
    for line in (root/filename).read_text(encoding='utf-8').splitlines():
        digest, rel=line.split('  ',1)
        rel=safe_relative(rel.removeprefix('./'))
        require(rel not in seen,'Duplicate manifest path')
        seen.add(rel)
        require(sha256_file(root/rel)==digest,'Manifest mismatch: '+rel)
    return len(seen)


def verify_kit():
    return verify_manifest(kit_root())


def load_binding(data_root):
    data_root=Path(data_root)
    value=load_json(data_root/'DATA_BINDING.json')
    require(value['protocol_sha256']==PROTOCOL_SHA256 and value['protocol_id']==PROTOCOL_ID,'Data binding protocol changed')
    require(value['status']=='BOUND_BEFORE_TRAINING_NO_TEST_ARRAYS_READ','Data is not bound before training')
    for rel,digest in value['bound_file_hashes'].items():
        require(sha256_file(data_root/safe_relative(rel))==digest,'Changed bound data file: '+rel)
    return value


def runtime_binding(data_root):
    return {'protocol_id':PROTOCOL_ID,'protocol_sha256':PROTOCOL_SHA256,'plan_sha256':PLAN_SHA256,'data_binding_sha256':sha256_file(Path(data_root)/'DATA_BINDING.json'),'instance_id':INSTANCE_ID}


def verify_authorization(path, data_root, require_instance=False):
    a=load_json(path)
    require(a.get('schema')=='aura-ksp.stream3-final-runtime-authorization.v1','Historical/wrong authorization refused')
    require(a.get('binding')==runtime_binding(data_root),'Authorization binding changed')
    for key in ('four_final_trainings','old_validation_training_only','one_xset_test_event','autostop_required'):
        require(a.get(key) is True,'Missing authorized scope: '+key)
    if require_instance:
        actual=(os.environ.get('CONTAINER_ID') or os.environ.get('VAST_CONTAINERLABEL') or '').removeprefix('C.')
        require(actual==INSTANCE_ID,'Wrong live Vast instance: '+actual)
    return a


def job_binding(job_id, data_root):
    require(job_id in [j['job_id'] for j in plan()['jobs']],'Unknown job ID')
    return {**runtime_binding(data_root),'job_id':job_id}


def receipt_path(run_root, job_id):
    return Path(run_root)/'receipts'/(job_id+'.json')


def write_receipt(run_root, job_id, data_root, files, seconds):
    root=Path(run_root)
    outputs={str(Path(f).relative_to(root)).replace('\\','/'):sha256_file(f) for f in files}
    require(bool(outputs),'Empty receipt is invalid')
    value={'schema':'aura-ksp.stream3-final-job-receipt.v1','created_utc':utc(),'status':'COMPLETE','binding':job_binding(job_id,data_root),'outputs':outputs,'wall_seconds_this_invocation':float(seconds)}
    create_once(receipt_path(root,job_id),value)
    return value


def verify_receipt(run_root, job_id, data_root):
    root=Path(run_root); v=load_json(receipt_path(root,job_id))
    require(v['status']=='COMPLETE' and v['binding']==job_binding(job_id,data_root),'Wrong completion receipt')
    require(bool(v['outputs']),'Empty receipt')
    for rel,h in v['outputs'].items():
        require(sha256_file(root/safe_relative(rel))==h,'Changed completed output: '+rel)
    return v


def begin_test_event(marker, data_root, models):
    expected={**runtime_binding(data_root),'model_sha256':models,'test_manifest_canonical_sha256':SOURCE_HASHES['test'],'event_id':PROTOCOL_ID+'-single-test-event'}
    marker=Path(marker)
    if marker.exists():
        require(load_json(marker).get('binding')==expected,'A different or corrupt test event already consumed this boundary')
        return expected
    create_once(marker,{'schema':'aura-ksp.stream3-sealed-test-event.v1','created_utc':utc(),'status':'CONSUMED_BEFORE_FIRST_TEST_TENSOR','binding':expected})
    return expected


def verify_test_event(marker, expected):
    v=load_json(marker)
    require(v.get('status')=='CONSUMED_BEFORE_FIRST_TEST_TENSOR' and v.get('binding')==expected,'Test event missing or mismatched')


def last_committed_epoch(directory, expected_binding):
    directory=Path(directory)
    commits=sorted(directory.glob('epoch_*.commit.json'))
    if not commits:
        return None
    c=load_json(commits[-1])
    require(c['binding']==expected_binding,'Resume binding mismatch')
    path=directory/safe_relative(c['checkpoint'])
    require(path.name==f"epoch_{int(c['epoch']):03d}.pt",'Bad checkpoint epoch name')
    require(sha256_file(path)==c['checkpoint_sha256'],'Corrupt committed checkpoint; no silent fallback')
    return c,path


def commit_epoch(directory, epoch, expected_binding, saver):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'epoch_{epoch:03d}.pt'
    commit=directory/f'epoch_{epoch:03d}.commit.json'
    require(not commit.exists(),'Epoch is already committed')
    saver(path)
    create_once(commit,{'epoch':int(epoch),'checkpoint':path.name,'checkpoint_sha256':sha256_file(path),'binding':expected_binding,'created_utc':utc()})
    # Delete only superseded, committed resumable states after a new commit exists.
    for old in sorted(directory.glob('epoch_*.commit.json'))[:-2]:
        data=load_json(old)
        old_state=directory/safe_relative(data['checkpoint'])
        old.unlink()
        old_state.unlink(missing_ok=True)
    return path


def write_csv(path, rows, fieldnames=None):
    import io
    require(bool(rows) or fieldnames is not None,'CSV requires columns')
    f=io.StringIO(newline='')
    w=csv.DictWriter(f,fieldnames=fieldnames or list(rows[0]))
    w.writeheader();w.writerows(rows)
    atomic_bytes(path,f.getvalue().encode('utf-8'))
