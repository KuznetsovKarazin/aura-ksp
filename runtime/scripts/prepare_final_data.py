"""Package existing NPZ bytes for the authorized final experiment. No preprocessing/GPU."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import struct
import tarfile
import time
from pathlib import Path

from final_common import (PROTOCOL_ID, PROTOCOL_SHA256, SOURCE_HASHES, SETUPS, atomic_json,
    canonical_hash, history_scan, json_bytes, kit_root, normalize, parity_rows, protocol,
    read_jsonl, require, safe_relative, sha256_file, utc, verify_partitions)

PACKAGE_ROOT='AURA-KSP_STREAM3_FINAL_DATA_20260905_v1.0.0'


def frozen_draws(test_rows):
    spec=protocol()['statistics']['bootstrap_pool']
    source=kit_root()/'protocol'/spec['file']
    require(sha256_file(source)==spec['sha256'],'Bootstrap pool changed')
    pool=json.loads(gzip.decompress(source.read_bytes()))
    require(len(pool)==20000 and all(len(r)==16 and all(type(x) is int and 0<=x<16 for x in r) for r in pool),'Bad pool shape/content')
    require(hashlib.sha256(b''.join(struct.pack('<16q',*r) for r in pool)).hexdigest()==spec['raw_int64_little_endian_sha256'],'Pool raw hash mismatch')
    support=[sum(int(r['setup'])==s and int(r['label'])>=60 for r in test_rows) for s in SETUPS['test']]
    selected=[];rejected=0
    for row in pool:
        if sum(support[i] for i in row)>0:
            selected.append(row)
            if len(selected)==10000:break
        else:rejected+=1
    require(len(selected)==10000,'Frozen pool does not provide 10000 valid draws')
    raw=b''.join(struct.pack('<16q',*r) for r in selected)
    content=gzip.compress((json.dumps(selected,separators=(',',':'))+'\n').encode(),mtime=0)
    return content,{'shape':[10000,16],'dtype':'little_endian_int64','bootstrap_seed':223607,'pool_sha256':spec['sha256'],'rejected_before_10000_valid':rejected,'raw_sha256':hashlib.sha256(raw).hexdigest(),'a61_a120_support_by_setup':support,'prediction_data_inspected':False}


def jsonl_bytes(rows):
    return ''.join(json.dumps(normalize(r),ensure_ascii=False,sort_keys=True)+'\n' for r in rows).encode('utf-8')


class HashingReader:
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256()
    def read(self,n=-1):
        b=self.stream.read(n);self.digest.update(b);return b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--history-root',type=Path,action='append',default=[])
    args=parser.parse_args()
    protocol()
    output=args.output.resolve()
    require(not output.exists(),'Output archive already exists; do not overwrite a bound data package')
    require(output.name.endswith('.tar.gz'),'Output must end in .tar.gz')
    root=args.project_root.resolve();processed=root/'data/processed/ntu120_xset'
    require(processed.is_dir(),'Processed NTU120 XSet directory is absent')
    history_roots=[root]+args.history_root
    if os.name=='nt':
        history_roots += [Path(x) for x in (r'E:\AURA',r'E:\aura-har',r'E:\aura-har-v023',r'E:\aura-har-v030')]
    history=history_scan(history_roots)
    output.parent.mkdir(parents=True,exist_ok=True)
    if history['unexpected_xset_test_artifacts']:
        report=Path(str(output)+'.HISTORY_BLOCKED.json')
        atomic_json(report,history)
        raise RuntimeError('Historical XSet test artifacts require audit before launch; see '+str(report))
    rows={role:read_jsonl(processed/'manifests'/(role+'.jsonl')) for role in SOURCE_HASHES}
    verify_partitions(rows)
    parity=parity_rows(rows['train'])
    transfer=[(r,'val') for r in rows['val']]+[(r,'test') for r in rows['test']]+[(r,'train_parity_only') for r in parity]
    total_bytes=0;missing=[]
    for row,role in transfer:
        path=processed/safe_relative(row['data_path'])
        if not path.is_file():missing.append(str(path))
        else:total_bytes+=path.stat().st_size
    require(not missing,'Missing NPZ files: '+json.dumps(missing[:20],ensure_ascii=False)+' total='+str(len(missing)))
    available=shutil.disk_usage(output.parent).free
    require(available>total_bytes*1.02+256*1024**2,'Insufficient free space for the data archive')
    print(json.dumps({'status':'DATA_PACKAGE_INPUTS_VERIFIED','npz_to_copy':len(transfer),'val_npz':len(rows['val']),'test_npz':len(rows['test']),'train_parity_npz':len(parity),'existing_train_sparse_samples_reused':38670,'source_bytes':total_bytes,'source_gib':total_bytes/1024**3,'gpu_used':False,'test_arrays_decoded':False},indent=2),flush=True)
    draw_content,draw_info=frozen_draws(rows['test'])
    files={
        **{'source_manifests/'+role+'.jsonl':(processed/'manifests'/(role+'.jsonl')).read_bytes() for role in SOURCE_HASHES},
        'FINAL_TRAIN.jsonl':jsonl_bytes([{**r,'_final_partition':'base_train'} for r in rows['train']]+[{**r,'_final_partition':'old_validation_training_only'} for r in rows['val']]),
        'TEST.jsonl':jsonl_bytes([{**r,'_final_partition':'sealed_test'} for r in rows['test']]),
        'PARITY_SAMPLES.json':json_bytes([r['sample_id'] for r in parity]),
        'BOOTSTRAP_DRAW_INDICES.json.gz':draw_content,
        'HISTORY_AUDIT_WINDOWS.json':json_bytes(history),
    }
    inventory=[];manifest=[]
    temporary=Path(str(output)+'.partial')
    require(not temporary.exists(),'A partial archive exists; inspect the previous error before removing that partial file')
    started=time.monotonic()
    with temporary.open('xb') as out:
        with tarfile.open(fileobj=out,mode='w:gz',compresslevel=1) as tar:
            def add_bytes(rel,data):
                info=tarfile.TarInfo(PACKAGE_ROOT+'/'+safe_relative(rel));info.size=len(data);info.mode=0o644;info.mtime=0
                tar.addfile(info,io.BytesIO(data))
                manifest.append(hashlib.sha256(data).hexdigest()+'  '+rel+'\n')
            for number,(row,role) in enumerate(transfer,1):
                rel=safe_relative(row['data_path']);source=processed/rel
                stat=source.stat()
                info=tarfile.TarInfo(PACKAGE_ROOT+'/'+rel);info.size=stat.st_size;info.mode=0o644;info.mtime=0
                with source.open('rb') as f:
                    reader=HashingReader(f);tar.addfile(info,reader);h=reader.digest.hexdigest()
                after=source.stat()
                require((stat.st_size,stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source changed while packaging: '+rel)
                inventory.append({'sample_id':row['sample_id'],'path':rel,'role':role,'bytes':stat.st_size,'sha256':h})
                manifest.append(h+'  '+rel+'\n')
                if number%2000==0 or number==len(transfer):
                    print(f'PACKED {number}/{len(transfer)} elapsed_seconds={time.monotonic()-started:.1f}',flush=True)
            files['NPZ_INVENTORY.jsonl']=jsonl_bytes(inventory)
            binding={
                'schema':'aura-ksp.stream3-final-data-binding.v1','protocol_id':PROTOCOL_ID,'protocol_sha256':PROTOCOL_SHA256,
                'status':'BOUND_BEFORE_TRAINING_NO_TEST_ARRAYS_READ','created_utc':utc(),
                'canonical_source_hashes':{role:canonical_hash(rs) for role,rs in rows.items()},
                'samples':{'base_train':38670,'old_validation_training_only':15798,'final_train':54468,'test':59477},
                'npz_files':len(inventory),'npz_bytes':total_bytes,'parity_samples':len(parity),
                'bootstrap':draw_info,
                'bound_file_hashes':{rel:hashlib.sha256(b).hexdigest() for rel,b in files.items()},
                'preparation_scope':{'test_manifest_metadata_read':True,'old_validation_manifest_metadata_read':True,'compressed_npz_copied_and_hashed':True,'test_npz_tensors_decoded':False,'old_validation_npz_tensors_decoded':False,'models_run':False,'preprocessing_repeated':False},
                'preparation_script_sha256':sha256_file(Path(__file__))}
            files['DATA_BINDING.json']=json_bytes(binding)
            for rel,data in files.items():add_bytes(rel,data)
            add_bytes('MANIFEST_SHA256.txt',''.join(manifest).encode('utf-8'))
    os.replace(temporary,output)
    h=sha256_file(output)
    Path(str(output)+'.sha256').write_text(h+'  '+output.name+'\n',encoding='utf-8')
    receipt={'schema':'aura-ksp.stream3-final-data-package-receipt.v1','status':'DATA_PACKAGE_CREATED_NO_TRAINING_NO_TEST_ARRAY_DECODING','created_utc':utc(),'archive':str(output),'archive_sha256':h,'archive_bytes':output.stat().st_size,'data_binding_sha256':hashlib.sha256(files['DATA_BINDING.json']).hexdigest(),'payload_files':len(inventory)+len(files),'samples':binding['samples'],'bootstrap':draw_info,'packaging_seconds':time.monotonic()-started,'preparation_scope':binding['preparation_scope']}
    atomic_json(Path(str(output)+'.RECEIPT.json'),receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
