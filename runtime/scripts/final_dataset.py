"""Reuse canonical sparse K64 operations with sparse or existing NPZ source supports."""
from __future__ import annotations
import sys
import zlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from final_common import (kit_root, read_jsonl, load_json, require, safe_relative,
    sha256_file, verify_test_event)
sys.path.insert(0,str(kit_root()/'project/src'))
from aura_har.data.sparse_reference import plan_sparse_resize64, execute_sparse_plan
from aura_har.data.sparse_store import NTU120FrameStore
from aura_har.data.reference import apply_modality, random_rotation
from aura_har.utils.reproducibility import seed_worker


class NPZSupports:
    def __init__(self, sequence, sample_id):
        self.sequence=sequence;self.sample_id=sample_id
    def read_frames(self, sample_id, indices):
        require(sample_id==self.sample_id,'NPZ sample mismatch')
        return np.ascontiguousarray(self.sequence[:,indices],dtype=np.float32)


def npz_joint(data_root, row, temporal_plan):
    path=Path(data_root)/safe_relative(row['data_path'])
    with np.load(path,allow_pickle=False) as npz:
        sequence=np.asarray(npz['skeleton'],dtype=np.float32)
        length=int(npz['sequence_length'])
    require(sequence.ndim==4 and sequence.shape[0]==3 and sequence.shape[2:]==(25,2),'NPZ shape changed')
    require(length==int(row['sequence_length']) and 0<length<=sequence.shape[1],'NPZ sequence length changed')
    return execute_sparse_plan(NPZSupports(sequence,str(row['sample_id'])),str(row['sample_id']),temporal_plan)


def transform(row, stream, seed, epoch, train, sparse_store=None, data_root=None):
    stable=zlib.crc32(str(row['sample_id']).encode('utf-8'))
    rng_seed=int(seed)+stable+(int(epoch)*1000003 if train else 0)
    plan_rng=np.random.default_rng(rng_seed)
    rotation_rng=np.random.default_rng(rng_seed ^ 0x5DEECE66D)
    p=plan_sparse_resize64(int(row['sequence_length']),[0.5,1.0] if train else [0.95],plan_rng)
    if sparse_store is not None:
        joint=execute_sparse_plan(sparse_store,str(row['sample_id']),p)
    else:
        require(data_root is not None,'Missing NPZ root')
        joint=npz_joint(data_root,row,p)
    if train:joint=random_rotation(joint,rotation_rng,theta=0.3)
    return apply_modality(joint,stream), np.array(p.output_source_indices,copy=True)


class FinalDataset(Dataset):
    def __init__(self,data_root,sparse_root,stream,seed,role,rows=None,test_marker=None,test_binding=None):
        require(role in ('train','test'),'Only frozen final train/test roles are supported')
        self.root=Path(data_root);self.sparse_root=Path(sparse_root)
        self.stream=stream;self.seed=int(seed);self.train=role=='train';self.epoch=0
        self.rows=rows if rows is not None else read_jsonl(self.root/('FINAL_TRAIN.jsonl' if self.train else 'TEST.jsonl'))
        allowed={'base_train','old_validation_training_only'} if self.train else {'sealed_test'}
        require(all(r['_final_partition'] in allowed for r in self.rows),'Dataset role crosses frozen split boundary')
        self.marker=test_marker;self.test_binding=test_binding
        if not self.train:
            require(self.marker is not None and self.test_binding is not None,'No test event binding')
            verify_test_event(self.marker,self.test_binding)
        self.store=NTU120FrameStore(self.sparse_root/'store.json') if self.train else None
        self.inventory={r['sample_id']:r for r in read_jsonl(self.root/'NPZ_INVENTORY.jsonl')}
        self.verified_npz=set()
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];sid=str(row['sample_id'])
        use_sparse=row['_final_partition']=='base_train'
        if not self.train:verify_test_event(self.marker,self.test_binding)
        if not use_sparse and sid not in self.verified_npz:
            rec=self.inventory[sid];path=self.root/safe_relative(rec['path'])
            require(path.stat().st_size==rec['bytes'] and sha256_file(path)==rec['sha256'],'NPZ changed since binding: '+sid)
            self.verified_npz.add(sid)
        x,indices=transform(row,self.stream,self.seed,self.epoch,self.train,
                            sparse_store=self.store if use_sparse else None,data_root=None if use_sparse else self.root)
        return {'skeleton':torch.from_numpy(x),'label':torch.tensor(int(row['label']),dtype=torch.long),
                'setup':torch.tensor(int(row['setup']),dtype=torch.long),'sample_id':sid,'indices':torch.from_numpy(indices)}


def loader_for(dataset,seed,shuffle,batch_size=64,num_workers=8):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),persistent_workers=False)
