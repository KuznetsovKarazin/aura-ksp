"""Execute one pinned final training/evaluation job, with verified technical resume."""
from __future__ import annotations
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
import numpy as np
from final_common import (STREAMS, atomic_bytes, atomic_json, begin_test_event, commit_epoch,
    create_once, job_binding, kit_root, last_committed_epoch, load_binding, load_json, plan,
    protocol, read_jsonl, receipt_path, require, runtime_binding, sha256_file, utc,
    verify_authorization, verify_kit, verify_receipt, verify_test_event, write_csv, write_receipt)


def atomic_npz(path,**arrays):
    import io
    f=io.BytesIO();np.savez_compressed(f,**arrays);atomic_bytes(path,f.getvalue())


def assert_runtime(args):
    verify_kit();protocol();load_binding(args.data_root)
    auth=verify_authorization(args.authorization,args.data_root,require_instance=True)
    for key in ('data_root','sparse_root','run_root','project_root'):
        require(str(getattr(args,key).resolve())==auth['paths'][key],'Runtime path differs from authorization: '+key)
    preflight=load_json(args.authorization.parent/'PREFLIGHT.json')
    require(sha256_file(args.authorization.parent/'PREFLIGHT.json')==auth['preflight_sha256'],'Preflight receipt changed')
    require(preflight['status']=='CPU_PREFLIGHT_PASS_NO_TEST_TENSOR_READ' and preflight['binding']==runtime_binding(args.data_root),'Missing or wrong preflight')
    return auth


def train(args,job):
    import torch
    from torch import nn
    from aura_har.models import build_model
    from aura_har.training import build_optimizer,build_scheduler,train_one_epoch
    from aura_har.utils.io import atomic_torch_save
    from aura_har.utils.reproducibility import (seed_everything,capture_rng_state,
        load_checkpoint_on_cpu,restore_rng_state,move_optimizer_state_to_device)
    from final_dataset import FinalDataset,loader_for
    output=args.run_root/job['output_relative'];output.mkdir(parents=True,exist_ok=True)
    binding=job_binding(job['job_id'],args.data_root)
    initial=output/'TRAINING_BINDING.json'
    if initial.exists():require(load_json(initial)==binding,'Training directory belongs to a different run')
    else:
        require(not any(output.iterdir()),'Unexpected nonempty training directory')
        create_once(initial,binding)
    config=job['config'];seed=int(config['experiment']['seed'])
    seed_everything(seed,True)
    device=torch.device('cuda')
    require(torch.cuda.is_available(),'CUDA is unavailable')
    model=build_model(config).to(device)
    optimizer=build_optimizer(model,config);scheduler=build_scheduler(optimizer,config)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    dataset=FinalDataset(args.data_root,args.sparse_root,job['stream'],seed,'train')
    loader=loader_for(dataset,seed,True)
    committed=last_committed_epoch(output/'epochs',binding)
    history=[];start=1;state=None
    if committed:
        c,checkpoint=committed;state=load_checkpoint_on_cpu(checkpoint)
        require(state['binding']==binding and state['epoch']==c['epoch'],'Resume state binding/epoch mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        move_optimizer_state_to_device(optimizer,device);scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']);restore_rng_state(state['rng_state'])
        history=state['history'];start=int(state['epoch'])+1
        require([r['epoch'] for r in history]==list(range(1,start)),'Checkpoint history is inconsistent')
        print(f'RESUME {job["job_id"]} next_epoch={start}',flush=True)
    started=time.monotonic()
    for epoch in range(start,66):
        dataset.set_epoch(epoch);loader.generator.manual_seed(seed+epoch)
        stats=train_one_epoch(model,loader,optimizer,scaler,nn.CrossEntropyLoss(label_smoothing=0.0),device,False,1)
        require(all(np.isfinite(stats[k]) for k in ('loss','top1','seconds')),'Nonfinite training statistics')
        history.append({'epoch':epoch,'learning_rate':optimizer.param_groups[0]['lr'],'train_loss':stats['loss'],'train_top1':stats['top1'],'epoch_seconds':stats['seconds']})
        scheduler.step()
        state={'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng_state':capture_rng_state(),'history':history,'config':config,'binding':binding}
        commit_epoch(output/'epochs',epoch,binding,lambda path:atomic_torch_save(state,path))
        write_csv(output/'history.csv',history)
        print(f'job={job["job_id"]} epoch={epoch}/65 train_loss={stats["loss"]:.5f} train_top1={stats["top1"]:.5f} epoch_seconds={stats["seconds"]:.2f}',flush=True)
    require(state is not None and state['epoch']==65,'No final epoch65 state')
    final=output/'final.pt'
    atomic_torch_save({'epoch':65,'model':model.state_dict(),'config':config,'binding':binding},final)
    write_csv(output/'history.csv',history)
    atomic_json(output/'training_summary.json',{'schema':'aura-ksp.stream3-final-training-summary.v1','status':'COMPLETE','created_utc':utc(),'binding':binding,'epochs_completed':65,'training_samples':54468,'base_train_samples':38670,'old_validation_training_samples':15798,'checkpoint_rule':'final_epoch_65_no_validation_monitoring','old_validation_metrics_computed':False,'test_tensors_read':False,'training_seconds_from_epochs':sum(r['epoch_seconds'] for r in history)})
    write_receipt(args.run_root,job['job_id'],args.data_root,[final,output/'history.csv',output/'training_summary.json',initial],time.monotonic()-started)


def expected_test_binding(args):
    models={}
    for stream in STREAMS:
        verify_receipt(args.run_root,'train_'+stream,args.data_root)
        models[stream]=sha256_file(args.run_root/'checkpoints'/stream/'final.pt')
    return {**runtime_binding(args.data_root),'model_sha256':models,
        'test_manifest_canonical_sha256':protocol()['data']['test_canonical_sha256'],
        'event_id':protocol()['protocol_id']+'-single-test-event'}


def evaluate(args,job):
    import torch
    from aura_har.models import build_model
    from aura_har.utils.reproducibility import load_checkpoint_on_cpu,seed_everything
    from final_dataset import FinalDataset,loader_for
    started=time.monotonic()
    marker=args.project_root/'runs/AURA_KSP_NTU120_XSET_TEST_EVENT.json'
    event=expected_test_binding(args)
    verify_test_event(marker,event)
    train_job=next(j for j in plan()['jobs'] if j['job_id']=='train_'+job['stream'])
    config=train_job['config'];seed=int(config['experiment']['seed']);seed_everything(seed,True)
    checkpoint=args.run_root/'checkpoints'/job['stream']/'final.pt'
    state=load_checkpoint_on_cpu(checkpoint)
    require(state['binding']==job_binding(train_job['job_id'],args.data_root) and state['epoch']==65,'Wrong final model')
    model=build_model(config).to('cuda');model.load_state_dict(state['model']);model.eval()
    rows=read_jsonl(args.data_root/'TEST.jsonl')
    output=args.run_root/job['output_relative'];chunks=output/'chunks';chunks.mkdir(parents=True,exist_ok=True)
    base_binding={**job_binding(job['job_id'],args.data_root),'checkpoint_sha256':sha256_file(checkpoint),'test_event_binding':event}
    chunk_paths=[]
    for first in range(0,len(rows),1024):
        stop=min(first+1024,len(rows));number=first//1024
        path=chunks/f'chunk_{number:04d}.npz';commit=chunks/f'chunk_{number:04d}.commit.json'
        binding={**base_binding,'first':first,'stop':stop}
        if commit.exists():
            saved=load_json(commit)
            require(saved['binding']==binding and sha256_file(path)==saved['sha256'],'Corrupt/mismatched committed evaluation chunk')
            print(f'VERIFIED CHUNK {job["job_id"]} {stop}/{len(rows)}',flush=True)
        else:
            dataset=FinalDataset(args.data_root,args.sparse_root,job['stream'],seed,'test',rows=rows[first:stop],test_marker=marker,test_binding=event)
            loader=loader_for(dataset,seed,False)
            logits=[];labels=[];setups=[];ids=[];indices=[]
            with torch.inference_mode():
                for batch in loader:
                    out=model(batch['skeleton'].to('cuda',non_blocking=True)).cpu().numpy()
                    require(np.isfinite(out).all(),'Nonfinite test logits')
                    logits.append(out.astype(np.float32));labels.append(batch['label'].numpy())
                    setups.append(batch['setup'].numpy());ids.extend(batch['sample_id']);indices.append(batch['indices'].numpy())
            require(ids==[r['sample_id'] for r in rows[first:stop]],'Evaluation sample order changed')
            atomic_npz(path,logits=np.concatenate(logits),labels=np.concatenate(labels),setups=np.concatenate(setups),sample_ids=np.asarray(ids),selected_indices=np.concatenate(indices))
            create_once(commit,{'binding':binding,'sha256':sha256_file(path),'created_utc':utc()})
            print(f'TEST CHUNK {job["job_id"]} {stop}/{len(rows)}',flush=True)
        chunk_paths.append(path)
    combined={name:[] for name in ('logits','labels','setups','sample_ids','selected_indices')}
    for path in chunk_paths:
        with np.load(path,allow_pickle=False) as npz:
            for name in combined:combined[name].append(npz[name].copy())
    combined={name:np.concatenate(values) for name,values in combined.items()}
    require(np.array_equal(combined['labels'],[int(r['label']) for r in rows]),'Test labels misaligned')
    require(np.array_equal(combined['setups'],[int(r['setup']) for r in rows]),'Test setups misaligned')
    atomic_npz(output/'predictions.npz',**combined)
    atomic_json(output/'evaluation_provenance.json',{'schema':'aura-ksp.stream3-final-evaluation-provenance.v1','status':'COMPLETE','binding':base_binding,'samples':len(rows),'sealed_event_marker_sha256':sha256_file(marker),'chunks':len(chunk_paths),'test_used':True,'old_validation_evaluated':False,'checkpoint_rule':'final_epoch_65','candidate_selection_performed':False})
    write_receipt(args.run_root,job['job_id'],args.data_root,[output/'predictions.npz',output/'evaluation_provenance.json'],time.monotonic()-started)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data-root','sparse-root','run-root','project-root','authorization'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--job-id',required=True)
    args=p.parse_args()
    assert_runtime(args)
    jobs=[j for j in plan()['jobs'] if j['job_id']==args.job_id]
    require(len(jobs)==1 and jobs[0]['action'] in ('train','evaluate'),'Unknown training/evaluation job')
    if receipt_path(args.run_root,args.job_id).is_file():
        verify_receipt(args.run_root,args.job_id,args.data_root);print('ALREADY_VERIFIED_COMPLETE');return
    sys.path.insert(0,str(kit_root()/'project/src'))
    import torch
    require(torch.__version__.startswith('2.11.0') and torch.version.cuda=='12.8','Unexpected project PyTorch environment')
    torch.set_num_threads(1)
    if jobs[0]['action']=='train':train(args,jobs[0])
    else:evaluate(args,jobs[0])


if __name__=='__main__':main()
