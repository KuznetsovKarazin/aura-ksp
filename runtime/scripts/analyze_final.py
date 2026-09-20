"""Frozen paired setup-cluster analysis of the one authorized XSet test event."""
from __future__ import annotations
import argparse
import gzip
import json
import time
from pathlib import Path
import numpy as np
from final_common import (STREAMS, atomic_bytes, atomic_json, job_binding, load_binding,
    plan, protocol, read_jsonl, receipt_path, require, sha256_file, utc, verify_receipt,
    write_csv, write_receipt)
from final_job import assert_runtime,atomic_npz,expected_test_binding


def confusion(labels,predictions):
    return np.bincount(np.asarray(labels,dtype=np.int64)*120+np.asarray(predictions,dtype=np.int64),minlength=14400).reshape(120,120)


def macro_f1(matrix):
    den=matrix.sum(0)+matrix.sum(1)
    return float(np.divide(2.0*np.diag(matrix),den,out=np.zeros(120,dtype=np.float64),where=den>0).mean())


def metric_values(labels,predictions):
    new=labels>=60
    return {'all_top1':float(np.mean(labels==predictions)),
        'a61_a120_top1':float(np.mean(labels[new]==predictions[new])) if new.any() else None,
        'macro_f1_all120':macro_f1(confusion(labels,predictions))}


def bootstrap_distribution(labels,reference,candidate,setups,draws,setup_order):
    require(np.array_equal(np.unique(setups),np.asarray(setup_order)),'Setup order does not match frozen test clusters')
    require(draws.shape==(10000,len(setup_order)) and draws.min()>=0 and draws.max()<len(setup_order),'Invalid bootstrap shape/indices')
    counts=np.stack([(draws==i).sum(1) for i in range(len(setup_order))],axis=1).astype(np.float64)
    masks=[setups==s for s in setup_order];new=labels>=60
    n_all=np.array([m.sum() for m in masks],dtype=np.float64)
    n_new=np.array([(m&new).sum() for m in masks],dtype=np.float64)
    delta=(candidate==labels).astype(np.int64)-(reference==labels).astype(np.int64)
    d_all=np.array([delta[m].sum() for m in masks],dtype=np.float64)
    d_new=np.array([delta[m&new].sum() for m in masks],dtype=np.float64)
    require(bool(np.all(counts@n_new>0)),'Undefined A61-A120 bootstrap draw')
    result=np.empty((len(draws),3),dtype=np.float64)
    result[:,0]=(counts@d_all)/(counts@n_all);result[:,1]=(counts@d_new)/(counts@n_new)
    rc=np.stack([confusion(labels[m],reference[m]) for m in masks])
    cc=np.stack([confusion(labels[m],candidate[m]) for m in masks])
    for start in range(0,len(draws),100):
        batch=counts[start:start+100]
        r=np.einsum('bs,sij->bij',batch,rc,optimize=True);c=np.einsum('bs,sij->bij',batch,cc,optimize=True)
        for i in range(len(batch)):result[start+i,2]=macro_f1(c[i])-macro_f1(r[i])
    return result


def assessment(estimate,distribution,margin,alpha):
    centered=np.asarray(distribution,dtype=np.float64)-float(estimate)
    lcb=float(estimate)-float(np.quantile(centered,1-alpha,method='higher'))
    p=(1.0+float(np.sum(centered>=float(estimate)-float(margin))))/(len(centered)+1)
    return {'one_sided_lcb':lcb,'margin_null_p':p,'bonferroni_adjusted_p':min(1.0,3*p),'passed':bool(lcb>margin)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data-root','sparse-root','run-root','project-root','authorization'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();assert_runtime(args)
    if receipt_path(args.run_root,'analysis').exists():
        verify_receipt(args.run_root,'analysis',args.data_root);print('ANALYSIS_ALREADY_VERIFIED');return
    from final_common import verify_test_event
    verify_test_event(args.project_root/'runs/AURA_KSP_NTU120_XSET_TEST_EVENT.json',expected_test_binding(args))
    started=time.monotonic();pr=protocol();binding=load_binding(args.data_root)
    expected=read_jsonl(args.data_root/'TEST.jsonl')
    ids=np.asarray([r['sample_id'] for r in expected]);labels=np.asarray([int(r['label']) for r in expected],dtype=np.int64);setups=np.asarray([int(r['setup']) for r in expected],dtype=np.int64)
    components=[];source_indices=None;stream_metrics=[]
    for stream in STREAMS:
        verify_receipt(args.run_root,'evaluate_'+stream,args.data_root)
        with np.load(args.run_root/'predictions'/stream/'predictions.npz',allow_pickle=False) as a:
            require(np.array_equal(a['sample_ids'],ids) and np.array_equal(a['labels'],labels) and np.array_equal(a['setups'],setups),'Stream prediction alignment mismatch')
            require(a['logits'].shape==(59477,120) and np.isfinite(a['logits']).all(),'Invalid final logits')
            if source_indices is None:source_indices=a['selected_indices'].copy()
            else:require(np.array_equal(a['selected_indices'],source_indices),'Streams used different temporal indices')
            values=np.asarray(a['logits'],dtype=np.float32).copy();components.append(values)
            stream_metrics.append({'stream':stream,**metric_values(labels,values.argmax(1))})
    stacked=np.stack(components,axis=1)
    reference_logits=np.einsum('nsc,s->nc',stacked,np.asarray([0.3,0.3,0.2,0.2],dtype=np.float32))
    candidate_logits=np.einsum('nsc,s->nc',stacked[:,:3],np.asarray([0.375,0.375,0.25],dtype=np.float32))
    reference=reference_logits.argmax(1);candidate=candidate_logits.argmax(1)
    rm=metric_values(labels,reference);cm=metric_values(labels,candidate)
    draws=np.asarray(json.loads(gzip.decompress((args.data_root/'BOOTSTRAP_DRAW_INDICES.json.gz').read_bytes())),dtype='<i8')
    require(__import__('hashlib').sha256(draws.tobytes()).hexdigest()==binding['bootstrap']['raw_sha256'],'Bootstrap raw hash mismatch')
    distribution=bootstrap_distribution(labels,reference,candidate,setups,draws,pr['statistics']['setup_order'])
    alpha=pr['statistics']['per_endpoint_alpha'];endpoints=[]
    for i,spec in enumerate(pr['statistics']['quality_gate_endpoints']):
        name=spec['name'];estimate=cm[name]-rm[name]
        endpoints.append({'endpoint':name,'reference':rm[name],'candidate':cm[name],'delta':estimate,'margin':spec['margin'],'per_endpoint_alpha':alpha,**assessment(estimate,distribution[:,i],spec['margin'],alpha)})
    passed=all(r['passed'] for r in endpoints)
    per_setup=[]
    for setup in pr['statistics']['setup_order']:
        m=setups==setup; r=metric_values(labels[m],reference[m]);c=metric_values(labels[m],candidate[m])
        per_setup.append({'setup':setup,'samples':int(m.sum()),'reference_top1':r['all_top1'],'candidate_top1':c['all_top1'],'delta_top1':c['all_top1']-r['all_top1'],'reference_a61_a120_top1':r['a61_a120_top1'],'candidate_a61_a120_top1':c['a61_a120_top1']})
    rc=confusion(labels,reference);cc=confusion(labels,candidate);classes=[]
    for i in range(120):
        support=int(rc[i].sum());rd=int(rc[:,i].sum()+support);cd=int(cc[:,i].sum()+support)
        classes.append({'action_id':f'A{i+1:03d}','support':support,'reference_recall':float(rc[i,i]/support),'candidate_recall':float(cc[i,i]/support),'delta_recall':float((cc[i,i]-rc[i,i])/support),'reference_f1':float(2*rc[i,i]/rd) if rd else 0.0,'candidate_f1':float(2*cc[i,i]/cd) if cd else 0.0})
    output=args.run_root/'analysis';output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'01_primary_endpoints.csv',endpoints);write_csv(output/'02_per_setup.csv',per_setup)
    write_csv(output/'03_per_class.csv',classes);write_csv(output/'04_single_stream_metrics.csv',stream_metrics)
    atomic_npz(output/'FINAL_TEST_PAIRED_RESULTS.npz',sample_ids=ids,labels=labels,setups=setups,reference_logits=reference_logits,candidate_logits=candidate_logits,reference_predictions=reference,candidate_predictions=candidate,reference_confusion=rc,candidate_confusion=cc,bootstrap_distribution=distribution)
    summary={'schema':'aura-ksp.stream3-final-xset-summary.v1','status':'FINAL_XSET_QUALITY_GATE_PASS' if passed else 'FINAL_XSET_QUALITY_GATE_FAIL_CLOSED','created_utc':utc(),'binding':job_binding('analysis',args.data_root),'samples':59477,'endpoints':endpoints,'gate_passed':passed,
      'paired_decisions':{'reference_correct':int((reference==labels).sum()),'candidate_correct':int((candidate==labels).sum()),'lost_correct':int(((reference==labels)&(candidate!=labels)).sum()),'gained_correct':int(((reference!=labels)&(candidate==labels)).sum())},
      'bootstrap':{**binding['bootstrap'],'family_alpha':0.05,'per_endpoint_alpha':alpha,'method':'paired_setup_cluster_centered_basic_higher_quantile'},
      'historical_status':{'development':'LATENCY_GATE_PASS_QUALITY_CONFIRMATION_PASS','original_k32':'FAILED_CLOSED','latency_repeated':False},
      'data_use':{'old_validation_used_for_training':True,'old_validation_evaluated_or_used_for_selection':False,'xset_test_event_count':1,'test_used':True,'candidate_search':False,'margins_changed':False},
      'interpretation':'Within NTU120 XSet. NTU120 includes original NTU60 records previously used in the project; A61-A120 supports the strongest untouched-data interpretation under the verified access history. No external-dataset or SOTA claim.'}
    atomic_json(output/'FINAL_TEST_SUMMARY.json',summary)
    text=['# AURA-KSP: финальный XSet test','',f"Статус: `{summary['status']}`",'', '| Endpoint | Full-4, % | Stream-3, % | Delta, п.п. | LCB, п.п. | Margin, п.п. | PASS |','|---|---:|---:|---:|---:|---:|---|']
    for e in endpoints:text.append(f"| {e['endpoint']} | {100*e['reference']:.4f} | {100*e['candidate']:.4f} | {100*e['delta']:.4f} | {100*e['one_sided_lcb']:.4f} | {100*e['margin']:.1f} | {e['passed']} |")
    text+=['','Old validation использована только в составе final training. Test открыт одним фиксированным событием; четыре потоковых файла служат обеим fusion.','Отрицательный K32 gate сохранён. Latency microbenchmark не повторялся.','Результат публикуется независимо от PASS/FAIL. Настройка по test, изменение margins и новый кандидат не разрешены.','NTU120 включает NTU60; A61–A120 имеют наиболее сильную untouched-data интерпретацию. Это не внешний dataset.','']
    atomic_bytes(output/'REPORT_RU.md','\n'.join(text).encode('utf-8'))
    files=[output/n for n in ['01_primary_endpoints.csv','02_per_setup.csv','03_per_class.csv','04_single_stream_metrics.csv','FINAL_TEST_PAIRED_RESULTS.npz','FINAL_TEST_SUMMARY.json','REPORT_RU.md']]
    write_receipt(args.run_root,'analysis',args.data_root,files,time.monotonic()-started)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
