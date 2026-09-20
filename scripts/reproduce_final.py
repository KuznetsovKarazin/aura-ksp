"""CPU-only reproduction from curated saved evidence. No forward or training."""
import argparse,gzip,json,hashlib,platform
from pathlib import Path
import numpy as np
from offline_math import fixed_fusion,endpoints,paired_bootstrap,assess
from checkpoint_inspect import inspect
from verify_release import verify,sha

def need(value,message):
    if not value:raise ValueError(message)
def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    need(not a.output.exists(),'Choose a new output directory')
    count=verify(a.assets);root=a.assets/'final';data=root/'data_binding';run=root/'run'
    protocol=read(root/'kit/protocol/PROTOCOL.json');plan=read(root/'kit/PLAN.json');binding=read(data/'DATA_BINDING.json')
    event=read(next((root/'sealed_event').glob('*.json')))
    records=[json.loads(x) for x in (data/'TEST.jsonl').read_text().splitlines() if x.strip()]
    y=np.array([r['label'] for r in records]);ids=np.array([r['sample_id'] for r in records]);setups=np.array([r['setup'] for r in records])
    need(len(y)==59477 and len(set(ids))==len(ids),'Test coverage')
    nominal=[]
    for r in records:
        length=r['sequence_length'];bias=int((1.-.95)*length/2)
        nominal.append(np.rint(np.linspace(bias,length-bias-1,64)).astype(np.int64))
    nominal=np.stack(nominal)
    recs={f.stem:read(f) for f in (run/'receipts').glob('*.json')}
    need(set(recs)=={x['job_id'] for x in plan['jobs']} and len(recs)==9,'Receipts')
    n_outputs=0
    for name,r in recs.items():
        need(r['status']=='COMPLETE' and r['binding']['job_id']==name,'Receipt status')
        for rel,h in r['outputs'].items():
            dest=(run/rel).resolve();need(dest.is_relative_to(run.resolve()),'Receipt path')
            need(sha(dest)==h,'Receipt output '+rel);n_outputs+=1
    logits=[];models=[]
    for i,s in enumerate(('joint','bone','joint_motion','bone_motion')):
        model=run/'checkpoints'/s/'final.pt';m=inspect(model)
        need(sha(model)==event['binding']['model_sha256'][s],'Model event pin')
        need(m['epoch']==65 and m['config']==plan['jobs'][i]['config'],'Model configuration')
        need(m['binding']==recs['train_'+s]['binding'],'Model receipt binding')
        models.append({'stream':s,'sha256':sha(model),'epoch':m['epoch']})
        with np.load(run/'predictions'/s/'predictions.npz',allow_pickle=False) as z:
            for k,v in [('labels',y),('setups',setups),('sample_ids',ids),('selected_indices',nominal)]:need(np.array_equal(z[k],v),'Alignment '+s+' '+k)
            logits.append(z['logits'].copy())
    rl,cl=fixed_fusion(logits);rp=rl.argmax(1);cp=cl.argmax(1)
    draws=np.asarray(json.loads(gzip.decompress((data/'BOOTSTRAP_DRAW_INDICES.json.gz').read_bytes())),dtype='<i8')
    need(draws.shape==(10000,16) and hashlib.sha256(draws.tobytes()).hexdigest()==binding['bootstrap']['raw_sha256'],'Draws pin')
    dist=paired_bootstrap(y,rp,cp,setups,draws,protocol['statistics']['setup_order'])
    rm,cm=endpoints(y,rp),endpoints(y,cp);gates=assess(cm-rm,dist)
    summary=read(run/'analysis/FINAL_TEST_SUMMARY.json')
    for i,g in enumerate(gates):
        spec=protocol['statistics']['quality_gate_endpoints'][i]
        need(g['endpoint']==spec['name'] and g['margin']==spec['margin'],'Frozen endpoint')
        for k,v in g.items():
            old=summary['endpoints'][i][k]
            need(v==old if isinstance(v,(str,bool)) else abs(v-old)<2e-15,'Endpoint '+k)
        need(abs(summary['endpoints'][i]['reference']-rm[i])<2e-15 and abs(summary['endpoints'][i]['candidate']-cm[i])<2e-15,'Absolute metric')
    rc=np.zeros((120,120),dtype=np.int64);cc=rc.copy();np.add.at(rc,(y,rp),1);np.add.at(cc,(y,cp),1)
    expected=dict(sample_ids=ids,labels=y,setups=setups,reference_logits=rl,candidate_logits=cl,reference_predictions=rp,candidate_predictions=cp,reference_confusion=rc,candidate_confusion=cc,bootstrap_distribution=dist)
    comparisons={}
    with np.load(run/'analysis/FINAL_TEST_PAIRED_RESULTS.npz',allow_pickle=False) as z:
        need(set(z.files)==set(expected),'Derived schema')
        for k,v in expected.items():
            exact=np.array_equal(v,z[k]);need(np.allclose(v,z[k],rtol=0,atol=2e-15) if k=='bootstrap_distribution' else exact,'Derived '+k)
            comparisons[k]={'exact_equal':bool(exact)}
    passed=all(g['passed'] for g in gates);need(summary['gate_passed']==passed,'Gate')
    report={'status':'CURATED_SAVED_OUTPUT_REPRODUCTION_PASS','gate_passed':passed,'manifest_files':count,'receipts':len(recs),'receipt_outputs':n_outputs,'models':models,'endpoints':gates,'derived_comparisons':comparisons,'reference_correct':int((rp==y).sum()),'candidate_correct':int((cp==y).sum()),'python':platform.python_version(),'numpy':np.__version__,'new_training':False,'new_test_forward':False,'scope':'Curated final saved outputs, not a re-audit of unavailable original TAR or provider control records.'}
    a.output.mkdir(parents=True);(a.output/'REPRODUCTION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
