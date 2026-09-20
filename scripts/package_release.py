"""Build verified release archives after intentional edits. No network operations."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from verify_release import sha

def files(root):
    return sorted(p for p in root.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.git','.venv','.reproduced','.pytest_cache']))
def manifest(root):
    entries=[p for p in files(root) if p!=root/'MANIFEST_SHA256.txt']
    (root/'MANIFEST_SHA256.txt').write_text(''.join(sha(p)+'  '+p.relative_to(root).as_posix()+'\n' for p in entries),encoding='utf-8')
def pack(root,out):
    if out.exists():raise ValueError('Output exists: '+str(out))
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files(root):
            info=zipfile.ZipInfo(root.name+'/'+p.relative_to(root).as_posix(),date_time=(2026,9,6,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,p.read_bytes())
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
    out.with_name(out.name+'.sha256').write_text(sha(out)+'  '+out.name+'\n')
    return {'filename':out.name,'bytes':out.stat().st_size,'sha256':sha(out)}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    need=a.output_dir.resolve().is_relative_to(a.root.resolve())
    if need:raise ValueError('Place archives outside the source tree')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    result=[]
    for name in ['repository','research-assets']:
        manifest(a.root/name);result.append(pack(a.root/name,a.output_dir/('AURA-KSP_'+name+'_v1.0.0-rc1.zip')))
    manifest(a.root);result.append(pack(a.root,a.output_dir/(a.root.name+'.zip')))
    print(json.dumps(result,indent=2))
