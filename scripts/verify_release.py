"""Verify release bytes without importing numerical or model libraries."""
import argparse, hashlib
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify(root):
    root=root.resolve();count=0;seen=set()
    for line in (root/'MANIFEST_SHA256.txt').read_text().splitlines():
        h,rel=line.split('  ',1);p=(root/rel).resolve()
        if not p.is_relative_to(root) or rel in seen:raise ValueError('Invalid manifest path')
        seen.add(rel)
        if not p.is_file() or sha(p)!=h:raise ValueError('Hash mismatch: '+rel)
        count+=1
    if not count:raise ValueError('Empty manifest')
    return count
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=a.parse_args()
    print('MANIFEST_PASS',verify(args.root))
