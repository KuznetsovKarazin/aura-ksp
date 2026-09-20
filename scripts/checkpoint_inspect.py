"""Restricted inspection of torch ZIP checkpoints WITHOUT importing torch.

Only OrderedDict and inert local replacements for two storage types and one
tensor rebuild function are accepted. No arbitrary pickle global is imported.
Every tensor is validated against raw ZIP storage bytes and checked for finite
values on CPU. This is structural/value inspection, not model execution.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import pickle
import zipfile
import numpy as np


class FloatStorage: pass
class LongStorage: pass


@dataclass
class Storage:
    key: str
    dtype: str
    count: int
    location: str


@dataclass
class Tensor:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple
    requires_grad: bool


def rebuild(storage, offset, size, stride, requires_grad, hooks, metadata=None):
    if not isinstance(storage, Storage) or hooks not in (None, OrderedDict()):
        raise ValueError('Unsupported tensor storage/hooks')
    return Tensor(storage, int(offset), tuple(size), tuple(stride), bool(requires_grad))


class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('collections','OrderedDict'):OrderedDict,
                   ('torch','FloatStorage'):FloatStorage,
                   ('torch','LongStorage'):LongStorage,
                   ('torch._utils','_rebuild_tensor_v2'):rebuild}
        if (module,name) not in allowed:
            raise pickle.UnpicklingError('Refused global: '+module+'.'+name)
        return allowed[module,name]

    def persistent_load(self, pid):
        if not isinstance(pid,tuple) or len(pid)!=5 or pid[0]!='storage':
            raise pickle.UnpicklingError('Unsupported persistent ID')
        _, kind, key, location, count = pid
        if kind not in (FloatStorage,LongStorage) or not str(key).isdigit() or type(count) is not int or count<0:
            raise pickle.UnpicklingError('Unsupported storage descriptor')
        return Storage(str(key), '<f4' if kind is FloatStorage else '<i8',count,str(location))


def inspect(path):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None: raise ValueError('Checkpoint ZIP CRC')
        names=z.namelist()
        if len(names)!=len(set(names)): raise ValueError('Duplicate checkpoint member')
        pk=[n for n in names if n.endswith('/data.pkl')]
        if len(pk)!=1: raise ValueError('Expected one data.pkl')
        root=pk[0].rsplit('/',1)[0]
        if z.read(root+'/byteorder')!=b'little':raise ValueError('Unexpected byteorder')
        obj=Restricted(io.BytesIO(z.read(pk[0]))).load()
        if type(obj) is not dict or set(obj)!={'epoch','model','config','binding'}:raise ValueError('Checkpoint schema')
        if not isinstance(obj['model'],OrderedDict):raise ValueError('Expected state dict')
        inventory=[];storage_cache={};float_elements=0;int_elements=0
        for name,t in obj['model'].items():
            if type(name) is not str or not isinstance(t,Tensor):raise ValueError('Unexpected state entry')
            if len(t.shape)!=len(t.stride) or any(type(n) is not int or n<0 for n in t.shape+t.stride) or t.offset<0:raise ValueError('Tensor geometry')
            s=t.storage;dtype=np.dtype(s.dtype)
            if s.key not in storage_cache:
                raw=z.read(root+'/data/'+s.key)
                if len(raw)!=s.count*dtype.itemsize:raise ValueError('Storage size')
                storage_cache[s.key]=(s,raw)
            saved,raw=storage_cache[s.key]
            if saved!=s:raise ValueError('Inconsistent shared storage')
            count=int(np.prod(t.shape,dtype=np.int64))
            if count:
                max_index=t.offset+sum((n-1)*st for n,st in zip(t.shape,t.stride))
                if max_index>=s.count:raise ValueError('Tensor exceeds storage')
            view=np.ndarray(t.shape,dtype=dtype,buffer=raw,offset=t.offset*dtype.itemsize,strides=tuple(st*dtype.itemsize for st in t.stride))
            if not np.isfinite(view).all():raise ValueError('Nonfinite checkpoint tensor: '+name)
            inventory.append({'name':name,'shape':list(t.shape),'stride':list(t.stride),'dtype':s.dtype,'elements':count,
                'value_sha256':hashlib.sha256(view.tobytes(order='C')).hexdigest(),'storage_key':s.key})
            if dtype.kind=='f':float_elements+=count
            else:int_elements+=count
        referenced={root+'/data/'+key for key in storage_cache}
        if referenced!={n for n in names if n.startswith(root+'/data/')}:raise ValueError('Unreferenced/missing storage')
        return {'epoch':obj['epoch'],'config':obj['config'],'binding':obj['binding'],
            'inspection':'RESTRICTED_INERT_DESERIALIZATION_AND_RAW_STORAGE_CPU_NO_TORCH_NO_FORWARD',
            'state_tensors':len(inventory),'storage_files':len(storage_cache),'all_tensor_values_finite':True,
            'float_state_elements':float_elements,'int_state_elements':int_elements,'tensor_inventory':inventory}
