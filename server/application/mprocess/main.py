from .base import (
    basePPool,
    baseProcess
)
from multiprocessing.sharedctypes import (
    Synchronized, 
    SynchronizedArray
)
from multiprocessing.managers import (
    SyncManager
)
import multiprocessing as mp
import numpy as np
import copy

class Process(baseProcess):
    ...

class PPool(basePPool):
    processModel = Process
    def __init__(self,target=None,args=tuple(),wargs={} , **kwargs):
        super().__init__(**kwargs)
        self._target = target
        self._args = args
        self._wargs= wargs
    @property
    def target(self):
        return self._target

    def _get_args(self, _input=None):
        return self._args
    
    def _get_wargs(self, _input=None):
        return self._wargs

class sharedValue:
    def __init__(self,_type,*args,manager:SyncManager=None):
        self.is_manager = True
        if manager:
            self.val = manager.Value(_type,*args)
        else:
            self.val = mp.Value(_type,*args)
            self.is_manager = False
    def set(self, item):
        if not self.is_manager:
            with self.val.get_lock():
                self.val.value = item
        else:
            self.val.set(item)
    def get(self):
        if not self.is_manager:
            with self.val.get_lock():
                ret = copy.deepcopy(self.val.value)
            return ret
        else:
            return self.val.get()
class sharedArray:
    def __init__(self,_type,*args):
        self.val = mp.Array(_type,*args)
    def set(self, item):
        with self.val.get_lock():
            buffer = np.frombuffer( self.val.get_obj(),dtype=np.int8)
            buffer[:] = item

    def get(self):
        return self.val.get_obj()

