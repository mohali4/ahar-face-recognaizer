from numpy import ndarray
from copy import deepcopy 
class Frame:
    def __init__(self, buffer:ndarray, time:int):
        self.buffer = buffer
        self.time = time
    def __deepcopy__(self,memo=None):
        return type(self)(self.buffer.copy(),deepcopy(self.time))