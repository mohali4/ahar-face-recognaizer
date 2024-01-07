from numpy import ndarray
from copy import deepcopy 
from typing import Iterable
class Frame:
    def __init__(self, buffer:ndarray, time:int):
        self.buffer = buffer
        self.time = time
    def __deepcopy__(self,memo=None):
        return type(self)(self.buffer.copy(),deepcopy(self.time))
    def copy(self):
        return deepcopy(self)
class Face:
    def __init__(self, points:Iterable[int], _id:int, time:int):
        self.points = points
        self.id = _id
        self.time = time
    def __deepcopy__(self,memo=None):
        return type(self)(deepcopy([x for x in self.points]),self.name,deepcopy(self.time))
    def copy(self):
        return deepcopy(self)