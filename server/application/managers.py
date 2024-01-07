from .mprocess import sharedValue,sharedArray
import cv2
import time
import copy
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SyncManager
import math, statistics
from .timer import now
from .models import Frame, Face
FACE_TYPE = list[int]|tuple[int]
NoneType = type(None)

class frameManager:
    def __init__ (self,shape):
        self.shape = shape
        self._status = sharedValue('L',0)
        self.size = 1
        for s in shape:
            self.size = self.size * s
        self.buffer = sharedArray('B',self.size,noLock=True)
    def now(self):
        return int(round(time.time(),2) * 100)
    def set(self, frame:Frame):
        # resized = cv2.resize(buffer,self.shape)
        flat = frame.buffer.reshape((self.size,))
        self.buffer.set(flat)
        self._status.set(frame.time)
        # resh = flat.reshape(self.shape)
        # cv2.imshow("Face Recognition", resh)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     exit()

    def get(self)->Frame:
        return Frame(
            np.frombuffer( 
                self.buffer.get(),
                dtype=np.uint8
            ).reshape(self.shape),
            self._status.get()
        )
    def copy(self):
        buffer = copy.deepcopy(
            self.get()
        )
        print(buffer[1])
        return buffer
    @property
    def status(self):
        return self._status.get()
class faceManager:
    buffer_size = 6
    expire=20
    def __init__(self,manager:SyncManager,settings:dict):
        self._q = manager.list()
        self.lastUpdate = manager.Value('L',0)
        self.lock = manager.Lock()
        self.locked = False
        self.expire = settings['DETECT_EXPIRE']
    def now(self):
        return now()
    @property
    def q(self)->list[Face]:
        while None in self._q : time.sleep(0.01)
        s = self.buffer_size
        faces = []
        for _face in np.array(
            self._q,
            dtype=np.int64
        ).reshape((len(self._q)//s,s)):
            faces.append(
                Face(
                    _face[2:],_face[1],_face[0]
                )
            )
        return faces
    def _find_alternative(self,item:Face):
        def is_aternative(one,two):
            x11, y11, x12, y12 = one
            x21, y21, x22, y22 = two
            if abs((x11-x12)+(y11-y12))>abs((x21-x22)+(y21-y22)):
                big = one
                les = two
            else:
                les = one
                big = two
            bx1, by1, bx2, by2=big
            lx1, ly1, lx2, ly2=les
            return (
                statistics.mean(
                    [
                        math.sqrt(
                            (bx1-lx1)**2 + (by1-ly1)**2
                        )
                        ,
                        math.sqrt(
                            (bx2-lx2)**2 + (by1-ly1)**2
                        )
                        ,
                        math.sqrt(
                            (bx1-lx1)**2 + (by2-ly2)**2
                        )
                        ,
                        math.sqrt(
                            (bx1-lx2)**2 + (by1-ly2)**2
                        )
                    ]
                ) < statistics.mean(
                    [ abs(bx1 - bx2), abs(by1 - by2) ]
                )
            )
        for condidate in self.q:
            if is_aternative(item.points, condidate.points):
                return condidate
        return None
    def sync(self, faces:list[Face]):
        self.acquire()
        for face in faces:
            self.push(face,noLock=True)
        self.release()
    def updateTime(self,time:int):
        if time > self.lastUpdate.get():
            self.lastUpdate.set(time)
    def push(self, face:Face, noLock=False):
        if not noLock:
            self.acquire()
        # self.remove_olds()
        alt = self._find_alternative(face)
        if type(alt) != NoneType:
            self.remove(alt)
        self.updateTime(face.time)
        for x in self.face2buffer(face):
            self._q.append(x)
        if not noLock:
            self.release()
    def get(self):
        self.acquire()
        self.remove_olds()
        ret = copy.deepcopy(self.q)
        self.release()
        return ret
    def face2buffer(self, face:Face):
        return [
            face.time,
            face.id,
            x for x in face.points
        ]
    def remove(self,face:Face):
        s = self.buffer_size
        buffer = self.face2buffer(face)
        for i in range(0,self._q.__len__(),s):
            if self._q[i:i+s][:] == buffer[:]:
                self._q[:] = self._q[:i] + self._q[i+s:] + [0 for _ in range(s)]
                for _ in range(s): self._q.pop()
                break
    def remove_olds(self):
        def is_old(face:FACE_TYPE):
            birth = face[0]
            return birth + self.expire < self.now()
        for face in self.q:
            if is_old(face):
                self.remove(face)
    def acquire(self):
        if not self.locked or True:
            self.lock.acquire(blocking=True)
            self.locked = True
    def release(self):
        if self.locked or True:
            self.lock.release()
            
            self.locked = False
    @property
    def status(self):
        return copy.deepcopy(self.lastUpdate.get())