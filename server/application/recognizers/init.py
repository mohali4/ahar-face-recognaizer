import dlib
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SyncManager
import time
from ..mprocess import PPool, Process, Synchronized, sharedValue, sharedArray
from ..managers import frameManager, faceManager
detector = dlib.get_frontal_face_detector()
# status:
#   0:
#       released
#   1:
#       start calc
#   2:
#       calculating
FREE = 0
GO_CALC = 1
CALCULATING = 2

def recognizerLoop(status:sharedValue, SHframe:frameManager, SHfaces:faceManager, ExitFlag, BUFFER_COEFICIENT):
    frame = None
    frameTime = 0
    def Exited():
        return ExitFlag.get()
    while True:
        # print('running')
        if Exited():
            break
        match status.get() :
            case 0: #FREE
                time.sleep(0.1)
            case 1: #GO_CALC
                frame, frameTime = SHframe.copy()
                status.set(2)
            case 2: #CALCULATING
                faces = detector(frame)
                # cv2.imwrite('/home/mohali/image.png',frame)
                for face in faces:
                    # x,y,w,h = face
                    # faceBuffer = tuple(map(int,np.array((x,y,x+w,y+h),dtype=np.int32)/BUFFER_COEFICIENT))
                    faceBuffer = (face.left(),face.top(),face.right(),face.bottom())
                    SHfaces.push(faceBuffer, time=frameTime)
                    print('found a face ', time.time())
                status.set(0)
                # cv2.imshow("Face Recognition", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                    # break



class recognizer(Process):
    def __init__(self, manager:SyncManager, *args):
        self.status = sharedValue('i',0,manager=manager)
        super().__init__(
            target = recognizerLoop,
            args = (self.status, *args)
        )
    @property
    def is_free(self):
        return self.status.get() == 0
    
    def render(self):
        self.status.set(1)

class recognizersPool(PPool):
    all:list[recognizer]
    processModel = recognizer
    def __init__(self,*args, **kwargs):
        super().__init__(**kwargs)
        self.init(*args)
    def make_process(self,*args,**wargs):
        return self.processModel(
            *args
        )
    @property
    def has_free(self):
        for p in self.all:
            if p.is_free :
                return True
        return False
    def render(self):
        for p in self.all:
            if p.is_free:
                p.render()
                break