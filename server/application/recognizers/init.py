import dlib
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SyncManager
import time
from ..mprocess import PPool, Process, Synchronized, sharedValue, sharedArray
from ..managers import frameManager, faceManager
from ..models import Frame

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
    def Exited():
        return ExitFlag.get()
    while True:
        # print('running')
        if Exited():
            break
        match status.get() :
            case 0: #FREE
                time.sleep(0.01)
            case 1: #GO_CALC
                frame = SHframe.get()
                status.set(2)
            case 2: #CALCULATING
                faces = detector(frame.buffer)
                # cv2.imwrite('/home/mohali/image.png',frame)
                for face in faces:
                    # x,y,w,h = face
                    # faceBuffer = tuple(map(int,np.array((x,y,x+w,y+h),dtype=np.int32)/BUFFER_COEFICIENT))
                    faceBuffer = (face.left(),face.top(),face.right(),face.bottom())
                    SHfaces.push(faceBuffer, time=frame.time)
                    print(f'{mp.current_process().name}: FAF {time.time():.2f}')
                status.set(0)
                # cv2.imshow("Face Recognition", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                    # break



class recognizer(Process):
    def __init__(self, manager:SyncManager,settings:dict, *args, **wargs):
        self.status = sharedValue('i',0,manager=manager)
        self.frameManager = frameManager(settings['FRAME_SHAPE'])
        super().__init__(
            target = recognizerLoop,
            args = (self.status, self.frameManager, *args),
            **wargs
        )
    @property
    def is_free(self):
        return self.status.get() == 0
    
    @property
    def is_buffer_lock(self):
        return self.status.get() == 1

    def render(self, frame):
        self.frameManager.set(frame)
        self.status.set(1)

class recognizersPool(PPool):
    all:list[recognizer]
    processModel = recognizer
    def __init__(self,*args, **kwargs):
        super().__init__(**kwargs)
        self.init(*args)
    def get_name(self):
        return f'recognizer-{self.len}'
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
    def render(self, frame:Frame):
        for p in self.all:
            if p.is_free:
                p.render(frame)
                break