import dlib
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SyncManager
import time
from ..mprocess import PPool, Process, Synchronized, sharedValue, sharedArray
from ..managers import frameManager, faceManager
from ..models import Frame, Face

# detector = dlib.get_frontal_face_detector()
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

class baseDetector :
    gray = False
    resize = (1,1)
    def simplifyFrame(self,_frame:np.ndarray):
        if self.gray:
            _frame = cv2.cvtColor(_frame, cv2.COLOR_RGB2GRAY)
        newH = int(_frame[0]*self.resize[0])
        newW = int(_frame[1]*self.resize[1])
        _frame = cv2.resize(_frame, (newH, newW))
        return _frame
    def detect(self,buffer):
        return []
    def normalyzeFaces(self,):...
    def handle(self,frame:Frame):
        buffer = frame.buffer
        simplified = self.simplifyFrame(buffer)
        _faces = self.detect(simplified)
        faces = map(lambda _face:Face(_face[1:], _face[0], frame.time), _faces)
        return list(faces)


class baseRecognizer(Process):
    detector = baseDetector()
    def __init__(self, manager:SyncManager,settings:dict, status:sharedValue, SHframe:frameManager, SHfaces:faceManager, ExitFlag, *args, **wargs):
        self.status = sharedValue('i',0,manager=manager)
        self.frameManager = frameManager(settings['FRAME_SHAPE'])
        self.flag = status
        self.framem = SHframe
        self.facem = SHfaces
        self.exited = ExitFlag
        super().__init__(
            target = self.loop,
            args = (self.status, self.frameManager, *args),
            **wargs
        )
    @property
    def is_free(self):
        return self.flag.get() == 0
    
    @property
    def is_buffer_lock(self):
        return self.flag.get() == 1

    def simplifyFrame(self,frame:Frame)->Frame:
        return frame


    def getFrame(self)->Frame:
        frame = self.framem.get()
        simplifiedFrame = self.simplifyFrame(frame)
        return simplifiedFrame

    def loop(self):
        frame = None
        def Exited():
            return self.exited.get()
        while True:
            # print('running')
            if Exited():
                break
            match self.flag.get() :
                case 0: #FREE
                    time.sleep(0.01)
                case 1: #GO_CALC
                    frame = self.getFrame()
                    self.flag.set(2)
                case 2: #CALCULATING
                    faces = self.detector.handle(frame)
                    # cv2.imwrite('/home/mohali/image.png',frame)
                    self.facem.sync(faces)
                    self.flag.set(0)
                    # cv2.imshow("Face Recognition", frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                        # break



    def render(self, frame):
        self.frameManager.set(frame)
        self.status.set(1)

class baseRecognizerPool(PPool):
    all:list[baseRecognizer]
    processModel = baseRecognizer
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