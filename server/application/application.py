import cv2
import dlib
import os
import time
import numpy as np
import math
import copy
import multiprocessing as mp
from multiprocessing.managers import SyncManager
import threading
from .timer import now
from .mprocess import (
    PPool,
    sharedValue
)
from .recognizers import (
    baseRecognizer,
    baseRecognizerPool
)
from .managers import (
    faceManager,
    frameManager
)
from .models import Frame

# camera = 'http://192.168.149.216:8080/video'
FRAME_RENEW = 15
# camera = 0
camera = 'http://192.168.255.225:8080/video'
NoneType = type(None)
os.environ.setdefault('QT_QPA_PLATFORM','xcb')
# MAX_FRAME_BUFFER_SIZE = 480*640
class forkModel(threading.Thread):...
# class forkModel(mp.Process):...
class displayManager(forkModel):
    def __init__(self, faceM:faceManager, frameM:frameManager,settings:dict, ExitFlag):
        self.faceM = faceM
        self.frameM= frameM
        self.exited = ExitFlag
        self._lastframe = None
        self.settings = settings
        self.status = 0
        self.last_face_get = 0
        self.faces = []
        super().__init__(
            target=self.loop,
            name='displayManager'
        )
    def read_faces(self):
        if self.last_face_get + self.settings['READ_FACES_DELAY'] < self.faceM.lastUpdate.get():
            self.faces = self.faceM.get()
        if type(self.faces) == NoneType:
            return []
        for face in self.faces:
            yield face[1:]
    def last_frame(self) -> Frame:
        return self._lastframe
    def loop(self):
        def Exited():
            return self.exited.get() == 1
        def Exit():
            cv2.destroyAllWindows()
            self.exited.set(1)
    
        while True:
            if Exited():
                break
            faces = self.read_faces()
            frame = self.frameM.get().buffer
            for face in faces:
                x1, y1, x2, y2 = face[0], face[1], face[2], face[3]
                cv2.rectangle(frame, (x1,y1), (x2, y2), self.settings['FACE_BORDER_COLOR'], self.settings['FACE_BORDER'])
            # print('write faces: ',time.time()-inpt)
            self.status = now()
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        Exit()

class cameraManager(forkModel):
    def __init__(self, manager:SyncManager, settings:dict, frameM:frameManager, ExitFlag):
        self.frameM = frameM
        self.status = sharedValue('B',0)
        self.exited = ExitFlag
        super().__init__(
            target=self.daemon,
            name='cameraManager'
        )
    def init(self):
        self.cap = cv2.VideoCapture(camera)
    def daemon(self):
        self.init()
        self.loop()
    def loop(self):
        def Exited():
            return self.exited.get() == 1
        def Exit():
            self.cap.release()
            self.exited.set(1)
    
        while True:
            if Exited():
                break
            ret, frame = self.cap.read()
            self.frameM.set(Frame(frame, now()))
        Exit()

class taskManager(mp.Process):
    def __init__ (self, PP:baseRecognizerPool, faceM:faceManager, frameM:frameManager, ExitFlag):
        self.pp = PP
        self.faceM = faceM
        self.frameM = frameM
        self.exited = ExitFlag
        super().__init__(
            target=self.loop,
            name='taskManager'
        )

    def frameNow(self) -> Frame:
        return self.frameM.get()
    def faceUpdated(self):
        dmStatus = self.displayM.status
        if not dmStatus:
            return False
        return  dmStatus <= self.faceM.status
    def loop(self):
        def Exited():
            return self.exited.get() == 1
        def Exit():
            self.exited.set(1)
        while True:
            if Exited():
                break
            # if not self.faceUpdated():
            if True :
                if self.pp.has_free:
                    frame = self.frameNow()
                    if frame is not None:
                        self.pp.render(
                            frame
                        )
            time.sleep(0.1)
        Exit()

def main():
    cap = cv2.VideoCapture(camera)
    capw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    caph = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CAP_SIZE = (caph,capw,3)
    cap.release()
    # PAPER_FRAME_BUFFER_SIZE = min(MAX_FRAME_BUFFER_SIZE, CAMERA_SIZE)
    # BUFFER_COEFICIENT = math.sqrt(PAPER_FRAME_BUFFER_SIZE / CAMERA_SIZE)
    # BUFFER_GEOGRERAPHY= tuple(map(int,(np.array((capw,caph),dtype=np.int32) * BUFFER_COEFICIENT).astype(np.int32)))
    # DETECT_EXPIRE = 200
    RECOGNIZER_NUM = 5


    with mp.Manager() as manager:
        ExitFlag = manager.Value('i',0)
        settings = {
            'FRAME_SHAPE':CAP_SIZE,
            'FACE_BORDER_COLOR':(255,100,100),
            'FACE_BORDER':2,
            'DETECT_EXPIRE':800,
            'READ_FACES_DELAY':150
        }
        framem = frameManager(CAP_SIZE)
        facem = faceManager(manager,settings)
        cm = cameraManager(manager,settings,framem,ExitFlag)
        dm = displayManager(facem,framem,settings,ExitFlag)
        pp = baseRecognizerPool(manager,settings,facem,ExitFlag,1,num=RECOGNIZER_NUM)
        tm = taskManager(pp,facem,framem,ExitFlag)
        cm.start()
        dm.start()
        tm.start()
        pp.start()
        cm.join()
        dm.join()
        tm.join()
        pp.join()
 