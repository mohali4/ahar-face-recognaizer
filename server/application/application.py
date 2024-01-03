import cv2
import dlib
import os
import time
import numpy as np
import math
import copy
import multiprocessing as mp
import threading
from .mprocess import (
    PPool
)
from .recognizers import (
    recognizer,
    recognizersPool
)
from .managers import (
    faceManager,
    frameManager
)

# camera = 'http://192.168.149.216:8080/video'
# camera = 0
camera = 'http://192.168.14.238:8080/video'
NoneType = type(None)
os.environ.setdefault('QT_QPA_PLATFORM','xcb')
# MAX_FRAME_BUFFER_SIZE = 480*640

class display_manager():
    def __init__(self):
        ...
    def init(self, faceM:faceManager, frameM:frameManager, ExitFlag):
        self.faceM = faceM
        self.frameM = frameM
        self.exited = ExitFlag
        self._lastframe = None
    def read_faces(self):
        faces = self.faceM.get()
        if type(faces) == NoneType:
            return []
        for face in faces:
            yield face[1:]
    def last_frame(self):
        return self._lastframe
    def loop(self):
        def Exited():
            return self.exited.get() == 1
        def Exit():
            cap.release()
            cv2.destroyAllWindows()
            self.exited.set(1)
    
        while True:
            if Exited():
                break
            inpt = time.time()
            ret, frame = cap.read()
            # print('capture time: ',time.time()-inpt)
            inpt = time.time()
            self._lastframe = frame
            # print('refrenc buffer: ',time.time()-inpt)
            # Iterate over detected faces
            inpt = time.time()
            for face in self.read_faces():
                x1, y1, x2, y2 = face[0], face[1], face[2], face[3]
                cv2.rectangle(frame, (x1,y1), (x2, y2), (0, 255, 0), 2)
            # print('write faces: ',time.time()-inpt)
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        Exit()


class task_manager():
    def __init__ (self, PP:recognizersPool, frameM:frameManager, faceM:faceManager, displayM:display_manager, ExitFlag):
        self.pp = PP
        self.frameM  = frameM
        self.faceM = faceM
        self.displayM = displayM
        self.exited = ExitFlag
        
    def frameNow(self):
        return self.displayM.last_frame()
    def faceUpdated(self):
        return self.frameM.status > self.faceM.status
    def setSHFrame(self,frame):
        self.frameM.set(frame)
    def loop(self):
        def Exited():
            return self.exited.get() == 1
        def Exit():
            self.exited.set(1)
        while True:
            if Exited():
                break
            if not self.faceUpdated():
                if self.pp.has_free:
                    frame = self.frameNow()
                    if frame is not None:
                        self.setSHFrame(
                            frame
                        )
                        self.pp.render()
            time.sleep(0.1)
        Exit()

def displayProcess(dm:display_manager,tm:task_manager):
    dmt = threading.Thread(
        target=dm.loop
    )
    tmt = threading.Thread(
        target=tm.loop
    )
    dmt.start()
    tmt.start()
    dmt.join()
    tmt.join()

def main():
    cap = cv2.VideoCapture(camera) 
    capw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    caph = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    CAMERA_SIZE = capw*caph
    CAMERA_GEOGRAPHY = (caph, capw)
    # PAPER_FRAME_BUFFER_SIZE = min(MAX_FRAME_BUFFER_SIZE, CAMERA_SIZE)
    # BUFFER_COEFICIENT = math.sqrt(PAPER_FRAME_BUFFER_SIZE / CAMERA_SIZE)
    # BUFFER_GEOGRERAPHY= tuple(map(int,(np.array((capw,caph),dtype=np.int32) * BUFFER_COEFICIENT).astype(np.int32)))
    BUFFER_GEOGRERAPHY = CAMERA_GEOGRAPHY
    DETECT_EXPIRE = 20
    RECOGNIZER_NUM=3
    REAL_FRAME_BUFFER_SIZE = BUFFER_GEOGRERAPHY[0] * BUFFER_GEOGRERAPHY[1]


    with mp.Manager() as manager:
        ExitFlag = manager.Value('i',0)
        facem = faceManager(manager,DETECT_EXPIRE)
        framem = frameManager(BUFFER_GEOGRERAPHY)
        dm = display_manager(facem,framem,ExitFlag)
        pp = recognizersPool(manager,framem,facem,ExitFlag,1,num=RECOGNIZER_NUM)
        tm = task_manager(pp,framem,facem,dm,ExitFlag)
        p1 = mp.Process(
            target=displayProcess,
            args=(dm,tm)
        )
        p1.start()
        pp.start()
        p1.join()
        pp.join()
        