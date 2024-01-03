import cv2
import dlib
import os
import multiprocessing as mp
import threading as th
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
import time
import numpy as np
import copy
# global NEW_FRAME_SENT
CALCULATE_FACES = 0
GET_NEW_FRAME = 1
NEW_FRAME_SENT=2
EXIT = 4
START_SEND_FACES = 6
GET_NEW_FACE = 3
NEW_FACE_SENT = 5
statDict = {
0:'CALCULATE_FACES'
,1:'GET_NEW_FRAME'
,2:'NEW_FRAME_SENT'
,4:'EXIT'
,6:'START_SEND_FACES'
,3:'GET_NEW_FACE'
,5:'NEW_FACE_SENT'

}


detect_way = 0
camera = 0
NoneType = type(None)
os.environ.setdefault('QT_QPA_PLATFORM','xcb')
# Load the pre-trained face recognition model
if detect_way :
    detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
else:
    detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the webcam or video stream
cap = cv2.VideoCapture(camera)  # Use 0 for default webcam, or provide the video file path
capw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
caph = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def find_face(SHstat:Synchronized, SHframe:SynchronizedArray, SHret:SynchronizedArray):
    global frame, ret, stat, NEW_FRAME_SENT
    frame = None
    ret = []
    stat = None
    while True:
        # time.sleep(1)
        with SHstat.get_lock():
            stat = copy.deepcopy(SHstat.value)
        match stat:
            
            case  4 : # EXIT
                break

            case 2 : #NEW_FRAME_SENT
                with SHframe.get_lock():
                    sh_frame = np.frombuffer(SHframe.get_obj(), dtype=np.uint8).reshape((caph, capw,3))
                    frame = copy.deepcopy(sh_frame)
                with SHstat.get_lock():
                    SHstat.value = CALCULATE_FACES
            case 3 : # GET_NEW_FACE
                if len(ret):
                    with SHret.get_lock():
                        sh_ret = np.frombuffer(SHret.get_obj(), dtype=np.int16)
                        sh_ret[:] = ret.pop()
                    with SHstat.get_lock():
                        SHstat.value = NEW_FACE_SENT
                else:
                    with SHstat.get_lock():
                        SHstat.value = GET_NEW_FRAME
            case 0 : # CALCULATE_FACES
                if type(frame)!=NoneType:
                    gray = cv2.cvtColor(frame,  cv2.COLOR_RGB2GRAY)        
                    faces = detector(gray)
                    for faceObject in faces :
                        if detect_way:
                            face = faceObject.rect
                        else:
                            face = faceObject
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        print(f'found {(x,y,w,h)}')
                        ret.append(
                            (x,y,w,h)
                        )
                    with SHstat.get_lock():
                        SHstat.value = START_SEND_FACES
                else:
                    with SHstat.get_lock():
                        SHstat.value = GET_NEW_FRAME
                    time.sleep(0.1)



def display (SHstat:Synchronized, SHframe:SynchronizedArray, SHret:SynchronizedArray):

    rets = []
    lastStat = 0
    frame = None
    Exit = False
    def setStat(newStat):
        with SHstat.get_lock():SHstat.value = newStat
    def display_render():
        nonlocal Exit, frame, rets
        while True:
            # Read a frame from the video stream
            if Exit :
                break
            ret, frame = cap.read()

            # Iterate over detected faces
            for face in rets:
                x, y, w, h = face[0], face[1], face[2], face[3]
                # print(f'printed {(x,y,w,h)}')
                startP = (x, y)
                endP = (x + w, y + h)
                # print(f'printed{startP}|{endP}')
                cv2.rectangle(frame, startP, endP, (0, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        Exit = True
    def display_helper(SHstat, SHframe, SHret):
        nonlocal Exit, frame , lastStat, rets
        while True:
            if Exit:
                break
            with SHstat.get_lock():
                if lastStat != SHstat.value :
                    lastStat = copy.deepcopy(SHstat.value)
                    print(statDict[lastStat])
            if lastStat == EXIT:
                break
            elif lastStat == GET_NEW_FRAME:
                if type(frame) != NoneType:
                    with SHframe.get_lock():
                        sh_frame = np.frombuffer(SHframe.get_obj(), dtype=np.uint8)
                        sh_frame[:] = frame.reshape(3*caph*capw)
                        setStat(NEW_FRAME_SENT)
            elif SHstat.value == START_SEND_FACES:
                rets = []
                setStat(GET_NEW_FACE)
            elif SHstat.value == NEW_FACE_SENT :
                with SHret.get_lock():
                    rets.append(
                        [*map(int,copy.deepcopy((np.frombuffer(SHret.get_obj(),dtype=np.int16))))]
                    )
                setStat(GET_NEW_FACE)
        Exit = True

    render_thread = th.Thread(target=display_render)
    helper_thread = th.Thread(target=display_helper,args=(SHstat, SHframe, SHret))
    render_thread.start()
    helper_thread.start()
    render_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    helper_thread.join()

if __name__ == "__main__":
    shared_values = (
            mp.Value('i', 0),
            mp.Array('B',capw*caph*3),
            mp.Array('B',8)
    )
    p1 = mp.Process(target=display,args=shared_values)
    p2 = mp.Process(target=find_face,args=shared_values)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
