import cv2  # opencv library
from picamera2 import Picamera2, Preview
from libcamera import Transform
import time
import threading
from collections import deque
import numpy as np

from adafruit_servokit import ServoKit

import mediapipe as mp
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime





class GDModel(nn.Module):
    def __init__(self, inDim=42, numClass=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inDim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, numClass)
        )
    def forward(self, x): 
        return self.net(x)


#video writer, face detection, gesture detection and servo controller are all workers to keep them processing seperatly from the main loop to allow for smooth video feed
class VideoWriter:
    def __init__(self, max_queue=120):
        self.out = None
        self.q = deque(maxlen=max_queue) 
        self.running = False
        self.thread = None

    #gets frame and adds to queue
    def write(self, frame):
        if not self.running:
            return
        self.q.append(frame.copy()) #so that we dont write frame w overlays

    #starts video
    def start(self, filename, fps, format="XVID"):
        frameSize = (640,480)

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        form = cv2.VideoWriter_fourcc(*format)
        self.out = cv2.VideoWriter(filename, form, fps, frameSize, True)
        self.running = True
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    #stops video
    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

        #processes remaning frames
        while self.q:
            f = self.q.popleft()
            self.out.write(f)

        self.out.release()
        self.out = None

    def worker(self):
        while self.running:
            frame = None
            if self.q:
                frame = self.q.popleft()
                self.out.write(frame)
            else:
                time.sleep(0.001)  
                
class FaceDetectionWorker:
    def __init__(self):
        #face model
        self.model = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        #input frame
        self.input = None
        #face output
        self.output = None
        self.prevTs = time.time()

    #creates thread that runs run
    def startThr(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self
    
    #passes latest frame into worker
    def submit(self, inFrame):
        self.input=inFrame

    def readLatest(self):
        return None if self.output is None else self.output

    def run(self):
        while True:
            #limit to process at 30hz
            if time.time() - self.prevTs < 0.033:
                time.sleep(0.005)
                continue
            self.prevTs = time.time()
            
            if self.input is None:
                time.sleep(0.003)
                continue
            frame = self.input
            box = None
            res = self.model.process(frame)
            if res.detections:
                inH, inW = frame.shape[:2]
                detect = res.detections[0]
                bb = detect.location_data.relative_bounding_box
                x = int(bb.xmin * inW)
                y = int(bb.ymin * inH)
                w = int(bb.width * inW)
                h = int(bb.height * inH)
                box= x,y,w,h

            self.output = {"ts": time.time(), "face": box}

class HandLandmarksWorker:
    def __init__(self):
        #frame input
        self.input = None
        #landmarks output
        self.output = None
        self.thread = None
        self.prevTs = time.time()

        #mp hands has two parts, detector and tracker
        #works by first using the detector to locate the hand (CNN, costly)
        #then the tracker predicts the landmarks and tracks the hand for the next frames (cheap)
        self.hands = mp.solutions.hands.Hands(
            #setting to false, detector only runs to find/refind the hand, rest is handled by tracker
            #much faster than running detector every frame
            static_image_mode=False,
            max_num_hands=1,

            #these tested parameters were good enough and ran fast on the pi
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        #Gesture Classifier
        savedModel = torch.load("mlpGestures.pt", map_location="cpu", weights_only=False)
        self.classes = savedModel["classes"]
        self.model = GDModel()
        self.model.load_state_dict(savedModel["model"])
        self.model.eval() 

        #scaler stats saved during training
        self.scaler_mean = savedModel["scalerMean"]
        self.scaler_scale = savedModel["scalerScale"]

    # def to create thread that runs run
    def startThr(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    # passes latest frame into worker
    def submit(self, inFrame):
        self.input = inFrame

    def readLatest(self):
        return None if self.output is None else self.output

    @staticmethod
    def normLM(xyLM, eps=1e-6):
        wrist = xyLM[0]
        relativeLMs = xyLM - wrist
        palm = float(np.linalg.norm(xyLM[9] - wrist) + eps)
        norm = relativeLMs / palm
        return norm

    def run(self):
        while True:
            #limit to process at 30hz
            if time.time() - self.prevTs < 0.033:
                time.sleep(0.005)
                continue

            self.prevTs = time.time()
            # get latest frame
            if self.input is None:
                time.sleep(0.003)
                continue
            frame = self.input

            res = self.hands.process(frame)

            handData = None
            # if hand is detected
            if res.multi_hand_landmarks:
                imgH, imgW = frame.shape[:2]
                lm = res.multi_hand_landmarks[0]
                points = np.array([[p.x * imgW, p.y * imgH] for p in lm.landmark],
                                dtype=np.float32)  # (21,2)
                xMin, yMin = points.min(axis=0)
                xMax, yMax = points.max(axis=0)
                bbox = np.array([xMin, yMin, xMax - xMin, yMax - yMin], dtype=np.float32)

                xyFeats = self.normLM(points)

                #l or r hand
                handed = res.multi_handedness[0].classification[0].label
                #mirror so that classifier see same orientation (was trained on r hand)
                if handed == "Left":
                    xyFeats = xyFeats.copy()
                    xyFeats[:, 0] *= -1.0

                ###
                #gesture classifier
                #organize data so that it match what the model expects
                feats = np.concatenate([xyFeats[:, 0], xyFeats[:, 1]], axis=0).astype(np.float32) 
                #standardize
                feats = (feats - self.scaler_mean) / self.scaler_scale
                #to tensor [1, 42]
                input = torch.from_numpy(feats).unsqueeze(0).to(torch.float32) 

                with torch.no_grad():
                    pred = self.model(input)
                    probs  = F.softmax(pred, dim=1)[0]
                    predIdx = int(torch.argmax(probs).item())
                    predProb = float(probs[predIdx].item())
                    predClass = self.classes[predIdx]

                handData = ({
                    "landmarkXY": points,
                    "bbox": bbox,
                    "predClass": predClass,
                    "predIdx": predIdx,
                    "predProb": predProb
                })

            self.output = {"ts": time.time(), "hand": handData}


#tilt +1 is down, -1 is up
#pan +1 is right, -1 is left

class ServoWorker:
    def __init__(self):
        
        kit = ServoKit(channels=16)
        self.pan  = kit.continuous_servo[0]
        self.tilt = kit.continuous_servo[3]
        self.pan.set_pulse_width_range(min_pulse=1050, max_pulse=2050)
        self.tilt.set_pulse_width_range(min_pulse=1050, max_pulse=2050)
        
        #input data
        self.faceData = None
        self.handData = None

        #tracking status
        self.track = False
        self.lastToggled = 0
        self.lastFaceTs = 0
        self.prevTs = time.time()

    #def to create thread that runs run
    def startThr(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self
    
    #passes latest face/hand into worker
    def submit(self, inFace, inHand):
        self.faceData =inFace
        self.handData = inHand

    def readLatest(self):
        return self.track

    def run(self):
        while True:
            
            #limit to process at 30hz
            if time.time() - self.prevTs < 0.033:
                time.sleep(0.005)
                continue
            self.prevTs = time.time()


            if self.faceData is None and self.handData is None:
                time.sleep(0.003)
                continue
            dataFace = self.faceData
            dataHand = self.handData

            #toggle tracking
            if dataHand and dataHand["hand"] and dataHand["hand"]["predClass"] == "openPalm":
                currTime = time.time()
                if currTime - self.lastToggled > 1.5:
                    self.track = not self.track
                    self.lastToggled = currTime

            if dataFace and dataFace["face"] and len(dataFace["face"]) > 0 and dataFace["ts"] != self.lastFaceTs and self.track:
                self.lastFaceTs = dataFace["ts"]
                x, y, w, h = dataFace["face"]
                errorX = x + (w / 2) - 224/2
                errorY = y + (h / 2) - 168/2

                if abs(errorX) >= 40:
                    # need to move right
                    if errorX > 0:
                        self.pan.throttle = 0.5
                        time.sleep(0.01)
                        self.pan.throttle = 0
                    else:
                        self.pan.throttle = -0.3
                        time.sleep(0.01)
                        self.pan.throttle = 0
                else:
                    self.pan.throttle = 0.0

                if abs(errorY) >= 40:
                    # need to move down
                    if errorY > 0:
                        self.tilt.throttle = .6
                        time.sleep(0.013)
                        self.tilt.throttle = 0
                    else:
                        self.tilt.throttle = -.7
                        time.sleep(0.01)
                        self.tilt.throttle = 0
                else:
                    self.tilt.throttle = 0.0

            else:
                self.tilt.throttle = 0.0
                self.pan.throttle = 0.0

            
###
#Main
###
# get video data
cam = Picamera2()

videoConfig = cam.create_video_configuration(
    #frame for displaying video feed
    main={"size": (640, 480), "format": "RGB888"},
    #frame passed into the workers, lower res 
    # lores={"size": (224 , 168), "format": "YUV420"},
    lores={"size": (224 , 168), "format": "YUV420"},
    # cheaper than cv2.flip
    transform=Transform(vflip=1)
)
cam.configure(videoConfig)

#caps at 60 fps
cam.set_controls({"FrameDurationLimits": (16666, 16666)})
cam.start()

#initalize workers
handWorker = HandLandmarksWorker().startThr()
faceWorker = FaceDetectionWorker().startThr()
servoWorker = ServoWorker().startThr()
vidWriter = VideoWriter(max_queue=180)


frameCount = 0
prevTime = 0
fps = 0
lastFaceTs = 0


fistlastToggled = time.time()
recording = False
lastMarking = time.time()
recMsgUntil = 0
markMsgUntil = 0

#Main Loop
while True:
    
    #reads frame from camera
    req = cam.capture_request()                
    frameMain = req.make_array("main")     
    frameWorker = req.make_array("lores")      
    req.release()                               

    #yuv->rgb
    frameWorker = cv2.cvtColor(frameWorker, cv2.COLOR_YUV2RGB_I420)

    #pass worker frame into hand tracker/face detector
    handWorker.submit(frameWorker)
    faceWorker.submit(frameWorker)

    dataFace = faceWorker.readLatest()
    dataHand = handWorker.readLatest()

    servoWorker.submit(dataFace, dataHand)

    if dataHand and dataHand["hand"]:
        currTime = time.time()
        if dataHand["hand"]["predClass"] == "fist":
            if currTime - fistlastToggled > 1.5:
                if not recording:
                    recStart = currTime
                    timeNow = datetime.now().strftime("%Y%m%d_%H%M%S")
                    vidWriter.start(filename=f"recordings/rec_{timeNow}.avi", fps=30, format="MJPG")

                else:
                    recLen = currTime - recStart
                    recMin, recSec = divmod(round(recLen), 60)
                    recMsgUntil = currTime + 3
                    vidWriter.stop()

                recording = not recording
                fistlastToggled = currTime
        if dataHand["hand"]["predClass"] == "wShape":
            if recording:
                if currTime - lastMarking > 1.5:
                    lastMarking = currTime
                    markTime = currTime - recStart
                    markMin, markSec = divmod(round(markTime), 60)
                    markMsgUntil = currTime + 3

    
    # scaleX = frameMain.shape[1]/frameWorker.shape[1]
    # scaleY = frameMain.shape[0]/frameWorker.shape[0]
    
    # #draws face bbox
    # if dataFace and dataFace["face"]:
    #     x, y, w, h =  dataFace["face"]
    #     cv2.rectangle(frameMain, (int(round(scaleX*x)), int(round(scaleY*y))), (int(round(scaleX*(x+w))), int(round(scaleY*(y+h)))), (255, 0, 0), 2)

    # data = handWorker.readLatest()
    # if data and data["hand"]:
    #     hand = data["hand"]
    #     x, y, w, h = hand["bbox"].astype(int)

    #     #draws hand bbox
    #     cv2.rectangle(frameMain, (int(round(scaleX*x)), int(round(scaleY*y))), (int(round(scaleX*(x+w))), int(round(scaleY*(y+h)))), (0, 255, 255), 1)

    #     #label gesture
    #     if hand["predClass"]:
    #         cv2.putText(frameMain, hand["predClass"], (int(round(scaleX*x)), max(0, int(round(scaleY*y-5)))),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) 
    if time.time() <= markMsgUntil:
        cv2.putText(
            frameMain, f"Added Marking at: {markMin} min {markSec} sec", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )  

    if recording:
        vidWriter.write(frameMain)

    frameCount += 1
    currTime = time.time()

    if prevTime == 0:
        fps=0
        prevTime = currTime
    elif currTime-prevTime >= 1.0:
        fps = frameCount/(currTime-prevTime)
        prevTime = currTime
        frameCount = 0
    cv2.putText(frameMain, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 1)
    fistToggleText = "on" if servoWorker.readLatest() else "off"
    recordingToggleText = "yes" if recording else "no"
    cv2.putText(
        frameMain, f"Follow: {fistToggleText}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
    )  

    cv2.putText(
        frameMain, f"Recording: {recordingToggleText}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
    )  

    if time.time() <= recMsgUntil:
        cv2.putText(
            frameMain, f"Recording Finished: {recMin} min {recSec} sec", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
        )  
    

    cv2.imshow('Video Feed', frameMain)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
















