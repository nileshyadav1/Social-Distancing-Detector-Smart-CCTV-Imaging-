
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
from imutils.video import VideoStream
import pandas as pd
import time
import datetime

# File path to the Yolo-colo algorithm folder provided in resource folder
MODEL_PATH = "D:\social-distance-detector\social-distance-detector\yolo-coco"


MIN_CONF = 0.3
NMS_THRESH = 0.3
USE_GPU = False
MIN_DISTANCE = 50

def detect_people(frame, net, ln, personIdx=0):
    (H, W) = frame.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:

        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personIdx and confidence > MIN_CONF:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))


    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)


    if len(idxs) > 0:

        for i in idxs.flatten():

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)


    return results


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="",
    help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
    help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
    help="whether or not output frame should be displayed")
ap.add_argument("-a",'--min-area',type=int,default=500,help="minimum area size")
args = vars(ap.parse_args())


labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join([MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])


print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


if USE_GPU:

    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

first_frame=None
status_list=[None,None]
times=[]
df=pd.DataFrame(columns=["Start","End"])

#File path to face_detect.xml Haar Cascade file provided in resource Folder 
face_cascade=cv2.CascadeClassifier("C:/Users/toor/Desktop/face_detect.xml")


print("[INFO] accessing video stream...")
if args.get("video",None) is None:
    vs=VideoStream(src=0).start()
    time.sleep(2.0)
else:
    vs=cv2.VideoCapture(args["video"])
writer = None


while True:
#	(grabbed, frame) = vs.read()
    frame=vs.read()
    status =0
    frame=frame if args.get("video",None) is None else frame[1]
    text="Unoccupied"


    if frame is None:
        break

    frame = imutils.resize(frame, width=700)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=4)
    print(faces)
    results= detect_people(frame, net, ln,personIdx=LABELS.index("person"))

    if first_frame is None:
        first_frame=gray
        continue
    delta_frame=cv2.absdiff(first_frame,gray)
    thresh_delta=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_delta=cv2.dilate(thresh_delta,None,iterations=2)
    cnts=cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue
        status=1
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        text="Occupied"
    for x,y,w,h in faces:
       frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
      #reized_img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.putText(frame,"Room Status :{}".format(text),(10,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.putText(frame,datetime.datetime.now().strftime("%A %d %B %Y %I : %M : %S%p"),
                (10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255),1)
    status_list.append(status)
    status_list=status_list[-2:]
    if status_list[-1]==1 and status_list[-2]==0:
       times.append(datetime.datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
       times.append(datetime.datetime.now())
    violate = set()

    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")


        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < MIN_DISTANCE:
                    violate.add(i)
                    violate.add(j)


    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        if i in violate:
            color = (0, 0, 255)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    if args["display"] > 0:

        cv2.imshow("Thresh",thresh_delta)
        cv2.imshow("Delta",delta_frame)
        cv2.imshow("OUTPUT Frame", frame)
        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):
            if status==1:
                times.append(datetime.datetime.now())
                break

    if args["output"] != "" and writer is None:

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 15,
            (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)

print(status_list)
print(times)
for j in range(0,len(times),2):
    df=df.append({"Start":times[j],"End":times[j+1]},ignore_index=True)
    df.to_csv("C:/Users/toor/Desktop/Record.csv")



vs.stop() if args.get("video",None) is None else vs.release()
cv2.destroyAllWindows()
