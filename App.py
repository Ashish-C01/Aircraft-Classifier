from fastai.learner import load_learner
import cv2
from ultralytics import YOLO
from math import ceil
from argparse import ArgumentParser
from PIL import Image
import numpy as np    
parser=ArgumentParser()
parser.add_argument('data',help='location of the data')
args=parser.parse_args()
learner=load_learner('model/new_aircraft_model.pkl')
yolo_model=YOLO("model/yolov8n.pt")

if args.data.lower().endswith(('.jpg','.jpeg','.tiff','.gif','.png','.bmp')):
    result=yolo_model.predict(args.data,save=False,classes=[4],verbose=False)
    if len(result[0].boxes.conf):
        for i in range(len(result[0].boxes.conf)):
            x1=ceil(result[0].boxes.xyxy[i][0])
            y1=ceil(result[0].boxes.xyxy[i][1])
            x2=ceil(result[0].boxes.xyxy[i][2])
            y2=ceil(result[0].boxes.xyxy[i][3])
            image=Image.open(args.data)
            image_array=np.asarray(image)
            image=cv2.resize(image_array,(400,400),interpolation=cv2.INTER_LINEAR)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            prediction=learner.predict(image)
            frame=cv2.rectangle(result[0].orig_img,(x1,y1),(x2,y2),(0,255,0),2)
            frame=cv2.putText(frame,prediction[0],(x1+2,y1+2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            cv2.imshow("frame",frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    cap=cv2.VideoCapture(args.data)
    while True:
        ret,frame=cap.read()
        result=yolo_model.predict(frame,save=False,classes=[4],verbose=False)
        if len(result[0].boxes.conf):
            for i in range(len(result[0].boxes.conf)):
                x1=ceil(result[0].boxes.xyxy[i][0])
                y1=ceil(result[0].boxes.xyxy[i][1])
                x2=ceil(result[0].boxes.xyxy[i][2])
                y2=ceil(result[0].boxes.xyxy[i][3])
                image=cv2.resize(frame[y1:y2,x1:x2],(400,400),interpolation=cv2.INTER_LINEAR)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                prediction=learner.predict(image)
                frame=cv2.rectangle(result[0].orig_img,(x1,y1),(x2,y2),(0,255,0),2)
                frame=cv2.putText(frame,prediction[0],(x1+2,y1+2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                cv2.imshow("frame",frame)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()