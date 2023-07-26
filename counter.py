from ultralytics import YOLO
import cv2
# import cvzone
import numpy as np
import math
from sort import *

cap=cv2.VideoCapture(r'E:\Pr\Car_Counter\cars.mp4')

model=YOLO('yolov8m.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask=cv2.imread(r'E:\Pr\Car_Counter\mask.png')

tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limit=[400,297,673,297]

totalcount=[]

while True:
    success,img=cap.read()
    imgRegion=cv2.bitwise_and(img,mask)
    results=model(imgRegion,stream=True)
    detections=np.empty((0,5))
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2= int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),3)
            conf=math.ceil((box.conf[0]*100))/100  #confidenc3
            cls=int(box.cls[0])  #classnames
            currentClass=classNames[cls]
            
            if currentClass=='car' and conf>0.3:
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),3)
                # cv2.putText(img,f'{currentClass} {conf}',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 0),1)
                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))
                
    resultsTracker=tracker.update(detections)   
    cv2.line(img,(limit[0],limit[1]),(limit[2],limit[3]),(0,0,255),5)

    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        print(result)
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)
        cv2.putText(img,f'{int(id)}',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0, 0),3)

        cx,cy=int((x1+x2)/2) ,int((y1+y2)/2)
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limit[0]<cx<limit[2] and limit[1]-15<cy<limit[3]+15:
            if id not in totalcount:
                totalcount.append(id)
                cv2.line(img,(limit[0],limit[1]),(limit[2],limit[3]),(0,255,0),5)

    cv2.putText(img,f'Count: {len(totalcount)}',(50,100),cv2.FONT_HERSHEY_COMPLEX,4,(0,255,0),4)

    cv2.imshow('Image',img)
    # cv2.imshow('ImageRegion',imgRegion)
    cv2.waitKey(1)