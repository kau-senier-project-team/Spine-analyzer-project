#https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
#https://gitlab.uni-koblenz.de/robbie/homer_gesture_recognition/blob/47afcbe461f38116507c51bcc54610f2b510a638/OpenPose/OpenPoseVideo.py
"""
COCO Output Format
Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4,
Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8,
Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12,
LAnkle – 13, Right Eye – 14, Left Eye – 15, Right Ear – 16,
Left Ear – 17, Background – 18

"""

import cv2
import time
import numpy as np
import json
import os

def createFile(name) :
    data = {}
    data['points'] = []
    
    #파일에 기본정보 입력
    with open(name+'.json', 'x') as outfile:
        json.dump(data, outfile, indent='\t')

def savePoints(dic):
    timestr = time.strftime("%Y-%m-%d")
    if not os.path.exists(timestr + '.json') : # 파일이 없으면 새로 만든다.
        createFile(timestr)
    
    with open(timestr + '.json') as data_file: # 파일 읽어서 데이터 저장
        data = json.load(data_file)

    data['points'].append({
        'Nose': dic['Nose'],
        'Neck': dic['Neck'],
        'Right Shoulder': dic['Right Shoulder'],
        'Right Elbow': dic['Right Elbow'],
        'Right Wrist': dic['Right Wrist'],
        'Left Shoulder': dic['Left Shoulder'],
        'Left Elbow': dic['Left Elbow'],
        'Left Wrist': dic['Left Wrist'],
        'Right Hip': dic['Right Hip'],
        'Right Knee': dic['Right Knee'],
        'Right Ankle': dic['Right Ankle'],
        'Left Hip': dic['Left Hip'],
        'Left Knee': dic['Left Knee'],
        'Left Ankle': dic['Left Ankle'],
        'Right Eye': dic['Right Eye'],
        'Left Eye': dic['Left Eye'],
        'Right Ear': dic['Right Ear'],
        'Left Ear': dic['Left Ear'],
    })
    
    with open(timestr+'.json', 'w') as outfile: # 데이터 저장
        json.dump(data, outfile, indent='\t')

    return

def detectPose():
    # Specify the paths
    MODE = "VIDEO"

    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    position=["Nose", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist", "Left Shoulder", "Left Elbow", "Left Wrist", 
              "Right Hip", "Right Knee", "Right Ankle", "Left Hip", "Left Knee", "Left Ankle", "Right Eye", "Left Eye", 
              "Right Ear", "Left Ear", "Background"]

    # Read video, webcam 이미지는 다른 코드가 필요하다. 아래의 while문 자체가 필요가 없다.
    if MODE is "VIDEO" :
        input_source = "YS.mp4" #video
        cap = cv2.VideoCapture(input_source)

    elif MODE is "WEBCAM" :
        cap = cv2.VideoCapture(0) #webcam .. 맞나?

    # Read the network into Memory
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Specify the input image dimensions
    inWidth = 368
    inHeight = 368
    threshold = 0.1

    hasFrame, frame = cap.read()
    cv2.imwrite('pose_start.jpg',frame)

    vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    while cv2.waitKey(1) < 0:
        t = time.time()
        hasFrame, frame = cap.read()
        copyFrame = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                  (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]
        #json을 만들기 위한 dic
        points = {}

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold : 
                cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                #x와 y좌표 저장
                points[position[i]]=[int(x),int(y)]
            else :
                #파이썬에서 None은 NULL과 같은 값이다. 예외처리하기 쉬우라고 넣었다.
                points[position[i]]="None"

        savePoints(points)

        cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.imshow('Output-Skeleton', frame)

        vid_writer.write(frame)

    cv2.imwrite('pose_end.jpg',copyFrame)
    
    vid_writer.release()
    print("done")

detectPose()
