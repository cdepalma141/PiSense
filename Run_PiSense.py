from imutils.video import FPS
import cv2 as cv
import imutils
import numpy as np
import pickle
import time
import requests

#setup for slack

headers = {
        'Content-type': 'applicatiojson',
}
def noticePerson():
    data = '{"attachments": [{"text":"Unidentified person!","color":"danger"}, {"attachment_type":"default"}]}'
    response = requests.post('https://hooks.slack.com/services/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',
            headers=headers, data=data)

def knownPerson(name):
    data = '{{"attachments": [{{"text":"Ya boy {0} is here", "color":"good"}}, {{"attachment_type":"default"}}]}}'.format(name)
    response = requests.post('https://hooks.slack.com/services/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', 
            headers=headers, data=data)

#variables for face detection
protoPath = "deploy.prototxt"
modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
recognizer = pickle.loads(open("recognizer.pickle", "rb").read())
le = pickle.loads(open("le.pickle", "rb").read())

#setup for person detection
cvNet = cv.dnn.readNetFromTensorflow('sorted_inference_graph.pb', 'graph.pbtxt')

cap = cv.VideoCapture(0)
c = 0
a = 0
detect = False
safe = False
b = time.time()
name = "unknown"
fps = FPS().start()
while True:
    names = []
    ret, img = cap.read()
    img = imutils.resize(img, width=400)
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.85:
            detect = True
            name = "unknown"
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), 
                    (0, 0, 255), thickness=2)

            label = "{}: {:.2f}%".format("person", 
                    score * 100)
            y = bottom - 15
            cv.putText(img, label, (int(left + 25), int(y)), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            c = time.time()

            if safe is True:
                e = time.time()
                while time.time() < e + 40:
                    ret, img = cap.read()
                    img = imutils.resize(img, width=400)
                    rows = img.shape[0]
                    cols = img.shape[1]
                    cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
                    cvOut = cvNet.forward()
                    for detection in cvOut[0,0,:,:]:
                        score = float(detection[2])
                        if score > 0.85:
                            detect = True
                            name = "unknown"
                            left = detection[3] * cols
                            top = detection[4] * rows
                            right = detection[5] * cols
                            bottom = detection[6] * rows
                            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), 
                                    (0, 0, 255), thickness=2)

                            label = "{}: {:.2f}%".format("person", 
                                    score * 100)
                            y = bottom - 15
                            cv.putText(img, label, (int(left + 25), int(y)), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    fps.update()
                    cv.imshow('img', img)
                    key = cv.waitKey(1) & 0xFF
                    safe = False

                    if key == ord("q"):
                        break
                    fps.update()
            elif detect is True:
                while time.time() < c + 10:

                    ret, img = cap.read()
                    img = imutils.resize(img, width=400)
                    rows = img.shape[0]
                    cols = img.shape[1]
                    cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
                    cvOut = cvNet.forward()
                    for detection in cvOut[0,0,:,:]:
                        score = float(detection[2])
                        if score > 0.85:
                            left = detection[3] * cols
                            top = detection[4] * rows
                            right = detection[5] * cols
                            bottom = detection[6] * rows
                            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), 
                                    (0, 0, 255), thickness=2)

                            label = "{}: {:.2f}%".format("person", 
                                    score * 100)
                            y = bottom - 15
                            cv.putText(img, label, (int(left + 25), int(y)), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)


                        imageBlob = cv.dnn.blobFromImage(
                                cv.resize(img, (300, 300)), 1.0, (300, 300),
                                (104.0, 177.0, 123.0), swapRB=False, crop=False)

                        detector.setInput(imageBlob)
                        detections = detector.forward()

                    for i in range(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.7:
                            box = detections[0, 0, i, 3:7] * np.array([cols, rows, cols, rows])
                            (startX, startY, endX, endY) = box.astype("int")

                            face = img[startY:endY, startX:endX]
                            (fH, fW) = face.shape[:2]

                            if fW < 20 or fH < 20:
                                    continue

                            faceBlob = cv.dnn.blobFromImage(face, 1.0 / 255,
                                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                            embedder.setInput(faceBlob)
                            vec = embedder.forward()

                            preds = recognizer.predict_proba(vec)[0]
                            j = np.argmax(preds)
                            proba = preds[j]
                            name = le.classes_[j]
                            names.append(name)
                            text = "{}: {:.2f}%".format(name, proba * 100)
                            y = startY - 10 if startY - 10 > 10 else startY + 10
                            cv.rectangle(img, (startX, startY), (endX, endY),
                                    (255, 0, 255), 2)
                            cv.putText(img, text, (startX, y),
                                    cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

                    fps.update()
                    cv.imshow('img', img)
                    key = cv.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break
                    fps.update()
    d = time.time()
    if  "unknown" not in names and names:
        safe = True
        knownPerson(names[0])

    elif d > b+10 and (detect and ("unknown" in names or not names)):
        noticePerson()
        detect = False
        b = time.time()
            
    fps.update()
    cv.imshow('img', img)
    key = cv.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv.destroyAllWindows()
cap.stop()
