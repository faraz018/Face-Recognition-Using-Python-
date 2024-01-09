import cv2
import os
cascPathface = os.path.dirname(
    cv2.__file__)+"/data/haarcascade_frontalface_alt2.xml"
cascPatheyes =os.path.dirname(
    cv2.__file__)+"/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)

video_capture= cv2.VideoCapture(0)
while True:
    ret, frame=video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(60,60),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        faceROI = frame[y:y+h,x:x+w]
        eyes = eyeCascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_centre=(x + x2 + w2 // 2,y + y2 + h2 // 2)
            radius= int(round(w2 + h2)*0.25)
            frame=cv2.circle(frame,eye_centre,radius,(255,0,0),4)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()