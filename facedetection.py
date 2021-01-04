import cv2
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
xs,ys,ws,hs = 0,0,640,360
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        h = int(w/9*16)
        y = y - int(w/9*3.5)
        
        if(h>=640) :
            y = 0
            h = 640
            x = x + int(w/2) - 180
            w = 360      
        else :
            if (y<0) : 
                y = 0
            if (h+y>640) :
                y = 640-h                 

        frame = frame[y:y+h, x:x+w]  
        frame = cv2.resize(frame, (360,640), interpolation = cv2.INTER_AREA)
        xs,ys,ws,hs = x,y,w,h

    if (len(faces) == 0):
        frame = frame[ys:ys+hs, xs:xs+ws] 
        frame = cv2.resize(frame, (360,640), interpolation = cv2.INTER_AREA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
