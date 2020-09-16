import cv2
import os

# Load and create faceCascade Classifier
eyePath = os.path.dirname(cv2.__file__)+"/data/haarcascade_eye.xml"
eyeCascade = cv2.CascadeClassifier(eyePath)


video = cv2.VideoCapture(0)
out = cv2.VideoWriter('eyeDetection.mp4', -1, 20.0, (640,480))

while True:
    try:
        ret, frames = video.read()
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        
        # Eye Detection
        eyes = eyeCascade.detectMultiScale(gray)
        for (x, y, w, h) in eyes: cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 165, 255), 2)
        
        # Save and Show the Video
        out.write(frames)
        cv2.imshow('WebCam', frames)
    
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    except Exception as e:
        print(str(e))
        break

video.release()
out.release()
cv2.destroyAllWindows()