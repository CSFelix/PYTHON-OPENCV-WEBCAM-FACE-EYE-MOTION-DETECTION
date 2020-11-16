import cv2
import os

# Load and create faceCascade Classifier
facePath = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(facePath)

# Create 'video' and 'out' objects ('out' is the video that will be saved)
video = cv2.VideoCapture(0)
out = cv2.VideoWriter('faceDetection.mp4', -1, 20.0, (640,480))

while True:
    try:
        # start the video with RGB colors
        ret, frames = video.read()
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        
        # Face Detection
        faces = faceCascade.detectMultiScale(gray)
        for (x, y, w, h) in faces: cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
        # Save and Show the Video
        out.write(frames)
        cv2.imshow('WebCam', frames)
    
    	# press 'q' to break
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    except Exception as e:
        print(str(e))
        break

# turn down the video and finish CV2 execution
video.release()
out.release()
cv2.destroyAllWindows()
