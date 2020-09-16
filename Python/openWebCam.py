import cv2

# Create video object
video = cv2.VideoCapture(0)

while True:
    # Capture each frame of the video
    ret, frame = video.read()
    
    # Create the color's channels
    #grayscale_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    colorfull_channel = frame
    
    # Open the cam
    #cv2.imshow('WebCam', grayscale_channel) # grayscale video (256 levels)
    cv2.imshow('Webcam', colorfull_channel) # colored video ([256, 256, 256] levels)
    
    # Press 'q' (quit '-') key to close the cam
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Turn off the cam and close all cv2's windows
video.release()
cv2.destroyAllWindows()