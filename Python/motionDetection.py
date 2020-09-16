import cv2
import time
import pandas
from datetime import datetime

static_back = None # static back to color
motion_list = [None, None]
time_list = []
df = df = pandas.DataFrame(columns = ['Start Motion', 'End Motion'])  # this dataframe will turn into a csv file later

video = cv2.VideoCapture(0)

while True:
    # Start the Video and motion's counter #
    ret, frames = video.read()
    motion = 0
    
    ######################################################################################################
    
    # Color Scales #
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # This peace of code is executed just once at the start
    # where the static back is assign with gray scale color
    if static_back is None: 
        static_back = gray
        continue
    
    differencial_frame = cv2.absdiff(static_back, gray)
    threshold_frame = cv2.threshold(differencial_frame, 30, 255, cv2.THRESH_BINARY)[1]
    threshold_frame = cv2.dilate(threshold_frame, None, iterations = 2)
    
    ######################################################################################################
    
    # Find motions and its contours #
    cnts, _ = cv2.findContours(threshold_frame.copy(),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in cnts:
        if cv2.contourArea(contour) < 10000: continue
        else: 
            motion = 1
            
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 255), 3)
            
    ######################################################################################################
    
    # Append list of motions for each new motion detected
    # and set the start\end time
    motion_list.append(motion)
    motion_list = motion_list[-2:]
    
    if motion_list[-1] == 1 and motion_list[-2] == 0: time_list.append(datetime.now()) # start motion's time
    if motion_list[-1] == 0 and motion_list[-2] == 1: time_list.append(datetime.now()) # end motion's time
        
    ######################################################################################################
        
    # Show the Video in four windows #
    cv2.imshow('GrayScale Frame', gray)
    cv2.imshow('Difference Frame', differencial_frame)
    cv2.imshow('Threshold Frame', threshold_frame)
    cv2.imshow('ColorFull Frame', frames) # detected motions' rectangles are shown here
    
    ######################################################################################################
    
    # Closing the Windows: press 'q' key #
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        if motion == 1: time_list.append(datetime.now()) # catch the last movement
        break
    
# Put list of motions into the dataframe
# and save it as csv file
for i in range(0, len(time_list), 2):
    df = df.append({"Start Motion": time_list[i], "End Motion": time_list[i + 1]}, ignore_index = True)
    
df.to_csv("Time_of_Motions.csv")     

# Turn of cam and Close cv2's windows
video.release()
cv2.destroyAllWindows()