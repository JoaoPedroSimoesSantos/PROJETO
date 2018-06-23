import cv2 as cv
import numpy as np
i = 0

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv.VideoCapture('images/video_salvamento_aquatico.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:
    i += 1
    print i
    # Display the resulting frame
    if 4510 <= i < 4535:
      cv.imshow('Frame',cv.resize(frame,None,fx = 0.5,fy = 0.5))
      cv.imwrite("Frame" + str(i) + ".jpg",frame)
 
    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q') or i == 4535:
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv.destroyAllWindows()