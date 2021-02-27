import cv2

# load our image
image_file = cv2.imread('car Image.png')

# video = cv2.VideoCapture('Tesla Autopilot Dashcam Compilation 2018 Version.mp4')
video = cv2.VideoCapture('video-1.mp4')
# video = cv2.VideoCapture("Pedestrians Compilation.mp4")

# our pre-trained car_pedestrians classifier
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = "haarcase_fullbody.xml"

# create car classifiter
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedistrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


while True:
    # read the current frame
    read_successful, frame = video.read()

    # safe coding
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedistrian_tracker.detectMultiScale(grayscaled_frame)
    
    # Draw the rectangles around the cars 
    for(x,y,w,h) in cars:
        cv2.rectangle(frame, (x+1,y+2), (x+w, y+h), (255,0,0),2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)

    # Draw the rectangles around the pedestrians
    for(x,y,w,h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255),2)

    # Display the images with the face spotted image
    cv2.imshow('Car Detector', frame)
    key = cv2.waitKey(1)

    # stop if the q is pressed
    if key ==81 or key ==113:
        break
    
# Release the video capture objects
video.release()

"""  
# conver image into grayscale image
# black_n_white = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(black_n_white) 

# print(cars)
for (x,y,w,h) in cars:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)


# display the image with the faces spotted
cv2.imshow('Yr Arjun Car Detector', frame)

# dont't autoclose wait here and not exit without presing
# any key
cv2.waitKey()
"""

print("code completed...")