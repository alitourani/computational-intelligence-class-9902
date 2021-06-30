import cv2
import dlib
from scipy.spatial import distance as d

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
face_landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# calculate aspect ratio
def aspect_ratio_cal(eye):
	AR = (d.euclidean(eye[1], eye[5])+  d.euclidean(eye[2], eye[4])) / (2.0 * d.euclidean(eye[0], eye[3]))
	return AR

while cap.isOpened():
    # capture frames
    ret, frame = cap.read()
    gry = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gry)

    for face in faces:
        lmarks = face_landmarks(gry, face)
        # array for the right eye
        R = []
        # array for the left eye
        L = []
        # number of next node
        next = 0
        
           
        for n in range (36, 42): 
          # filling the left eye array
        	L.append((lmarks.part(n).x ,lmarks.part(n).y))
          # find the number of the next node
        	next = n + 1
          # for the last node
        	if n == 41:
        	  next = 36
          # draw a line between node and next node
        	cv2.line(frame, (lmarks.part(n).x, lmarks.part(n).y), (lmarks.part(next).x ,lmarks.part(next).y), (255, 0, 0), 1)
         
        for n in range (42, 48): 
          # filling the right eye array
        	R.append((lmarks.part(n).x ,lmarks.part(n).y))
          # find the number of the next node
        	next = n + 1
          # for the last node
        	if n == 47:
        	  next = 42
          # draw a line between node and next node
        	cv2.line(frame, (lmarks.part(n).x, lmarks.part(n).y), (lmarks.part(next).x ,lmarks.part(next).y), (255, 0, 0), 1)

        # if the eyes are closed (the average of aspect ratios are less than 0.20)
        if round((aspect_ratio_cal(L) + aspect_ratio_cal(R))/2, 2) < 0.20:
          # detect the drawsiness and print on the screen
          cv2.putText(frame,"Drowsiness Detected", (175, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3) 
          
       
    cv2.imshow("", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()