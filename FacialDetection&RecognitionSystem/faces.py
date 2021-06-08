#import Needed Libaries
import numpy as np
import cv2
import pickle

    

Face_Classifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
#read From The Trained Data
recognizer.read("./Trained_Model/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}


Image = cv2.imread('SRK4.jpg')
#Setting the Resolution so that our output screen is caped to 720p
Image = cv2.resize(Image, (1280, 1000))

while(True):
    # Capture Image-by-Image
   
    gray  = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	#face Detection
    faces = Face_Classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
    	#print(x,y,w,h)
    	Region_OF_Interest_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
    	Region_OF_Interest_color = Image[y:y+h, x:x+w]

    	# Predict using trained Model
    	id_, conf = recognizer.predict(Region_OF_Interest_gray)
    	if conf>=4 and conf <= 85:
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = labels[id_]
    		color = (255, 255, 255)
    		stroke = 2
    		cv2.putText(Image, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		#rename the Result output image
    	img_item = "Detection.png"
    	cv2.imwrite(img_item, Region_OF_Interest_color)

    	color = (255, 0, 0) #BGR 0-255 
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(Image, (x, y), (end_cord_x, end_cord_y), color, stroke)
    # Display the resulting Image
    cv2.imshow('Image',Image)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()
