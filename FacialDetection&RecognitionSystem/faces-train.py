import cv2
import os
import numpy as np #
#used For training the classifier
from PIL import Image
import pickle #for Serializing the Ids

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
#path to images 
image_dir = os.path.join(BASE_DIRECTORY, "Dataset")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#using KBPHFaceReconizer
Face_recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		#only png and jpg files will be detected
		if file.endswith("png") or file.endswith("jpg"):  # if image in jpg or png
			path = os.path.join(root, file) #print the file path
			label = os.path.basename(root).replace(" ", "-").lower()
			
			if not label in label_ids: 
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]

			#Covert All the Image to Numeric form 
			pil_image = Image.open(path).convert("L") # grayscale
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")
			
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
			#Append the Data
			for (x,y,w,h) in faces:
				Region_of_interest = image_array[y:y+h, x:x+w]
				x_train.append(Region_of_interest)
				y_labels.append(id_)




with open("pickles/face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

Face_recognizer.train(x_train, np.array(y_labels))
Face_recognizer.save("Trained_Model/face-trainner.yml")