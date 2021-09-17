

#1. Read and show video streams, capture image
#2. Detect faces and show bounding images(haarcascade)
#3. Flatten the largest face image(gray scale) and save in a numpy array
#4. Repeat the above for multiple people to generate training data


import cv2
import numpy as np


#Init Camera
cap=cv2.VideoCapture(0)

#Face Detection
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0

face_data=[]
dataset_path='./data/'

file_name=input("Enter the name of the person : ")

while True:
	ret, img=cap.read()

	if ret==False:
		continue

	gray_frame=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(img,1.3,5)	#object scaling factor no of neighbour
	faces=sorted(faces, key=lambda f:f[2]*f[3])

	#Pick the last face(because it is the largest face acc to area     area=f[2]*f[3])
	for face in faces:
		x,y,w,h=face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)

		#extract (crop out the required face ): Region of interest
		offset=10
		face_section=img[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section=cv2.resize(face_section, (100,100))

		skip+=1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))

		cv2.imshow("priya", img)
		cv2.imshow("face section", face_section)

		# #store every 10th face
		# if(skip%10==0):
		# 	#store the 10th face later on
		# 	pass
	

	key_pressed = cv2.waitKey(1) & 0xFF		#bitwise and	#32 bit integer and 8-bit int to make the 8 bit ans

	if key_pressed == ord('q'):		#ord tells the ascii value of that character
		break

#convert our face list array into numpy array
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save this data into file system
np.save(dataset_path+file_name+'.npy', face_data)
print("Data successfully save at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()