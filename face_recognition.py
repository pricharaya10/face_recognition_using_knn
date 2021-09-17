#recognise face using some classification algorithm - like logistic, knn, svm etc

#1. load the training data(numpy arrays of all the persons)
		#x-values are stored in the numpy arrays
		#y-values we need to assign for each person
#2.Read a video stream using opencv
#3. extract faces out of it (for testing purpose)
#4. use knn to find the prediction of face(int )
#5.	map the predicted id of the user
#6. Display the predictions on the screen--bounding box and name

import  cv2
import numpy as np 
import os


########## KNN CODE ###########
def distance(v1,v2):
	#eucledian
	return np.sqrt(sum(v1-v2)**2)

# def knn(train, test, k=5):
# 	dist = []

# 	for i in range(train.shape[0]):
# 		#get the vector and label
# 		ix=train[i, :-1]
# 		iy=train[i, :-1]
# 		#compute distance from test point
# 		d=distance(test, ix)
# 		dist.append([d, iy])

# 	#sort based on distance and get top k
# 	dk=sorted(dist, key=lambda x: x[0])[:k]
# 	#retrieve only the labels
# 	labels=np.asarray(dk)[:, -1]

# 	#get frequency of each label
# 	output = np.unique(labels, return_counts=True)

# 	#find max frequency and corresponding label
# 	index=np.argmax(output[1])
# 	return output[0][index]

def knn(x, train, targets, k=5):
    m = train.shape[0]
    dist = []
    for ix in range(m):
        # compute distance from each point and store in dist
        dist.append(distance(x, train[ix]))
    dist = np.asarray(dist)
    indx = np.argsort(dist)
    sorted_labels = targets[indx][:k]
    counts = np.unique(sorted_labels, return_counts=True)
    return counts[0][np.argmax(counts[1])]

#########################################


#Init Camera
cap=cv2.VideoCapture(0)

#Face Detection
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0
dataset_path='./data/'

face_data=[]
labels=[]

class_id=0		#labels for the given file
names={}		#mapping bet id and name

#data preparation
for fx in os.listdir(dataset_path):		#we get all the file in the data folder
	if fx.endswith('.npy'):
		#create a mapping bet class_id and name
		names[class_id]=fx[:-4]
		print("loaded "+fx)
		data_item=np.load(dataset_path+fx)
		face_data.append(data_item)

		#create labels for the class
		target=class_id*np.ones((data_item.shape[0],))
		class_id+=1
		labels.append(target)


face_dataset=np.concatenate(face_data, axis=0)
face_labels=np.concatenate(labels, axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)


trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)


#testing 
while True:
	ret, frame=cap.read()

	if ret==False:
		continue

	faces=face_cascade.detectMultiScale(frame, 1.3, 5)

	for face in faces:
		x,y,w,h=face

		#get the face region of interest(ROI)
		offset=10
		face_section=frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))
		print(face_section, "123")
		# predicted label (out)
		out=knn(face_section.flatten(), face_dataset, face_labels)

		# Display on the screen and rectangle around it
		pred_name=names[int(out)]
		cv2.putText(frame, pred_name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)


	cv2.imshow("Faces",frame)

	key=cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

cap.release() 
cv2.destroyAllWindows()
