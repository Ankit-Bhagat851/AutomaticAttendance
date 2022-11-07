import cv2
import numpy as np
import face_recognition
imgAnkit=face_recognition.load_image_file('Ankit Kumar Bhagat.jpg')
imgAnkit = cv2.cvtColor(imgAnkit,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Ujjawal Gahoi.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
faceLoc = face_recognition.face_locations(imgAnkit)[0] #detect face location in the image
encodeElon = face_recognition.face_encodings(imgAnkit)[0] # encode the detected face
cv2.rectangle(imgAnkit,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) #put the rectangle on the face
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
results = face_recognition.compare_faces([encodeElon],encodeTest) #compare the image on the basis of encoding and returns true or false
faceDis = face_recognition.face_distance([encodeElon],encodeTest) #how similar the image is
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('Ankit',imgAnkit)
cv2.imshow('Test',imgTest)
k=cv2.waitKey(0)
cv2.destroyAllWindows()
