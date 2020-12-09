import numpy as np
import cv2
import requests

class capture_video(object):
    def __init__(self):
        self.cap= cv2.VideoCapture(0)
        

    def __del__(self):
        self.cap.release()
        

    def get_video(self):

        face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
         

         


        

        ret, image = self.cap.read()
            
        if ret ==True:
            face= face_cascade.detectMultiScale(image, 1.3, 5)
            

            for (x, y, w, h) in face:
                image = cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 3)

                roi_color= image[y:y+h, x:x+w]

                eyes = eye_cascade.detectMultiScale(roi_color)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

            frame = cv2.imencode('.jpg', image)[1]
            data= []
            data.append(frame.tobytes())

        return data       

               

        

        

