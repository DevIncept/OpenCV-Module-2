# Object Detection:
Object detection is related to computer vision and image processing where objects of different class like car, human, clock etc is detected.

<img src="https://github.com/SoyabulIslamLincoln/OpenCV-Module-2/blob/main/Soyabul_islam/real_time_object_detection.jpg" alt="Object Detection" style="height: 100px; width:100px;"/> 




For getting better result in detecting object the main thing we need is process our imgae. With the help of openCV we can do  all that. This open source  library has so many functions that we can execute to our photo. I will be discussing some of them , how they work and what we can do with the help of them.


### Image Segmentation

We can view any grayscale image as topographic surface. In topographic surface high intensity means peaks and low intensity denotes valleys.There is a "philosophy" naming watershed. This approach gives oversegmented result due to noise. So in openCV, a marker based watershed algorithm is used. Here we give different labels for our object, label the region we know sure of foreground and give other label to which region we don't know. First we can see by using otsu's binaries. Noice can be very problematic.To remove noice we can use morphological closing.By using Erosion we can remove the boundaries of the object we are trying to separate by using proper threshold.To detect which are not our object , we use Dilation. Dilation increases object boundary to backgorund.The remaining regions are those which we don't have any idea, whether it is our desired object or background. Watershed algorithm should find it.

```python
image_name= cv2.imread("location")
gray= cv2.cvtColor(image_name, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)


ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
```



### Contour:
Contours are basically a list of points of x and y , which represent the position of object as outerline or border. Contour is very handy on getting the size nad shape of the object. But for detecting contour perfectly we cannot use BGR image, We need to convert it to grayscale image or hsv image.

```python
image_name= cv2.imread("location")
gray= cv2.cvtColor(image_name, cv2.COLOR_BGR2GRAY)
```


or 
```python
image = cv2.cvtColor(image_name, cv2.COLOR_BGR2HSV)
```

The function <i><b>cv2.findContours()</b></i> give three arguments, <b>Image, contour retreaval mode, contour approximation mode</b>. To obtain all the contours 
<b><i>cv2.drawContours()</i></b> is necessary.By blurring the background we can detect contours more nicely.

```python 
 blur= cv2.GaussianBlur(gray, (5,5), 0)
        

thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh= cv2.dilate(thresh, None, iterations=2)
contour= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image= cv2.drawContours(thresh, contour, -1, (0, 255, 255), 3)
```


### Canny Edge Detection:

<i><b>cv2.Canny()</b></i> function is very handy for making contours more efficient. This function reduce noise, find intensity gradient, non maximum suppression, hysteresis thresholding all by itself.
We can detect edge easily eith the help of this function.

```python
canny= cv2.Canny(image, threshold1, theshold2)
```


### Finding Deifferent Shapes:

For this first we have get the contour from image or video that we load. Then the perimeter of the contour is calculated using <b><i>cv2.arcLength()</i></b> function. Then we use <b><i>cv2.approxPolyDP()</i></b> function for appoximation
 The approximation gives us number of vetices.Then we specify some conditions for detecting shapes from image.
 
 ```python
 perimeter= cv2.arcLength(contour, True)
approx= cv2.approxPolyDP(contour, 0.04*perimeter, True
if len(approx) == 3:
   shape = "Triangle"
elif len(approx) == 4:
   (x , y , w , h) = cv2.boundingRect(approx)
   av = w / float(h)
   shape = "Square" if av >= 0.95 and av <= 1.05 else "Rectangle"
elif len(approx) == 5:
    shape = "Pentagon"
else:
    shape = "Circle"
    
# We can give more conditions to fing hexagon, octagon , heptagon etc
```


### Corner Detection:

Harris Corner detection is the best method for detecting corners . It basically finds the difference in the intensity for a displacement in all direction.

In openCV we call <i><b>cv2.cornerHarris()</b></i> function for detecting corners. It takes 4 parameters. Those parameters are Image (which smust be converted to gray image), blocksize, ksize(Aperture parameter of Sobel derivative used), k(Harris detector free parameter)

```python

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

image[dst>0.01*dst.max()]=[0,0,255]
```


### Face Detection:

Haar Classifier is the most used library for detection face, moth, body, eye, number plate etc propsed by aul Viola and Michael Jones. In their paper, "Rapid Object Detection using Boosted Cascade of Simple Features" in 2001. It is mainly a machin learning based approach.

There are xml files for specific part detection for face, eyes, fullbody etc we simply load them to our model. Here is a code of mine detecting face and eyes from live video stream.

```python
face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

video_capture= cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('record.avi', fourcc, 20.0, (640, 480)) 

while (video_capture.isOpened()):
    ret, frame= video_capture.read()

    if ret ==True:
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face= face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in face:
            frame= cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 3)
            roi_gray= gray[y:y+h, x:x+w]
            roi_color= frame[y:y+h, x:x+w]
            
            eyes= eye_cascade.detectMultiScale(roi_color)
            #mouth= mouth_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 0, 255), 2)
        
        out.write(frame)

     
        cv2.imshow('Image', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    
    else:
        break


video_capture.release()
out.release()
cv2.destroyAllWindows()
```

Here you can see the result.


[![Face Detection](https://share.gifyoutube.com/SnBnwKXd5Rg.gif)](https://youtu.be/SnBnwKXd5Rg)


## Flask:
It is micro web framework written in python. To deploy our model to server it is a very helpul framework. 
After importing  <b>Flask</b> , we craete a n instance of this class. We then use the route() decorator to tell Flask what URL should trigger our function.
Here's a demo code for running a simple URL.


```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'
    
```    

You can find Face detection app developemnt with flask's code from the link.


[Face Detection Web App](https://github.com/SoyabulIslamLincoln/OpenCV-Module-2/tree/main/Soyabul_islam/web_app_for_face_detection)
