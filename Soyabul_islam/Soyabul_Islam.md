# Object Detection:
Object detection is related to computer vision and image processing where objects of different class like car, human, clock etc is detected.

<img src="https://github.com/SoyabulIslamLincoln/OpenCV-Module-2/blob/main/Soyabul_islam/real_time_object_detection.jpg" alt="Object Detection" style="height: 100px; width:100px;"/> 




For getting better result in detecting object the main thing we need is process our imgae. With the help of openCV we can do  all that. This open source  library has so many functions that we can execute to our photo. I will be discussing some of them , how they work and what we can do with the help of them.


### Image Segmentation

We can view any grayscale image as topographic surface. In topographic surface high intensity means peaks and low intensity denotes valleys.There is a "philosophy" naming watershed. This approach gives oversegmented result due to noise. So in openCV, a marker based watershed algorithm is used. Here we give different labels for our object, label the region we know sure of foreground and give other label to which region we don't know. First we can see by using otsu's binaries. Noice can be very problematic.To remove noice we can use morphological closing.By using Erosion we can remove the boundaries of the object we are trying to separate by using proper threshold.To detect which are not our object , we use Dilation. Dilation increases object boundary to backgorund.The remaining regions are those which we don't have any idea, whether it is our desired object or background. Watershed algorithm should find it.
```python
image_name= cv2.imread("location")
gray= cv2.cvtColor(image_name, cv2.COLOR_BGR2GRAY)
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
