<h2><b>A summary on object detection using OpenCV</b></h2>
OpenCV (Open Source Computer Vision library), originally developed by Intel, is a library which is very useful for solving real-time 
computer vision problems and other image related solutions.
The topic of focus in this module is Object detection and its development. This module covers image segmentation, contours, shape detection and feature detection
and the models are deployed in Flask.
Object detection is a process of identifying the required class of objects in an image. Object detection becomes handy for solving real life problems like traffic checking, robot vision,
vehicle detection, face detection, etc.
<ul><br>
<li><b>Image processing</b></li>
The images are supposed to be processed before further actions are performed on it. Most of the algorithms and commands require that the image be converted to grayscale before operations are performed on it.
Blending, blurring, smoothing, color mapping, corner detection, edge detection and grid detection are some of the common processing techniques. These processes enhance the images and make it ready for feature extraction.
<br><br>
<li><b>Image segmentation</b></li>
The first method that we come across is image segmentation which is a process of dividing an image into parts called segments to simplify image analysis. Commands like cv.watershed
are used along with distance transform to identify objects. cv.watershed uses the principle that a grayscale image can be identified as a topography of peaks and valleys based on the pixel values and
that boundaries can be identified based on how the outlines form when colors are filled in the valleys. This method is efficient for the recognition of simple non touching objects but it has its limitations when objects are touching.
<br><br>
<li><b>Contours</b></li>
Contours are curves which define the outline of an object based on same pixel intensities. The image has to be HSV(hue, saturation and lightness) or grayscale for contours to work. The main application 
of contours is shape and size detection of objects, i.e, finding the different shapes present in an image. Contouring can be made more effective by using thresholding and cv2.Canny() for thresholding hysteresis and to made the edges sharper to make recognition easier. 
<br><br>
<li><b>Feature Detection</b></li>
Features are distinctive properties of an object which help us to distinguish one object from another. Our brain registers different attributes of what we see to recognise and object and differentiate it from other objects.
A computer works on similar lines by recording various properties to identify an object. For an image, corners are the best features as they give maximum variation when moved by a small amount in regions around it.
Harris corner detection, Shi-Tomasi detector(Good features to track), SIFT, SURF, FAST, BRIEF and ORB are the corner detectors which have improved on speed and performance over time in the order given. 
Feature matching is a way of matching features between two images. A brute force based matcher matches one feature of an image to all features in another image and returns the feature with the minimum distance measured. 
There is another improved faster version called the FLANN based matcher which used various algorithms to find the nearest neighbour features. 
<br><br>
<li><b>Model building and Deployment</b></li>
After the completion of making and training a model, we have to deploy in so that non technical people can also use our application. For this purpose, the model and its weights are generally stored
as a pickle file. Flask API is commonly used to deploy web applications and opencv projects can be brought into deployment using Flask. It is an easy to use API and can work well with other python programs. The pickle files are imported 
into the flask program and the application can be hosted on a website.</ul>
<br>
Name: Mariya Jojy<br>
Contact no.: 7020448621<br>
email id: mariyajojy27@gmail.com
