# Advanced-Lane-Lines
##Udacity SDC Nanodegree Project 4

In this project, I found acceptable thresholds upon which to detect lane lines from an image, gave it binary activation, transformed the image to a bird's eye-like view, calculated the curvature of the lane lines, and drew back on the lane lines detected onto the image. The output video was the re-drawn lines, along with the car's position with respect to the center of the lane, and the curvature of the lane.

###Camera Calibration

The code for this step is contained in lines 12-38 of 'full_pipeline.py'.  

Every camera comes with a little bit of distortion in the images it puts out. Since each camera's distortion can be different, a calibration must be done in order to correct the image and make it appear as in the real-world, undistorted. Luckily, OpenCV provides an easy way to do so using chessboard images taken by the camera. I pulled in some of Udacity's provided chessboard images from their camera to begin with. 

Next, based on the number of inner corners on the chessboard, I prepare the "object points", which are the (x, y, z) coordinates of where the chessboard corners are in the real world. This particular instance will have me assume z=0, as the chessboard is fixed on a flat (x, y) plane so that each calibration image I use will have the same object points. Next, I am going to grab my "image points", which are where the chessboard corners are in the image. I can do this with cv2.findChessboardCorners(), a useful function for finding just where those inner corners lie in the image. Note that I have appended the object points and image points to lists to use in the next function.

Now that I have my object points and image points, I can calibrate my camera and return the distortion coefficients. Using at least 10 images, cv2.calibrateCamera() can calculate the camera matrix and distortion coefficients (along with a few other pieces of information). I then use cv2.undistort() to see an undistorted chessboard, as shown below.

![Chessboard](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/chessboard.PNG "Original and Undistorted Chessboard")

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Orig & Undist Road](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/orig_and_undist.PNG "Original and Undistorted Road")
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![Mag](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/undist_and_mag.PNG "Magnitude of sobelx and sobely")
![Sobelx](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/orig_and_sobelx.PNG "Sobelx thresholded")
![RGB](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/rgb.PNG "RGB thresholds")
![HLS](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/hls.PNG "HLS thresholds")

![R, 200-255](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/r_threshold.PNG "R thresholded, 200-255")
![S, 150-255](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/s_threshold.PNG "S thresholded, 150-255")
![Activation](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/pipeline_color_and_binary.PNG "S and Sobelx activations and full pipeline activation (S, Sobelx, or R)")

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Parallel Check](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/birds_eye.PNG "Checking for parallel")
![Bird's Eye](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/birds_eye2.PNG "Warped image of the road")
![Binary Warp](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/birds_eye3.PNG "Binary version of warp")
![Binary Road](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/binary_bird.PNG "Binary road")

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![Histogram](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/histogram.PNG "Histogram of activation areas")
![Sliding Windows](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/sliding_window.PNG "Using sliding windows")
![Search](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/search.PNG "Search areas")

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![Result](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/final.PNG "Final result")

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./reg_vid.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
