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

Now that we can undistort the images from the camera, I'll apply this to a road image, as shown below - it is not quite as obvious of a change as the chessboard is, but the same undistortion is being done to it.

![Orig & Undist Road](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/orig_and_undist.PNG "Original and Undistorted Road")

The code for this step is in the pipeline() function in my code, or lines 79-117.

Next up, I looked at various color and gradient thresholds to choose how I want to perform my thresholding and generate a binary image. Note that not all of what I looked at is still in the 'full_pipeline.py' file, so I have added in a few clips below as necessary to show how to arrive at these.

####Magnitude threshold using combined Sobelx and Sobely

This uses the square root of the combined squares of Sobelx and Sobely, which check for horizontal and vertical gradients (shifts in color, or in the case of our images, lighter vs. darker gray after conversion), respectively.

Here's the code I used to run this on an image, since it's not in my final file.
```
def mag_thresh(img, sobel_kernel, mag_thresh):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Calculate the magnitude
    mag = np.sqrt(np.square(sobelx)+np.square(sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_mag = np.uint8(255*mag/np.max(mag))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_mag)
    binary_output[(scaled_mag >= mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
    
# Run the function
mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(50, 200))

```
And the end result!
![Mag](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/undist_and_mag.PNG "Magnitude of sobelx and sobely")
I did not use this one because it does not do a great job at detecting the left yellow line, especially over the lighter portion of the road.

####Sobelx threshold

I already explained this one a bit above, and it is in the final product, so I'll just show the resulting image. I used a threshold of between 10 and 100 here (from between 0-255 in a 256 color space).
![Sobelx](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/orig_and_sobelx.PNG "Sobelx thresholded")
This one detects the yellow well on the lighter portion of the image, and white is also clear. I like this one.

####RGB color thresolds

I next checked the different RGB color thresholds. The end result only uses the R treshold, but the below code snippet can get you any of them. Note that you must set cmap='gray' to see the images like the below versions.
```
R = image[:,:,0]
G = image[:,:,1]
B = image[:,:,2]
```
![RGB](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/rgb.PNG "RGB thresholds")
The R color channel definitely appears to see the lines the best, so I'll use this.

####HLS color thresholds

The last thresholds I checked were in the HLS color space. My final product only uses S, but here's how to pull out all the HLS channels.
```
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]
```
![HLS](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/hls.PNG "HLS thresholds")
The S color channel looks the best here, so I'll continue on with that.

####Limiting the thresholds

I also did some limiting of how much of the color space of each threshold I wanted to try to narrow it down to just the lane lines. I have shown what I found to be the optimal thresholds for the binary images in the R & S spaces below. Note that I used 200-255 for the R threshold, and 150-255 for the S threshold in these images.
![R, 200-255](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/r_threshold.PNG "R thresholded, 200-255")
![S, 150-255](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/s_threshold.PNG "S thresholded, 150-255")

I came up with a final pipeline where I actually used a combination of Sobelx, S and R. In my code (see lines 112-115), if any two of the three are activated, then I want the binary image to be activated. If only one is activated then it does not get activated. Note that due to this approach, I expanded the thresholds on S from the above images to 125-255, as it improved the final returned binary image by a little bit (R and Sobelx stayed the same).  

Below is an undistorted image, followed by one showing in separate colors the S and Sobelx activations, and then the final binary image where any two of the three being activated causes a final binary activation.
![Activation](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/pipeline_color_and_binary.PNG "S and Sobelx activations and full pipeline activation (S, Sobelx, or R)")

####Perspective transformation

Next up is perspective transformation. This will make the image look like a bird's eye view of the road. In fact, that's exactly how I named the function that does this (Lines 114-148). After undistorting the image, I define source ("src") points of where I want the image to be transformed out from - these are essentially the bottom points of the left and right lane lines (based on when the car was traveling on a straight road), and the top of the lines, slightly down from the horizon to account for the blurriness that begins to appear further out in the image. From there, I also chose destination ("dst") points which are where I want the source points to end up in the transformed image. The code containing these points and a chart is shown below:

```
src = np.float32([[690,450],[1110,img_size[1]],[175,img_size[1]],[595,450]])

offset = 300 # offset for dst points
dst = np.float32([[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],
    [offset, img_size[1]],[offset, 0]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 690, 450      | 980, 0        | 
| 1110, 720     | 980, 720      |
| 175, 720      | 300, 720      |
| 595, 450      | 300, 0        |

To verify that these points worked, I drew the lines (using cv2.line() with the source points on the original image and the destination points onto the transformed image) back onto the images to check that they were mostly parallel. Note that this was done on one of the provided images of a straight road, as opposed to some of the curves I have been using as example images so far.

![Parallel Check](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/birds_eye.PNG "Checking for parallel")

Here is the original image and an image that has been perspective transformed of a curve.
![Bird's Eye](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/birds_eye2.PNG "Warped image of the road")

Here is a binary version of doing the same process.
![Binary Warp](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/birds_eye3.PNG "Binary version of warp")

####Finding and Fitting the Lines

The code for this section can be found primarily in lines 157-280 (for finding the first lines, or if the program has lost track of the lines) and 294-453 (if it has a guess as to where the lines are). I am also keeping of some important information about each line by using python classes in lines 40-77.

At this point, I have some nice, perspective transformed, binary images. The next step was to plot a histogram based on where the binary activations occur across the x-axis, as the high points in a histogram are the most likely locations of the lane lines.

This histogram follows pretty close to what I expected from the last binary warped image above.
![Histogram](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/histogram.PNG "Histogram of activation areas")

Now that we have a decent histogram, we can search based off the midpoint of the histogram for two different peak areas. Once the function 'first_lines()' has its peaks, it will use sliding windows (the size of which can be changed within the function) to determine where the line most likely goes from the bottom to the top of the image.

Note that the final version of my code does not stop to spit out an image anymore like the below one. At the end of the included code for first_lines(), adding back the below code would spit out the image of the sliding windows. 
```
# Generate x and y values for plotting
fity = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(fit_leftx, fity, color='yellow')
plt.plot(fit_rightx, fity, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()
```
The calculated line is not perfectly parallel, but it still does a decent job.
![Sliding Windows](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/sliding_window.PNG "Using sliding windows")

Note that I ended up saving important information about the lines into separate classes - I do not end up using it for much in my current final version, but a previous iteration in which I took on the challenge videos (further discussed in the "Discussion" section at the end) included various checks utilizing this information. The "try" and "except" portions are based on various errors I ran into working on the challenge videos.

Now that I have found the lines the first time, the full sliding windows calculation is no longer needed. Instead, using the old lines as a basis, the program will search within a certain margin of the first lines detected (in the draw_lines() function). I also added in a counter (Lines 150-155), where if 5 frames in a row fail to detect a line, first_lines() will be run again and the sliding windows will again be used. Note that the counter gets reset if the line is detected again before reaching five.

The below image shows the search areas used around the original line to check for the subsequent line.
![Search](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/search.PNG "Search areas")

####Radius of curvature and position of the vehicle

Two important pieces of information (which can also be used to determine the reasonableness of the returned lines) about the image are what the curvature of the road is, and where the vehicle is in the lane compared to center. If the radius of the curvature were too low, it is probably unlikely, unless it is an extreme curve. A high radius of curvature would seem odd unless it is on a straight road. Also, if the car were very far from the center, perhaps the car is calculating a line for a different lane, or off the road.

I calculated these within the draw_lines() function, primarily within Lines 392-430. For the lane curvature, I first have to convert the space from pixels to real world dimensions (Lines 397-399). Then, I calculate the polynomial fit in that space. I used the average of the two lines as my lane curvature.

For the car's position vs. center, I calculated where the lines began at the bottom of the picture (using second_ord_poly() defined at 282-292 with the image's y-dimension size plugged in). I then compared this to where the middle of the image was (assuming the car camera is in the center of the car), after converting the image's x-dimension to meters. This gets printed onto the image as shown below.

####The Result

Lines 455-461 can process an image through all of the above (the function is 'process_image'). Here is an example of the final result on a single image:

![Result](https://github.com/mvirgo/Advanced-Lane-Lines/blob/master/Images/final.PNG "Final result")

---

###Pipeline (video)

Lines 467-476 will process a video by going through each video frame as an image in the above described process.

Here's a [link to my video result](./reg_vid.mp4)

My program does a pretty good job!
---

###Discussion

####Challenges

This project took me longer than the first three, likely due to my lack of overall experience in computer vision. Given that I have done most of the Machine Learning nanodegree, the deep learning portions of earlier projects fit more into what I had prior knowledge of. However, I eventually found that changing thresholds (both which ones used as well as restricting the range within those) was still similar to machine learning - I was finding the important features and tuning parameters. The same was the case in determining the source and destination points, as well as the window size and search area of the first_lines() and draw_lines() functions.

The best approach I eventually found with the thresholds was to try to attack the hardest test images first. This led to me focusing on the images with changes in the color of the lane, as well as large shadows. By doing so (similar to how certain boosting algorithms work in machine learning), it actually made for a better ending pipeline. This took a significant amount of time, but I think definitely improved the end product (which only produces some slight shakiness when the road changes color from dark to light and back again in the video).

As I mentioned above, I did take a stab at the challenge videos, with a little (but unfortunately not complete) success. My final version included in this repository has removed some of what I had added to help with the challenge videos, but I will discuss that further below.

####Potential improvements

A big part of the code that is now slightly unnecessary (although included because I hope to iterate directly on it in the future, and I still believe it improves the end product) was my inclusion of classes for each line. These stemmed from my effort to remove the remaining slight shakiness of the line when the road changes color, as well as taking a crack at the challenge videos. Older versions included both a single_check() and dual_check() that first would compare a single line to its previous fit for reasonabless (if it change too much it was likely incorrect and should be discarded in favor of either the last fit or the best fit), and then to compare both lines together for whether they were parallel, as well as whether the left line was actually an appropriate distance left of the right line, and vice versa.

These changes actually got me fairly close on the challenge videos - one version had decent lines drawn for roughly 75% of the challenge video! But when I went back to check it on the regular project video, it had actually managed to get significantly worse under normal conditions. So while I had improved it in more challenging situations, the improvements had clearly been too overfit to the challenge at hand, and was no longer generalizing well (I really did start seeing this like a machine learning problem).

Another potential improvement I thought of was to potentially use deep learning, in one of two ways. First, I could try to enhance the images by training a deep neural network using original images followed by the result being manually drawn on lane lines in more vibrant colors. The neural network would therefore be trained to spit out these improved images. This could then be fed into the above process using more restricted thresholds (as the lane lines could be put down to very specific color spaces after being manually drawn). This of course could take massive amounts of manual image processing.

The second potential option with deep learning would be to jump the process above almost entirely, and instead teach the deep neural network not to generate a new image, but to calculate out the resulting polynomial function - essentially, two regression problems to each of the two lane lines. The lines could then be drawn on using only the very end of the above process (or potentially some new process). This would probably be a bit easier, as the lines could probably be fit with software that calculates polynomials on a perspective transform image for training purposes, but the neural network could potentialy learn to skip this step and figure it out just based on the normal image.

####Conclusion
Overall, it was a very challenging project, but I definitely have a much better feel for computer vision and how to detect lane lines after spending extra time on it. I hope to eventually tackle the challenge videos as well, but for now I think I have a pretty solid pipeline!
