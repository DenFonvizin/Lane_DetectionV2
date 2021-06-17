Lane detection
To complete the first part of the comprehensive assignment we should track the lines on the road and measure curvature and offset from the center of the road within the given video. For this task we use openCV Computer Vision using Python (Jupyter Notebook variation, because of the usage of Coolab for better and faster computation). First, we should prepare the calibration part of the video processing algorithm in order to avoid optic distortion - a physical phenomenon that occurs in image recording, in which straight lines are projected as slightly curved ones when perceived through camera lenses. The highway driving video is recorded using the front facing camera on the car and the images are distorted. The distortion coefficients are specific to each camera and can be calculated using known geometrical forms.
For calibration purposes we used chessboard settled on the walls in different positions and angles of view (The images provided present 9 * 6 corners to work with):

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
nx = 9
ny = 6
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
After processing all images in the calibration set, the image point list has enough data to compare against the object points in order to compute camera matrix and distortion coefficients. This leads to an accurate camera matrix and distortion coefficient identification using the ‘calibrateCamera’ function.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
undist = cv2.undistort(img, mtx, dist, None, mtx)
After calibration we define the AoI (area of interest for the video) and offset (additional distance from the AoI).
#Source points become parallel after warp transformation

src = np.float32([
    (320, 540), # bottom-left corner
    (460, 380), # top-left corner
    (525, 380), # top-right corner
    (730, 540) # bottom-right corner
])

# Destination points must be parallel
dst = np.float32([
    [offset, img_size[1]],             # bottom-left corner
    [offset, 0],                       # top-left corner
    [img_size[0]-offset, 0],           # top-right corner
    [img_size[0]-offset, img_size[1]]  # bottom-right corner
])
For getting a warped image we use the transformation matrix calculated by the transformation matrix provided by the OpenCV.
# Calculate the transformation matrix and its inverse transformation
M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(undist, M, img_size)
To process the image in a way that the lane line pixels are preserved and easily differentiated from the road we use x sobel transformation on the gray-scaled image.
# Transform image to gray scale
gray_img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply sobel (derivative) in x direction, this is usefull to detect lines that tend to be vertical
sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
abs_sobelx = np.absolute(sobelx)

# Scale result to 0-255
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
sx_binary = np.zeros_like(scaled_sobel)

# Keep only derivative values that are in the margin of interest
sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1

The lane line detection is performed on processed binary thresholded images that have already been undistorted and warped.

# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

To speed up the lane line search from one video frame to the next, information from the previous cycle is used. It is more likely that the next image will have lane lines in proximity to the previous lane lines. This is where the polynomial fit for the left line and right line of the previous image are used to define the searching area.
To calculate the radius and the vehicle's position on the road in meters, scaling factors are needed to convert from pixels. A polynomial fit is used to make the conversion. Using the x coordinates of the aligned pixels from the fitted line of each right and left lane line, the conversion factors are applied and polynomial fit is performed on each.
After that all information is added to the video frame and polygons between detected lines are drawn.
