<h1>Command to run the file<h1>
<code>python test_grader.py --image images/omr_test_01.png</code>
<h1>Code Explanation: Exam Sheet Processing</h1>
    
<div class="explanation">
<h2>1. Import Packages</h2>
<p>
<code>from imutils.perspective import four_point_transform</code><br>
<code>from imutils import contours</code><br>
<code>import numpy as np</code><br>
<code>import argparse</code><br>
<code>import imutils</code><br>
<code>import cv2</code>
</p>
<p>
These lines import the necessary packages:
<ul>
<li><code>imutils.perspective</code>: Contains utilities for geometric transformations.</li>
<li><code>imutils.contours</code>: Contains functions for working with contours in images.</li>
<li><code>numpy</code>: A library for numerical operations in Python.</li>
<li><code>argparse</code>: A module for parsing command-line arguments.</li>
<li><code>imutils</code>: A collection of convenience functions for computer vision tasks.</li>
<li><code>cv2</code>: The OpenCV library for computer vision tasks.</li>
</ul>
</p>
</div>

<div class="explanation">
<h2>2. Argument Parsing</h2>
<p>
<code>ap = argparse.ArgumentParser()</code><br>
<code>ap.add_argument("-i", "--image", required=True, help="path to the input image")</code><br>
<code>args = vars(ap.parse_args())</code>
</p>
<p>
<code>argparse.ArgumentParser()</code>:
<ul>
<li><strong>Description:</strong> Initializes the argument parser.</li>
<li><strong>Parameters:</strong> None</li>
<li><strong>Returns:</strong> An ArgumentParser object.</li>
</ul>
<code>add_argument</code>:
<ul>
<li><strong>Description:</strong> Adds a command-line argument.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>name or flags</code>: "-i" or "--image" for the input image path.</li>
<li><code>required</code>: Indicates that this argument is mandatory.</li>
<li><code>help</code>: Provides a description of the argument.</li>
</ul>
</li>
<li><strong>Returns:</strong> The ArgumentParser object, allowing further configuration.</li>
</ul>
<code>parse_args</code>:
<ul>
<li><strong>Description:</strong> Parses command-line arguments.</li>
<li><strong>Parameters:</strong> None</li>
<li><strong>Returns:</strong> Namespace object with arguments as attributes.</li>
</ul>
</p>
</div>

<div class="explanation">
<h2>3. Image Loading and Preprocessing</h2>
<p>
<code>image = cv2.imread(args["image"])</code><br>
<code>gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)</code><br>
<code>blurred = cv2.GaussianBlur(gray, (5, 5), 0)</code><br>
<code>edged = cv2.Canny(blurred, 75, 200)</code>
</p>
<p>
<code>cv2.imread</code>:
<ul>
<li><strong>Description:</strong> Loads an image from a file.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>filename</code>: Path to the image file.</li>
</ul>
</li>
<li><strong>Returns:</strong> The loaded image as a NumPy array.</li>
</ul>
<code>cv2.cvtColor</code>:
<ul>
<li><strong>Description:</strong> Converts an image from one color space to another.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>src</code>: Source image.</li>
<li><code>code</code>: Color conversion code, <code>cv2.COLOR_BGR2GRAY</code> converts to grayscale.</li>
</ul>
</li>
<li><strong>Returns:</strong> The converted image.</li>
</ul>
<code>cv2.GaussianBlur</code>:
<ul>
<li><strong>Description:</strong> Applies Gaussian blur to an image.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>src</code>: Source image.</li>
<li><code>ksize</code>: Size of the Gaussian kernel (5, 5).</li>
<li><code>sigmaX</code>: Standard deviation in the X direction.</li>
</ul>
</li>
<li><strong>Returns:</strong> The blurred image.</li>
</ul>
<code>cv2.Canny</code>:
<ul>
<li><strong>Description:</strong> Applies the Canny edge detector.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>image</code>: Source image.</li>
<li><code>threshold1</code>: Lower threshold for edge detection (75).</li>
<li><code>threshold2</code>: Upper threshold for edge detection (200).</li>
</ul>
</li>
<li><strong>Returns:</strong> Image with detected edges.</li>
</ul>
</p>
</div>

<div class="explanation">
<h2>4. Contour Detection and Document Extraction</h2>
<p>
<code>cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)</code><br>
<code>cnts = imutils.grab_contours(cnts)</code><br>
<code>docCnt = None</code><br>
<code>if len(cnts) > 0:</code><br>
<code>cnts = sorted(cnts, key=cv2.contourArea, reverse=True)</code><br>
<code>for c in cnts:</code><br>
<code>peri = cv2.arcLength(c, True)</code><br>
<code>approx = cv2.approxPolyDP(c, 0.02 * peri, True)</code><br>
<code>if len(approx) == 4:</code><br>
<code>docCnt = approx</code><br>
<code>break</code>
</p>
<p>
<code>cv2.findContours</code>:
<ul>
<li><strong>Description:</strong> Finds contours in a binary image.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>image</code>: The source image.</li>
<li><code>mode</code>: Contour retrieval mode. <code>cv2.RETR_EXTERNAL</code> retrieves only the outer contours.</li>
<li><code>method</code>: Contour approximation method. <code>cv2.CHAIN_APPROX_SIMPLE</code> compresses horizontal, vertical, and diagonal segments.</li>
</ul>
</li>
<li><strong>Returns:</strong> A list of contours.</li>
</ul>
<code>imutils.grab_contours</code>:
<ul>
<li><strong>Description:</strong> Grabs the contours from the result of <code>cv2.findContours</code>.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>contours</code>: The result from <code>cv2.findContours</code>.</li>
</ul>
</li>
<li><strong>Returns:</strong> The contours in a standardized format.</li>
</ul>
<code>cv2.arcLength</code>:
<ul>
<li><strong>Description:</strong> Computes the perimeter of a contour.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>curve</code>: The contour.</li>
<li><code>closed</code>: Whether the contour is closed.</li>
</ul>
</li>
<li><strong>Returns:</strong> The perimeter length.</li>
</ul>
<code>cv2.approxPolyDP</code>:
<ul>
<li><strong>Description:</strong> Approximates a contour shape to another shape with fewer vertices.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>curve</code>: The contour.</li>
<li><code>epsilon</code>: The approximation accuracy (e.g., 0.02 * peri).</li>
<li><code>closed</code>: Whether the contour is closed.</li>
</ul>
</li>
<li><strong>Returns:</strong> The approximated contour.</li>
</ul>
</p>
</div>

<div class="explanation">
<h2>5. Perspective Transformation and Thresholding</h2>
<p>
<code>paper = four_point_transform(image, docCnt.reshape(4, 2))</code><br>
<code>warped = four_point_transform(gray, docCnt.reshape(4, 2))</code><br>
<code>thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]</code><br>
<code>cv2.imwrite("thresed_image.jpg", thresh)</code>
</p>
<p>
<code>four_point_transform</code>:
<ul>
<li><strong>Description:</strong> Applies a four-point perspective transform to obtain a top-down view of the paper.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>image</code>: The source image.</li>
<li><code>pts</code>: The four points defining the perspective transform.</li>
</ul>
</li>
<li><strong>Returns:</strong> The warped image.</li>
</ul>
<code>cv2.threshold</code>:
<ul>
<li><strong>Description:</strong> Applies a threshold to an image.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>src</code>: The source image.</li>
<li><code>thresh</code>: The threshold value (0 here, as Otsu’s method is used).</li>
<li><code>maxval</code>: The maximum value to use with the binary thresholding (255).</li>
<li><code>type</code>: The type of thresholding (binary inverse and Otsu’s threshold).</li>
</ul>
</li>
<li><strong>Returns:</strong> A tuple containing the threshold value used and the thresholded image.</li>
</ul>
<code>cv2.imwrite</code>:
<ul>
<li><strong>Description:</strong> Saves an image to a file.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>filename</code>: The path to the output file.</li>
<li><code>img</code>: The image to be saved.</li>
</ul>
</li>
<li><strong>Returns:</strong> <code>True</code> if the image was saved successfully, otherwise <code>False</code>.</li>
</ul>
</p>
</div>

<div class="explanation">
<h2>6. Contour Detection for Questions and Answers</h2>
<p>
<code>cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)</code><br>
<code>cnts = imutils.grab_contours(cnts)</code><br>
<code>questionCnts = []</code><br>
<code>for c in cnts:</code><br>
<code>(x, y, w, h) = cv2.boundingRect(c)</code><br>
<code>ar = w / float(h)</code><br>
<code>if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:</code><br>
<code>questionCnts.append(c)</code>
</p>
<p>
<code>cv2.boundingRect</code>:
<ul>
<li><strong>Description:</strong> Calculates the bounding rectangle of a contour.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>contour</code>: The contour to analyze.</li>
</ul>
</li>
<li><strong>Returns:</strong> A tuple (x, y, w, h) representing the bounding rectangle's top-left corner coordinates and dimensions.</li>
</ul>
</p>
</div>

<div class="explanation">
<h2>7. Sorting and Answer Processing</h2>
<p>
<code>questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]</code><br>
<code>correct = 0</code><br>
<code>for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):</code><br>
<code>cnts = contours.sort_contours(questionCnts[i:i + 5])[0]</code><br>
<code>bubbled = None</code><br>
<code>for (j, c) in enumerate(cnts):</code><br>
<code>mask = np.zeros(thresh.shape, dtype="uint8")</code><br>
<code>cv2.drawContours(mask, [c], -1, 255, -1)</code><br>
<code>mask = cv2.bitwise_and(thresh, thresh, mask=mask)</code><br>
<code>total = cv2.countNonZero(mask)</code><br>
<code>if bubbled is None or total > bubbled[0]:</code><br>
<code>bubbled = (total, j)</code>
</p>
<p>
<code>contours.sort_contours</code>:
<ul>
<li><strong>Description:</strong> Sorts contours based on a specified method.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>contours</code>: The list of contours to sort.</li>
<li><code>method</code>: The sorting method ("top-to-bottom" or "left-to-right").</li>
</ul>
</li>
<li><strong>Returns:</strong> A tuple containing the sorted contours.</li>
</ul>
<code>cv2.drawContours</code>:
<ul>
<li><strong>Description:</strong> Draws contours on an image.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>image</code>: The image on which to draw.</li>
<li><code>contours</code>: List of contours to draw.</li>
<li><code>contourIdx</code>: Index of the contour to draw (-1 draws all contours).</li>
<li><code>color</code>: Color of the contour (e.g., 255 for white).</li>
<li><code>thickness</code>: Thickness of the contour line.</li>
</ul>
</li>
<li><strong>Returns:</strong> None</li>
</ul>
<code>cv2.bitwise_and</code>:
<ul>
<li><strong>Description:</strong> Performs a bitwise AND operation on two images.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>src1</code>: First image.</li>
<li><code>src2</code>: Second image.</li>
<li><code>mask</code>: Optional mask to apply.</li>
</ul>
</li>
<li><strong>Returns:</strong> The result of the bitwise AND operation.</li>
</ul>
<code>cv2.countNonZero</code>:
<ul>
<li><strong>Description:</strong> Counts the number of non-zero pixels in an image.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>image</code>: The image to analyze.</li>
</ul>
</li>
<li><strong>Returns:</strong> The count of non-zero pixels.</li>
</ul>
</p>
</div>

<div class="explanation">
<h2>8. Result Display</h2>
<p>
<code>color = (0, 0, 255)</code><br>
<code>k = ANSWER_KEY[q]</code><br>
<code>if k == bubbled[1]:</code>
<br>
<code>color = (0, 255, 0)</code><br>
<code>correct += 1</code><br>
<code>cv2.drawContours(paper, [cnts[k]], -1, color, 3)</code><br>
<code>score = (correct / 5.0) * 100</code><br>
<code>cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)</code><br>
<code>cv2.imshow("Original", image)</code><br>
<code>cv2.imshow("Exam", paper)</code><br>
<code>cv2.waitKey(0)</code>
</p>
<p>
<code>cv2.putText</code>:
<ul>
<li><strong>Description:</strong> Draws text on an image.</li>
<li><strong>Parameters:</strong>
<ul>
<li><code>img</code>: The image on which to draw the text.</li>
<li><code>text</code>: The text to draw.</li>
<li><code>org</code>: The bottom-left corner of the text string in the image.</li>
<li><code>fontFace</code>: The font type (e.g., cv2.FONT_HERSHEY_SIMPLEX).</li>
<li><code>fontScale</code>: The scale factor of the font size.</li>
<li><code>color</code>: The color of the text.</li>
<li><code>thickness</code>: The thickness of the text stroke.</li>
</ul>
</li>
<li><strong>Returns:</strong> The image with text drawn on it.</li>
</ul>
</p>
</div>
