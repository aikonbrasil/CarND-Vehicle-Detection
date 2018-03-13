## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/cars_classes_0.png
[image2]: ./writeup_images/generic_channel_HUG.png
[image3]: ./writeup_images/script_training_result.png
[image4]: ./writeup_images/sliding_window_0.png
[image5]: ./writeup_images/sliding_windows_1.png
[image6]: ./writeup_images/sliding_windows_2.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.    

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook called `project_solution.ipynb` inside the section called `Histogram of Oriented Gradients `(or the customized version of HOG in lines 54 through 55 of the file called `script.py`). I have re-used the class code with the function `extract_features(DATA_SET) `.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using an generic channel and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. In this case I used the function `get_hog_features()` that is defined in the file library called `own_functions.py` in the line 65:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. In this stage, I used also Udacity code provided in code. However parameter selection was done using a custom function called `tweak_function_hug_extracting_features()`. Extracting features using HOG is done using different color spaces, with specific orientation vectors number for each cell (with its pix_per_cell value), it is also considered a normalization in every block of subset image. The best configuration of parameters was: `color_space='YCrCb', spatial_size=(16, 16),hist_bins=32, orient=8,pix_per_cell=8, cell_per_block=2, hog_channel="ALL",hist_range=(0, 256)`. This values were defined in the optimized code in the file `own_function.py` since the line code number 8.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the 7th code cell of the IPython notebook called `project_solution.ipynb` there is a section called "Training the classifier with dataset of cars and non cars groups". First I defined the definitive extracting features parameters of function called `extract_features()` (check code line 8 of library `own_function.py`). In the next cell (labeled as training classifier) I created an array stack of feature vectors (car and non-cars), I defined the labels vector, I scaled and normalized using `standardScaler()` function. After that I split the dataset in randomized training sets and also I divided the dataset in dataset for training (X_train, X_test, y_train, and y_test).

The next important step  is related to classify using SVC.  Check the accuracy rate using the previous parameters as the final result of the classify train.

![alt text][image3]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search using sub-sample technique following the code provided in the class. So, in the IPython notebook called ` project_solution.ipynb` in the cell 9 there is a section dedicated to slide window. In this cell I defined a helper function called `getting_windows_info()`, this function uses the udacity class code `slide_window()` and `search_windows()`. In the script defined in the class `own_functions.py` since the line 81.

![alt text][image4]

The same sliding window search was optimized with the function `find_cars()`  code provided in the class (check code line 223 in `own_function.py` class function ). Check the result of this step.

![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
