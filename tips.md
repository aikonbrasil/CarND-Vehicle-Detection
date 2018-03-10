UDACITY - Vehicle Detection - TIPs
==========================

in Spatial Binning of Color (Class 16 of Lesson 20).
 
- [ ] Ok, but 3072 elements is still quite a few features! Could you get away with even lower resolution? I'll leave that for you to explore later when you're training your classifier. [We can change the spatial resolution in project] —> in feature_vec()
- [ ] HOG Features
- [ ] sklearn —> StandardScaler() [Used to normalize] but [it is important to have your data in right format]  import bumpy as np feature_list = [ feature_1, feature_2, …] X = np.vstack(feature_list).astype(np.float64)  CHECK Class 22 of Lesson 20
- [ ] 22
- [ ] sklearn version issue: Check Class 28 of Lesson 20 (train_test_split)
- [ ] Warning: when dealing with image data that was extracted from video, you may be dealing with sequences of images where your target object (vehicles in this case) appear almost identical in a whole series of images. In such a case, even a randomized train-test split will be subject to overfitting because images in the training set may be nearly identical to images in the test set. For the subset of images used in the next several quizzes, this is not a problem, but to optimize your classifier for the project, you may need to worry about time-series of images!
- [ ] 
