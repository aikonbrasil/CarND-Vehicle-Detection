import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog


def extract_features(imgs, color_space='YCrCb', spatial_size=(16, 16),hist_bins=32, orient=8,pix_per_cell=8, cell_per_block=2, hog_channel="ALL",hist_range=(0, 256)):
    # Parameters extraction
    # HOG parameters
    features=[]
    cspace = color_space
    orient = orient
    pix_per_cell = pix_per_cell
    cell_per_block = cell_per_block
    hog_channel = hog_channel
    # Spatial parameters
    size = spatial_size
    # Histogram parameters
    hist_bins = hist_bins
    hist_range = hist_range
    hist_range = hist_range
    imaage = cv2.imread(imgs[0])
    imaage = cv2.cvtColor(imaage, cv2.COLOR_BGR2RGB)
    print(imaage.shape)
    print(imgs[0])
    for file in imgs:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if cspace != 'RGB':
            if cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],
        	 								orient, pix_per_cell, cell_per_block,
        	 								vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)

        spatial_features = bin_spatial(feature_image, size)
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    return features
    #return imaage


    # for file in imgs:
    #     image = cv2.imread(file)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     # apply color conversion if other than 'RGB'
    #     if cspace != 'RGB':
    #         if cspace == 'HSV':
    #             feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #         elif cspace == 'LUV':
    #             feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    #         elif cspace == 'HLS':
    #             feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    #         elif cspace == 'YUV':
    #             feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    #         elif cspace == 'YCrCb':
    #             feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    #     else: feature_image = np.copy(image)
    #
	# 	# Call get_hog_features() with vis=False, feature_vec=True
	# 	if hog_channel == 'ALL':
	# 		hog_features = []
	# 		for channel in range(feature_image.shape[2]):
	# 			hog_features.append(get_hog_features(feature_image[:,:,channel],
	# 								orient, pix_per_cell, cell_per_block,
	# 								vis=False, feature_vec=True))
	# 		hog_features = np.ravel(hog_features)
	# 	else:
	# 		hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
	# 					pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    #
	# 	# Apply bin_spatial() to get spatial color features
	# 	spatial_features = bin_spatial(feature_image, size)
    #
	# 	# Apply color_hist()
	# 	hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    #
	# 	features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # return features

# Basic functions provided on Udacity's course to extract features.

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

####### Search Windows functions
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)
