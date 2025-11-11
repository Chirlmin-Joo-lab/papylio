import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import math

def find_peaks(image=None, method='local-maximum', **kwargs):
    if method == 'AKAZE':
        coordinates = analyze(image, **kwargs)[2]
    elif method == 'absolute-threshold':
        coordinates = find_peaks_absolute_threshold(image, **kwargs)
    elif method == 'adaptive-threshold':
        coordinates = find_peaks_adaptive_threshold(image, **kwargs)
    elif method == 'local-maximum':
        coordinates = find_peaks_local_maximum(image, **kwargs)
    elif method == 'local-maximum-auto':
        coordinates = find_peaks_local_maximum_auto(image, **kwargs)
    elif method == 'relative-local-maximum':
        coordinates = find_peaks_relative_local_maximum(image, **kwargs)
    else:
        raise ValueError(f'Unkown method {method}.')
    return coordinates

def find_peaks_absolute_threshold(image, threshold = None, minimum_area = 5, maximum_area = 15):
    if threshold is None: threshold = (np.max(image) + np.min(image)) / 2
    image_thresholded = image > threshold
    coordinates = coordinates_from_contours(image_thresholded, minimum_area, maximum_area)
    return coordinates

def find_peaks_adaptive_threshold(image, minimum_area = 5, maximum_area = 15):
    # This may be needed if we go from a 16-bit image to an 8 bit image
    # if bounds is None:
    #     lower_bound = np.min(image)
    #     upper_bound = np.percentile(image.flatten(), 99.999)
    # else:
    #     lower_bound = bounds[0]
    #     upper_bound = bounds[1]
    #
    # # Change threshold and image to 8-bit, as cv2 can only analyse 8-bit images
    # # threshold = ((threshold - lower_bound) / (upper_bound - lower_bound) * 255)
    # # threshold = np.clip(threshold, 0, 255).astype('uint8')
    # image = ((image - lower_bound) / (upper_bound - lower_bound) * 255)
    # image = np.clip(image, 0, 255).astype('uint8')

    image_thresholded = cv2.adaptiveThreshold(image.astype(np.uint8),
                                              maxValue=1,
                                              adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              thresholdType=cv2.THRESH_BINARY,
                                              blockSize=9,
                                              C=0)
    coordinates = coordinates_from_contours(image_thresholded, minimum_area, maximum_area)

    return coordinates

def find_peaks_local_maximum(image,
                             minimum_intensity_difference=25,
                             maximum_intensity_difference=math.inf,
                             filter_neighbourhood_size_min=10,
                             filter_neighbourhood_size_max=5):

    """
    Find peaks based on local maxima.

    Parameters
    ----------
    image : NxM numpy.ndarray

    minimum_intensity_difference : int or float
        Lower threshold on the intensity difference between local minimum and local maximu
    maximum_intensity_difference : int or float, optional
        Upper threshold on the intensity difference between local minimum and local maximum.
    filter_neighbourhood_size_min : int
        Size of the minimum filter.
    filter_neighbourhood_size_max : int
        Size of the maximum filter.

    Returns
    -------
    Nx2 numpy.ndarray
        Peak coordinates

    Notes
    -----
    First minimum filtered and maximum filtered images are created from the input image.
    Local maxima are determined by finding the pixels where the maximum filtered image is equal to the input image.
    Local maxima where the intensity difference between the local maximum and local minimum is outside the interval
    (minimum_intensity_difference, maximum_intensity_difference) are discarded.
    """

    # Perhaps make filter round?
    image_max = filters.maximum_filter(image, filter_neighbourhood_size_max)
    maxima = (image == image_max)
    image_min = filters.minimum_filter(image, filter_neighbourhood_size_min)
    # Probably I need to make the neighbourhood_size of the minimum filter larger.

    difference_above_minimum = ((image_max - image_min) > minimum_intensity_difference)
    difference_below_maximum = ((image_max - image_min) < maximum_intensity_difference)
    difference_within_bounds = np.logical_and(difference_above_minimum, difference_below_maximum)
    maxima[difference_within_bounds == 0] = 0

    coordinates = np.fliplr(np.vstack(np.where(maxima)).T)

    # labeled, num_objects = ndimage.label(maxima)
    # if num_objects > 0:
    #     coordinates = np.fliplr(np.array(ndimage.center_of_mass(image, labeled, range(1, num_objects + 1))))
    # else:
    #     coordinates = np.array([])
    #     print('No peaks found')

    return coordinates



def find_peaks_local_maximum_auto(image,
                                  fraction_difference = 0.5,
                                  filter_neighbourhood_size_min=10,
                                  filter_neighbourhood_size_max=5):

    """
    Find peaks based on local maxima.

    Parameters
    ----------
    image : NxM numpy.ndarray

    minimum_intensity_difference : int or float
        Lower threshold on the intensity difference between local minimum and local maximu
    maximum_intensity_difference : int or float, optional
        Upper threshold on the intensity difference between local minimum and local maximum.
    filter_neighbourhood_size_min : int
        Size of the minimum filter.
    filter_neighbourhood_size_max : int
        Size of the maximum filter.

    Returns
    -------
    Nx2 numpy.ndarray
        Peak coordinates

    Notes
    -----
    First minimum filtered and maximum filtered images are created from the input image.
    Local maxima are determined by finding the pixels where the maximum filtered image is equal to the input image.
    Local maxima where the intensity difference between the local maximum and local minimum is outside the interval
    (minimum_intensity_difference, maximum_intensity_difference) are discarded.
    """

    # Perhaps make filter round?
    image_max = filters.maximum_filter(image, filter_neighbourhood_size_max)
    maxima = (image == image_max)
    image_min = filters.minimum_filter(image, filter_neighbourhood_size_min)
    # Probably I need to make the neighbourhood_size of the minimum filter larger.

    maxima_values = image_max[maxima]


    maxima_values = maxima_values[maxima_values>np.max(image_min)+np.std(image_min)*5]

    minimum_intensity_difference = (np.median(maxima_values) - np.median(image_min.flatten())) * fraction_difference

    difference_above_minimum = ((image_max - image_min) > minimum_intensity_difference)
    # difference_below_maximum = ((image_max - image_min) < maximum_intensity_difference)
    # difference_within_bounds = np.logical_and(difference_above_minimum, difference_below_maximum)
    difference_within_bounds = difference_above_minimum
    maxima[difference_within_bounds == 0] = 0

    coordinates = np.fliplr(np.vstack(np.where(maxima)).T)

    # labeled, num_objects = ndimage.label(maxima)
    # if num_objects > 0:
    #     coordinates = np.fliplr(np.array(ndimage.center_of_mass(image, labeled, range(1, num_objects + 1))))
    # else:
    #     coordinates = np.array([])
    #     print('No peaks found')

    return coordinates


def find_peaks_relative_local_maximum(image,
                                      minimum_times_background=4.5,
                                      maximum_times_background=math.inf,
                                      filter_neighbourhood_size_min=10,
                                      filter_sigma_min=25,
                                      filter_neighbourhood_size_max=5):

    image_min = filters.minimum_filter(image, filter_neighbourhood_size_min)
    image_min_gaussian = filters.gaussian_filter(image_min, sigma=filter_sigma_min)
    relative_image = image/image_min_gaussian

    relative_image_max = filters.maximum_filter(relative_image, filter_neighbourhood_size_max)
    maxima = (relative_image == relative_image_max)
    maxima[relative_image_max < minimum_times_background] = False

    labeled, num_objects = ndimage.label(maxima)
    if num_objects > 0:
        coordinates = np.fliplr(np.array(ndimage.center_of_mass(image, labeled, range(1, num_objects + 1))))
    else:
        coordinates = np.array([])
        print('No peaks found')

    return coordinates

def coordinates_from_contours(image_thresholded, minimum_area=5, maximum_area=15):
    contours, hierarchy = cv2.findContours(image_thresholded.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    x = []
    y = []

    # colorImg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    coordinates = []

    for c in contours:
        # Calculate moments for each contour
        M = cv2.moments(c)

        # Calculate x, y coordinate of center

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            area = M["m00"]  # Is the same as cv2.contourArea(c) # Is the same as M["m00"]

            if (area > minimum_area) and (area < maximum_area):
                x = np.append(x, cX)
                y = np.append(y, cY)
                coordinates.append(np.array([cX, cY]))
        else:
            cX, cY = 0, 0

        # cv2.circle(colorImg, (cX, cY), 8, (0, 0, 255), thickness=1)

    return np.array(coordinates)


# # -*- coding: utf-8 -*-
# """
# Created on Wed Apr 17 13:50:42 2019
#
# @author: https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
#
# returns the number of spots, and per spot its centroid and the number of pixels (can be used to discard too large spots)
# """
# import cv2
# import bisect #This module provides support for maintaining a list in sorted order without having to sort the list after each insertion.
# import numpy as np
# import matplotlib.pyplot as plt
#
# from movie.background_correction import get_threshold
#
# def analyze(src, threshold=None):
#     if not threshold:
#         fL = get_threshold(src)
#     else:
#         fL = threshold
#     # gray1 = enhance_blobies_single(src, fL, 1)  # remove_background(src, fL)
#     gray1 = src
#     detector = cv2.AKAZE_create()
#     (kps1, descs1) = detector.detectAndCompute(gray1, None)
#     ctrd = np.array(cv2.KeyPoint_convert(kps1))
#
#     # remove all pixels at the edge (within 10 pix)
#     num_labels = len(ctrd)
#     dim1, dim0 = np.shape(src)
#     for ii in range(num_labels - 1, -1, -1):
#         discard = ctrd[ii, 0] < 10 or ctrd[ii, 1] < 10 or ctrd[ii, 0] > dim0 - 10 or ctrd[ii, 1] > dim1 - 10 or src[
#             int(ctrd[ii, 1]), int(ctrd[ii, 0])] == 0  # or size_label[ii]>100
#         # for some reason also spots are found on immean20 with no intensity --> discard
#         if discard:
#             ctrd = np.delete(ctrd, ii, axis=0)
#     #         size_label=np.delete(size_label,ii, axis=0)
#     num_labels = len(ctrd)
#
#     return num_labels, 0, ctrd
#
#
# def imadjust(src, tol=1, vout=(0, 255)):
#     # src : input one-layer image (numpy array)
#     # tol : tolerance, from 0 to 100.
#     # vin  : src image bounds
#     # vout : dst image bounds
#     # return : output img
#
#     assert len(src.shape) == 2, 'Input image should be 2-dims'
#
#     tol = max(0, min(100, tol))
#
#     vin = [np.min(src), np.max(src)]
#     vout = [0, 65535]  # 65535=16 bits
#     if tol > 0:
#         # Compute in and out limits
#         # Histogram
#         hist = np.histogram(src, bins=list(range(vin[1] - vin[0])), range=tuple(vin))[0]
#
#         # Cumulative histogram
#         cum = hist.copy()
#         for i in range(0, vin[1] - vin[0] - 1): cum[i] = cum[i - 1] + hist[i]  # why not hist.cumsum() here?
#
#         # Compute bounds
#         total = src.shape[0] * src.shape[1]
#         low_bound = total * tol / 100
#         upp_bound = total * (100 - tol) / 100
#         vin[0] = bisect.bisect_left(cum, low_bound)
#         vin[1] = bisect.bisect_left(cum, upp_bound)
#
#     # Stretching
#     scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
#     vs = src - vin[0]
#     vs[src < vin[0]] = 0  # everything below zero becomes 0
#     vd = vs * scale + 0.5 + vout[0]  # why +0.5?
#     vd[vd > vout[1]] = vout[1]
#     dst = vd
#
#     return dst.astype(np.uint16)
#
#
# def im_binarize(img, f):
#     temp = img.copy()
#     temp[temp < f] = 0
#     return temp.astype(np.uint8)
#
#
# def enhance_blobies(image, f):
#     l, r = image[:, :image.shape[1] // 2], image[:, image.shape[1] // 2:]
#     l_adj, r_adj = imadjust(l.copy()), imadjust(r.copy())
#     l_bin, r_bin = im_binarize(l_adj, f).astype(np.uint8), im_binarize(r_adj, f).astype(np.uint8)
#     return l, r, l_bin, r_bin
#
# detector = cv2.AKAZE_create()
#
# (kps1, descs1) = detector.detectAndCompute(gray1, None);
# (kps2, descs2) = detector.detectAndCompute(gray2, None);
#
# print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
# print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))
#
# # Match the features
# bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# matches = bf.knnMatch(descs1, descs2, k=2)  # typo fixed
#
# # Apply ratio test
# pts1, pts2 = [], []
# for m in matches:
#     pts1.append(kps1[m[0].queryIdx].pt)
#     pts2.append(kps2[m[0].trainIdx].pt)
#
# pts1 = np.array(pts1).astype(np.float32)  # xy position
# pts2 = np.array(pts2).astype(np.float32)
# # AA=cv2.KeyPoint_convert(kps1);
#
# transformation_matrix, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 20)
#


if __name__ == '__main__':
    from papylio import Experiment
    exp = Experiment(r'D:\ivoseverins\SURFdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\20141017 - Holliday junction - Copy\HJC-50pM')
    file = exp.files[2]
    movie = file.movie
    image = movie.make_average_image(write=False)

    coordinates = find_peaks(image=image, method='adaptive-threshold', minimum_area=5, maximum_area=15)

    plt.imshow(image)
    plt.scatter(coordinates[:,0],coordinates[:,1],color='r')

