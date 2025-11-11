# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:10:42 2021

input = image, coordinates, method, filter_neighbourhood_size, radius
methods+args:
    Channel_mean
    Channel_median
    ROI_minimum, uses filter_neighbourhood_size ** DEFAULT
    ROI_median, uses filter_neighbourhood_size
    Surrounding_mean, uses radius
    Surrounding_median, uses radius
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters


def extract_background(image, coordinates, method='ROI_minimum', filter_neighbourhood_size=10, radius=5):
    background = np.zeros(coordinates.shape[0])

    if method == 'channel_mean':
        background[:] = np.mean(image)

    elif method == 'channel_median':
        background[:] = np.median(image)

    elif method == 'ROI_minimum':
        image_min = filters.minimum_filter(image, filter_neighbourhood_size)
        # TODO: improve edge between donor&acceptor channel
        # plt.imshow(image_min)
        for i, (x, y) in enumerate(coordinates):
            background[i] = image_min[int(y), int(x)]

    elif method == 'ROI_median':
        image_median = filters.median_filter(image, filter_neighbourhood_size)
        # TODO: improve edge between donor&acceptor channel
        for i, (x, y) in enumerate(coordinates):
            background[i] = image_median[int(y), int(x)]

    elif method == 'surrounding_mean':  # not optimal when multiple spots are close, better to use median
        from papylio.coordinate_optimization import crop, circle
        circle_matrix = circle(radius)
        for i, c in coordinates:
            cropped_peak = crop(image, c, radius * 2 + 1)
            circle_image = cropped_peak * circle_matrix
            background[i] = np.mean(circle_image[np.nonzero(circle_image)])

    elif method == 'surrounding_median':  # similar to coordinate_optimization, coordinates_without_intensity_at_radius
        from papylio.coordinate_optimization import crop, circle
        circle_matrix = circle(radius)
        for i, c in coordinates:
            cropped_peak = crop(image, c, radius * 2 + 1)
            circle_image = cropped_peak * circle_matrix
            background[i] = np.median(circle_image[np.nonzero(circle_image)])

    else:
        raise ValueError('Unknown method')

    return background
