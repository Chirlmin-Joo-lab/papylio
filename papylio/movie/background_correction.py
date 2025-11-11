import numpy as np
import scipy.optimize
import scipy.ndimage
import tqdm

from papylio.movie.shading_correction import get_photobleach

def determine_temporal_background_correction(frames, method, flatfield=None, darkfield=None):
    if method == 'BaSiC':
        channel_dimensions = frames.shape[-2:][::-1]  # size should be given in (x,y) for get_photobleach
        size = (channel_dimensions / np.max(channel_dimensions) * 256).astype(int)
        correction = get_photobleach(frames, flatfield, darkfield, size=size).flatten()
    elif method == 'BaSiC_crop':
        # Crop the image instead of resizing, may be better for single-molecules. However, may give higher noise.
        raise NotImplementedError('')
    else:
        correction = np.array([determine_single_value_background_correction(frame, method, flatfield, darkfield)
                               for frame in tqdm.tqdm(frames)])

    return correction


def determine_single_value_background_correction(frame, method, flatfield=None, darkfield=None):
    frame_corrected = frame
    if darkfield is not None:
        frame_corrected = frame - darkfield
    if flatfield is not None:
        frame_corrected = frame_corrected / flatfield

    if method == 'BaSiC':
        channel_dimensions = frame.shape[-2:][::-1]  # size should be given in (x,y) for get_photobleach
        size = (channel_dimensions / np.max(channel_dimensions) * 256).astype(int)
        correction = get_photobleach(frame, flatfield, darkfield, size=size).flatten()
    elif method == 'BaSiC_crop':
        # Crop the image instead of resizing, may be better for single-molecules. However, may give higher noise.
        raise NotImplementedError('')
    elif method == 'mean':
        correction = frame_corrected.mean()
    elif method == 'median':
        correction = np.median(frame_corrected)
    elif method == 'fit_background_peak':
        correction = gaussian_maximum_fit(frame_corrected)
    elif 'filter' in method:
        correction = determine_spatial_background_correction(frame_corrected, method).mean()  # TODO: take mean only over x, y not over channel
    else:
        raise ValueError(f'Method {method} not found')

    return correction


def determine_spatial_background_correction(frame, method, flatfield=None, darkfield=None, **kwargs):
    # TODO: improve edge between donor&acceptor channel
    if method == 'gaussian_filter':
        scipy_filter = scipy.ndimage.gaussian_filter
        scipy_filter_kwargs = {'sigma': 0.5, 'mode': 'wrap'}
        # This comes down to taking the mean of the corrected image
    elif method == 'minimum_filter':
        scipy_filter = scipy.ndimage.minimum_filter
        scipy_filter_kwargs = {'size': 15, 'mode': 'wrap'}
    elif method == 'median_filter':
        scipy_filter = scipy.ndimage.median_filter
        scipy_filter_kwargs = {'size': 30, 'mode': 'wrap'}
    else:
        raise ValueError(f'Method {method} not found')

    if darkfield is not None:
        frame = frame - darkfield
    if flatfield is not None:
        frame = frame / flatfield

    scipy_filter_kwargs.update(kwargs)

    correction = scipy_filter(frame, **scipy_filter_kwargs)

    return correction

def gauss_function(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gaussian_maximum_fit(frame, width_around_peak_fitted=200):
    # count, edges = np.histogram(frame.flatten(), bins=width_around_peak_fitted)
    # max_bin_center = edges[count.argmax():count.argmax()+2].mean()

    frame_min = np.floor(frame.min()).astype(int)
    frame_max = int(np.quantile(frame, 0.999))  # np.ceil(frame.max()).astype(int)

    frame_flattened = frame.flatten()

    # factor = 10
    # number_of_bins = int((frame_max - frame_min + 1)/factor)
    number_of_bins = 500
    factor = (frame_max - frame_min + 1)/number_of_bins

    count, edges = np.histogram(frame_flattened, bins=number_of_bins, range=(frame_min - 0.5, frame_max + 0.5))

    bincenters = (edges[:-1] + edges[1:]) / 2
    max_bin_center = bincenters[count.argmax()]

    width = width_around_peak_fitted

    x_peak = None
    while x_peak is None:
        selection = np.vstack(
            [max_bin_center - width / 2 < bincenters, bincenters < max_bin_center + width / 2]).all(axis=0)
        x = bincenters[selection]
        y = count[selection]

        std = np.std(frame_flattened[(frame_flattened > x.min()) & (frame_flattened < x.max())])
        try:
            popt, pcov = scipy.optimize.curve_fit(gauss_function, x, y, p0=[count.max(), max_bin_center, std], maxfev=2000)
            x_peak = popt[1]
        except RuntimeError:
            # x_peak = gaussian_maximum_fit(frame, width_around_peak_fitted*2)
            width *= 2

    # plt.figure();
    # plt.hist(frame_flattened, bins=number_of_bins, range=(frame_min - 0.5, frame_max + 0.5))
    # plt.plot(x, gauss_function(x, *popt))
    return x_peak






 #     if method == 'rollingball':
    #         background = rollingball(image, self.width_pixels / 10)[1]  # this one is not used in pick_spots_akaze
    #         image_correct = image - background
    #         image_correct[image_correct < 0] = 0
    #         threshold = get_threshold(image_correct)
    #         return remove_background(image_correct, threshold)
    #     elif method == 'per_channel':  # maybe there is a better name
    #         sh = np.shape(image)
    #         threshold_donor = get_threshold(self.get_channel(image, 'donor'))
    #         threshold_acceptor = get_threshold(self.get_channel(image, 'acceptor'))
    #         background = np.zeros(np.shape(image))
    #         background[:, 0:sh[0] // 2] = threshold_donor
    #         background[:, sh[0] // 2:] = threshold_acceptor
    #         return remove_background(image, background)
    #
    #     # note: optionally a fixed threshold can be set, like with IDL
    #     # note 2: do we need a different threshold for donor and acceptor?






































# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:16:25 2019

@author: mwdocter
this code takes an images, calculates the rolling_ball background, and subtracts it
"""
#http://imagejdocu.tudor.lu/doku.php?id=gui:process:subtract_background
#Based on the a „rolling ball“ algorithm described in Stanley Sternberg's article, „Biomedical Image Processing“, IEEE Computer, January 1983. 
import scipy.ndimage as scim
import skimage
#from skimage.morphology import ball
import matplotlib.pyplot as plt

def rollingball(*args): # Matlab[im_out,im_bg]=rollingball(im_in,size_ball,im_bg)
    varargin = args
    im_in=varargin[0]
    if len(varargin)==1:
        size_ball=30
    else:
        size_ball=varargin[1]
    if len(varargin)<3:
        # from https://stackoverflow.com/questions/29320954/rolling-ball-background-subtraction-algorithm-for-opencv
        # Create 3D ball structure
        s = skimage.morphology.ball(size_ball)
        # Take only the upper half of the ball
        h = int((s.shape[1] + 1) / 2)
        # Flat the 3D ball to a weighted 2D disc
        s = s[:h, :, :].sum(axis=0)
        # Rescale weights into 0-255
        s = (255 * (s - s.min())) / (s.max()- s.min())
        ss=s[2*h//4:2*3*h//4,2*h//4:2*3*h//4]
        ss = (255 * (ss - ss.min())) / (ss.max()- ss.min())
       
        #im_bg=scim.grey_closing(im_in,structure=ss)
        im_bg=skimage.morphology.opening(im_in,ss)
        #im_out = scim.white_tophat(im, structure=s)
    else:
        im_bg=varargin[2]
        
    im_out=im_in-im_bg #note match 3s dimension im_bg to im_in
    im_out[im_out<0]=0
    
    return im_out, im_bg
    # Use im-opening(im,ball) (i.e. white tophat transform) (see original publication)
    

# def get_threshold(image_stack, show=0):
#     ydata = (np.sort(image_stack.ravel()))
#     ydata_original = ydata
#     xdata = np.array(range(0, len(ydata)))
#     # scale the data to make x and y evenly important
#     ymaxALL = float(max(ydata))
#     xmaxALL = float(max(xdata))
#     ydata = ydata * xmaxALL / ymaxALL  # don't forget this is scaled
#
#     # fit a line through the lowest half of x
#     xd = xdata[:int(np.floor(len(xdata) / 2))]
#     yd = ydata[:int(np.floor(len(xdata) / 2))]
#     p_start = np.polyfit(xd, yd, 1)
#
#     # fit a line through the upper half of y
#     ymax = max(ydata)
#     yhalf = ymax / 2
#     x2 = np.argwhere(abs(ydata - yhalf) == min(abs(ydata - yhalf)))
#     x2 = int(x2[0])
#     xd = xdata[x2:]
#     yd = ydata[x2:]
#     p_end = np.polyfit(xd, yd, 1)
#
#     # find the crossing of these lines
#     # a1*x+b1=a2*x+b2
#     # (a1-a2)*x=b2-b1
#     # x=(b2-b1)/(a1-a2)
#     x_cross = int((p_end[1] - p_start[1]) / (p_start[0] - p_end[0]))
#     y_cross = int(np.polyval(p_start, x_cross))
#
#     # add polyfits to the plot
#     y_fit_start = np.polyval(p_start, xdata[:x_cross])
#     x_fit_end = xdata[x_cross:]  # start to draw from crossing y=0
#     y_fit_end = np.polyval(p_end, x_fit_end)
#     # now find the closest distance from cross to actual data. x and y should be simarly scaled
#     xx = xdata - x_cross
#     xx = [float(ii) for ii in xx]
#     yy = ydata - y_cross
#     yy = [float(ii) for ii in yy]
#     rr = (np.array(xx) ** 2 + np.array(yy) ** 2) ** 0.5  # int32 is not large enough
#     x_found = np.argwhere(min(rr) == rr)
#     x_found = x_found[0, 0]
#     if show:
#         plt.figure()
#         fig2 = plt.subplot(1, 2, 2)
#         fig2.plot(xdata, rr * ymaxALL / xmaxALL)
#         # fig2.title("{:s}".format(x_found))
#
#         fig1 = plt.subplot(1, 2, 1)
#         fig1.plot(xdata, ydata * ymaxALL / xmaxALL, 'b')
#         fig1.plot(xdata[:x_cross], y_fit_start[:x_cross] * ymaxALL / xmaxALL, 'g')
#         fig1.plot(x_fit_end, y_fit_end * ymaxALL / xmaxALL, 'r')
#         fig1.plot(x_cross, y_cross * ymaxALL / xmaxALL, 'kx')
#
#         fig1.plot(x_found, ydata[x_found] * ymaxALL / xmaxALL, 'mo')
#
#         fig1.plot(xdata[:-1], ydata_original[1:] - ydata_original[:-1])
#         plt.show()
#
#     thr = ydata[x_found] * ymaxALL / xmaxALL
#     im_uit = image_stack - thr.astype(type(image_stack[0, 0]))
#     im_uit[im_uit < 0] = 0
#     return thr
#
#
# def remove_background(image_stack, thr, show=0):
#     im_uit = image_stack - thr.astype(type(image_stack[0, 0]))
#     im_uit[im_uit < 0] = 0
#     return im_uit
#
#