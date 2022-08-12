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