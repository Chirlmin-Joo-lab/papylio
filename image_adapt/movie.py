# -*- coding: utf-8 -*-
"""
Created on Jun 2019
@author: carlos

some features are shared within an image collection, like the file name, mapping, background, threshold, ...)
so imagecollection is a class ;)

warning blackboax number: (size+fwhm) gauss for extracting donor&acceptor
"""


from .load_file import read_one_page#_pma, read_one_page_tif
from .load_file import read_header
from .rolling_ball import rollingball

from .find_threshold import remove_background
from .find_threshold import get_threshold
import matplotlib.pyplot as plt
import numpy as np
import tifffile as TIFF
#from cached_property import cached_property
from .Mapping import Mapping
from .Image import Image
from .analyze_label import analyze # note analyze label is differently from the approach in pick spots
#import cv2
import os
from ..find_xy_position.Gaussian import makeGaussian
import time
from .polywarp import polywarp, polywarp_apply
import cv2
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
 
class Movie():
    def __init__(self, filepath):#, **kwargs):
        self.filepath = filepath
        self._average_image = None
       
        if self.filepath.suffix == '.sifx':
            self.writepath = self.filepath.parent.parent
            self.name = self.filepath.parent.name
        else:
            self.writepath = self.filepath.parent
            self.name = self.filepath.with_suffix('').name
        
        self.read_header()
        
        #self.set_background_and_transformation() # Carlos: isn't this double, you call set_background twice, Margreet: this is the bg of the image, not the tetra_image. Same formulas though
        
        #Ivo: This is a good idea I think, but I think we should put this option in the method that produces the pks file.
        #self.pks_fn=kwargs.get('pks_fn',filepath) # default pks_fn=filepath, but you could give in a different name; ImageCollection(name, name, pks_fn='   ')
        
        #Ivo: same here
        #self.choice_channel=kwargs.get('choice_channel','d') #default 'd', 'd' for donor, 'a' for acceptor, 'da' for the sum
        
        #Ivo: what are these?
        #self.generic_map=kwargs.get('generic',0)
        #self.ii=np.array(range(1024*1024))
    
        #Ivo: Commented this because I would like to be able to instantiate the object without doing this initially
        #self.mapping = Mapping(tetra_fn,generic=self.generic_map)
#        (self.background,
#         self.pts_number,
#         self.dstG,
#         self.ptsG,
#         self.im_mean20_correct,
#         self.Gauss,
#         self.ptsG2) = self.set_background_and_transformation()
#        self.show_selected_spots()
 
#    @cached_property
#    def read_one_page(self):
#        return read_one_page ##normally this funciton needs two inputs, why not here?
#        if '.pma' in self.filepath:
#            return read_one_page_pma
#        elif '.tif' in self.filepath:
#            return read_one_page_tif

    def __repr__(self):
        return(f'{self.__class__.__name__}({str(self.filepath)})')
        
    @property
    def average_image(self):
        if self._average_image is None: self.make_average_tif()
        return self._average_image

    def read_header(self):
        self.width_pixels, self.height_pixels, self.number_of_frames, self.movie_file_object = read_header(self.filepath)

    def get_channel(self, image, channel = 'd'):
        sh = np.shape(image)
        if channel in ['d', 'donor']:
            return image[:,:sh[0]//2]
        elif channel in ['a','acceptor']:
            return image[:,sh[0]//2:]
        
    def saveas_tif(self):
        tif_filepath = self.writepath.joinpath(self.name+'.tif')
        for i in range(self.number_of_frames):
            
            #frame = self.get_image(ii).image
            frame = read_one_page(self.filepath, pageNb=i, A = self.movie_file_object)
            print(i)
            #naam=r'M:\tnw\bn\cmj\Shared\margreet\Cy3 G50\ModifiedData\Python'+'{:03d}'.format(ii)+'.tif'
            TIFF.imwrite(tif_filepath, np.uint16(frame))
    

    def make_average_tif(self, number_of_frames = 20, write = False):
        frame_list = [(read_one_page(self.filepath, pageNb=i, A=self.movie_file_object)).astype(float) 
                        for i in range(np.min([self.number_of_frames, number_of_frames]))]
        frame_array = np.dstack(frame_list)
        frame_array_mean = np.mean(frame_array, axis=2).astype(int)
        
        if write:
            tif_filepath = self.writepath.joinpath(self.name+'_ave.tif')
            TIFF.imwrite(tif_filepath, np.uint16(frame_array_mean))
        
        self._average_image = frame_array_mean
        
        return frame_array_mean
    
    def show_average_tif(self, mode='2d'):
        plt.figure()
        if mode == '2d':
            plt.imshow(self.average_image)
        if mode == '3d':
            from matplotlib import cm
            ax = plt.gca(projection='3d')
            X = np.arange(self.average_image.shape[1])
            Y = np.arange(self.average_image.shape[0])
            X, Y = np.meshgrid(X, Y)
            ax.plot_surface(X,Y,self.average_image, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
        plt.show()
    
    
    def subtract_background(self, image, method = 'per_channel'):
        if method == 'rollingball':
            background = rollingball(image,self.width_pixels/10)[1] # this one is not used in pick_spots_akaze
            image_correct = image - background
            image_correct[image_correct < 0] = 0       
            threshold = get_threshold(image_correct)
            return remove_background(image_correct,threshold)  
        elif method == 'per_channel': #maybe there is a better name
            sh=np.shape(image)
            threshold_donor = get_threshold(self.get_channel(image,'donor'))
            threshold_acceptor = get_threshold(self.get_channel(image, 'acceptor'))
            background = np.zeros(np.shape(image))
            background[:,0:sh[0]//2]=threshold_donor
            background[:,sh[0]//2:]=threshold_acceptor
            return remove_background(image,background)
        
        # note: optionally a fixed threshold can be set, like with IDL
        # note 2: do we need a different threshold for donor and acceptor?


    def find_peaks(self, image = None, method = 'AKAZE', threshold = 100):
        if image is None: image = self.average_image
        
        if method == 'AKAZE':
            coordinates = image_adapt.analyze_label.analyze(image)[2]
        elif method == 'threshold':
            image = ((image-450)/850*255).astype('uint8')
            ret,image_thresholded = cv2.threshold(image,threshold,255,cv2.THRESH_BINARY)
               
            im2, contours, hierarchy = cv2.findContours(image_thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            x=[]
            y=[]
        
            colorImg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            coordinates = []
        
            for c in contours:
                # calculate moments for each contour
                M = cv2.moments(c)
               
                # calculate x,y coordinate of center
            
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    x = np.append(x,cX)
                    y = np.append(y,cY)
                else:
                    cX, cY = 0, 0
        
                cv2.circle(colorImg, (cX, cY), 8, (0, 0, 255), thickness=1)
        
                coordinates.append(np.array([cX,cY]))
            
            coordinates = np.array(coordinates)

        
        elif method == 'local-maximum':
            neighborhood_size = 10
                
            image_max = filters.maximum_filter(image, neighborhood_size)
            maxima = (image == image_max)
            image_min = filters.minimum_filter(image, neighborhood_size)
            diff = ((image_max - image_min) > threshold)
            maxima[diff == 0] = 0
            
            labeled, num_objects = ndimage.label(maxima)
            coordinates = np.fliplr(np.array(ndimage.center_of_mass(image, labeled, range(1, num_objects+1))))
            
        return coordinates
    
    def show_coordinates(self, image, coordinates, **kwargs):
        plt.figure()
#        sorted_intensities = np.sort(image)
#        vmin = np.percentile(sorted_intensities, 5)
#        vmax = np.percentile(sorted_intensities, 99)
        plt.imshow(image, **kwargs)
        plt.scatter(coordinates[:,0],coordinates[:,1], marker = 'o', facecolors='none', edgecolors='r')
        #plt.show()
        
        plt.savefig(self.writepath.joinpath(self.name+'_ave_circles.png'), dpi=600)

    def is_within_margin(self, coordinates, 
                      edge = np.array([[0,1024],[0,2048]]), 
                      margin = 10):
        
        criteria = np.array([(coordinates[:,0] > edge[0,0] + margin),
                             (coordinates[:,0] < edge[0,1] - margin),
                             (coordinates[:,1] > edge[1,0] + margin),
                             (coordinates[:,1] < edge[1,1] - margin) 
        ])
        
        return criteria.all(axis=0)

    
    def get_acceptor_coordinates(self, donor_coordinates):
        acceptor_coordinates = polywarp_apply(self.mapping.P,self.mapping.Q,donor_coordinates)
        return acceptor_coordinates
    
    def get_donor_coordinates(self, acceptor_coordinates):
        donor_coordinates = polywarp_apply(self.mapping.P21,self.mapping.Q21,acceptor_coordinates)
        return donor_coordinates
    
    def write_coordinates_to_pks_file(self, coordinates):
        pks_filepath = self.writepath.joinpath(self.name+'.pks')
        with pks_filepath.open('w') as outfile:
            for i, coordinate in enumerate(coordinates):
                outfile.write(' {0:4.0f} {1:4.4f} {2:4.4f} {3:4.4f} {4:4.4f} \n'.format(i, coordinate[0], coordinate[1], 0, 0, width4=4, width6=6))
    
    def generate_pks_file(self, channel):
        
        image_mean = self.make_average_tif(number_of_frames=20)
        
        image_mean_corrected = self.subtract_background(image_mean, method = 'per_channel')
        
        
        if channel == 'd':
            donor_coordinates = self.find_peaks(self.get_channel(image_mean_corrected, 'donor'))
            acceptor_coordinates = self.get_acceptor_coordinates(donor_coordinates)
        
        elif channel == 'a':
            acceptor_coordinates = self.find_peaks(self.get_channel(image_mean_corrected, 'acceptor'))
            donor_coordinates = self.get_donor_coordinates(acceptor_coordinates)
          
        elif channel == 'da':
            print('I have no clue yet how to do this')
            # most likely they do not overlap before finding transformation, so what is the point of doing D+A?
            #pts_number, label_size, ptsG = analyze_label.analyze(im_mean20_correctA[:, 0:self.height_pixels//2+im_mean20_correctA[:, self.height_pixels//2:]])       

            # Ivo: I think they always transform the entire image of the acceptor channel using the mapping file, 
            # so that they do overlap. From what I can see in IDL at least.

        else:
            print('make up your mind, choose wisely d/a/da')    


        # Discard point close to edge image
        donor_edge = np.array([[0,self.width_pixels//2],[0,self.height_pixels]])
        acceptor_edge = np.array([[self.width_pixels//2,self.width_pixels],[0,self.height_pixels]])
        margin = 10
                
        both_within_margin = (self.is_within_margin(donor_coordinates, donor_edge, margin) & 
                                  self.is_within_margin(acceptor_coordinates, acceptor_edge, margin) )
        
        donor_coordinates = donor_coordinates[both_within_margin]
        acceptor_coordinates = acceptor_coordinates[both_within_margin]
        
        all_coordinates = np.array([donor_coordinates,acceptor_coordinates])
        s = all_coordinates.shape
        all_coordinates = np.reshape(all_coordinates.T,(s[0],s[1]*s[2])).T
        
        self.write_coordinates_to_pks_file(all_coordinates)
        
    

#    def set_background_and_transformation(self):
#        """
#        sum 20 image, find spots& background. Then loop over all image, do background subtract+ extract traces
#        :return:
#        """
#        
#        im_mean20 = self.make_average_tif(number_of_frames=20)
#        
#        im_mean20_correct = self.subtract_background(im_mean20, method = 'per_channel')
#        
#       
#        root, name = os.path.split(self.pks_fn)
#        pks_fn=os.path.join(root,name[:-4]+'-P.pks') 
#        pks2_fn=os.path.join(root,name[:-4]+'-P2.pks') 
#        if os.path.isfile(pks2_fn): 
#        # if you can load the pks data, load it
#             ptsG=[]
#             dstG=[]
#             with open(pks_fn, 'r') as infile:
#                 for jj in range(0,10000): # there will be a time when more than 10000 frames are generated
#                     A=infile.readline()
#                     if A=='':
#                         break
#                     ptsG.append([float(A.split()[1]),float(A.split()[2])])
#                     A=infile.readline()
#                     dstG.append([float(A.split()[1]),float(A.split()[2])])
#                     
#             ptsG=np.array(ptsG)
#             dstG=np.array(dstG)
#             pts_number =len(ptsG)
#             im_mean20_correctA=im_mean20_correct
#             
#             ptsG2=[]
#             with open(pks2_fn, 'r') as infile:
#                for jj in range(0,pts_number):
#                    for jj in range(0,10000): # there will be a time when more than 10000 frames are generated
#                     A=infile.readline()
#                     if A=='':
#                         break
#                     ptsG2.append([float(A.split()[1]),float(A.split()[2])])
#        else: 
#        # if you cannot load the pks data, calculate it
#            if self.pks_fn== self.filepath: 
#            # if they are the same, reuse im_mean20_correct; im_mean20_correctA is not stored/exported
#                im_mean20_correctA=im_mean20_correct
#            else: 
#            #otherwise make your own im_mean correct for pks detection
#                self.width_pixels, self.height_pixels,  self.number_of_frames, A = read_header(self.pks_fn)
#                im_array = np.dstack([read_one_page(self.pks_fn, pageNb=jj,A=self.movie_file_object,ii=self.ii).astype(float) for jj in range(20)])
#                im_mean20 = np.mean(im_array, axis=2).astype(int)
#                if 0: #older version, not matching pick spots
#                    bg = rollingball(im_mean20)[1]
#                    im_mean20_correctA = im_mean20 - bg
#                    im_mean20_correctA[im_mean20_correctA < 0] = 0       
#                    threshold = get_threshold(im_mean20_correctA)
#                    im_mean20_correctA=remove_background(im_mean20_correctA,threshold)      
#                else:
#                    sh=np.shape(im_mean20)
#                    thr_donor=get_threshold(im_mean20[:,1:sh[0]//2])
#                    thr_acceptor=get_threshold(im_mean20[:,sh[0]//2:])
#                    bg=np.zeros(sh)
#                    bg[:,1:sh[0]//2]=thr_donor
#                    bg[:,sh[0]//2:]=thr_acceptor
#                    im_mean20_correctA=remove_background(im_mean20,bg)   
#            if self.choice_channel=='d': 
#            # with donor channel ptsG, calculate position in acceptor dstG
#                pts_number, label_size, ptsG = image_adapt.analyze_label.analyze(im_mean20_correctA[:, 0:self.height_pixels//2])       
#                ptsG2 = image_adapt.analyze_label.analyze(im_mean20_correctA[:, self.height_pixels//2:])[2] 
#                ptsG2 = np.array([[ii[0] + self.width_pixels/2, ii[1]] for ii in ptsG2])
#
#                dstG=polywarp_apply(self.mapping.P,self.mapping.Q,ptsG)
#           #discard point close to edge image
#                for ii in range(pts_number-1,-1,-1): # range(5,-1,-1)=5,4,3,2,1,0
#                    discard_dstG=dstG[ii,0]<self.width_pixels//2-10 or dstG[ii,1]<10 or dstG[ii,0]>self.width_pixels-10 or dstG[ii,1]>self.height_pixels-10
#                    discard_ptsG=ptsG[ii,0]<10 or ptsG[ii,1]<10 or ptsG[ii,0]>self.width_pixels/2-10 or ptsG[ii,1]>self.height_pixels-10
#                    discard=discard_dstG+discard_ptsG
#                    if discard:
#                        ptsG=np.delete(ptsG,ii, axis=0)
#                        dstG=np.delete(dstG,ii, axis=0)
#                pts_number=len(ptsG) 
#                
#                print(pts_number)
#            
#            elif self.choice_channel=='a':
#            # with acceptor dstG, calculate position in donor channel ptsG
#                pts_number, label_size, dstG = image_adapt.analyze_label.analyze(im_mean20_correctA[:, self.height_pixels//2:])   
#                ptsG2 = image_adapt.analyze_label.analyze(im_mean20_correctA[:, :self.height_pixels//2])[2]  
##                ptsG = cv2.perspectiveTransform(dstG.reshape(-1, 1, 2),(self.mapping._tf2_matrix))#transform_matrix))
##                ptsG = ptsG.reshape(-1, 2)
#                ptsG=polywarp_apply(self.mapping.P21,self.mapping.Q21,dstG)
#               
#            #discard point close to edge image
#                for ii in range(pts_number-1,-1,-1): # range(5,-1,-1)=5,4,3,2,1,0
#                    discard_dstG=dstG[ii,0]<self.width_pixels//2-10 or dstG[ii,1]<10 or dstG[ii,0]>self.width_pixels-10 or dstG[ii,1]>self.height_pixels-10
#                    discard_ptsG=ptsG[ii,0]<10 or ptsG[ii,1]<10 or ptsG[ii,0]>self.width_pixels/2-10 or ptsG[ii,1]>self.height_pixels-10
#                    discard=discard_dstG+discard_ptsG
#                    if discard:
#                        ptsG=np.delete(ptsG,ii, axis=0)
#                        dstG=np.delete(dstG,ii, axis=0)
#                pts_number=   len(ptsG) 
#                print(pts_number)
#           
#            elif self.choice_channel=='da':
#                print('I have no clue yet how to do this')
#                # most likely they do not overlap before finding transformation, so what is the point of doing D+A?
#                #pts_number, label_size, ptsG = analyze_label.analyze(im_mean20_correctA[:, 0:self.height_pixels//2+im_mean20_correctA[:, self.height_pixels//2:]])                                
#            else:
#                print('make up your mind, choose wisely d/a/da')
#            
#            #saving to pks file
#            with open(pks_fn, 'w') as outfile:
#                for jj in range(0,pts_number):
#                    pix0=ptsG[jj][0]
#                    pix1=ptsG[jj][1]
#                    outfile.write(' {0:4.0f} {1:4.4f} {2:4.4f} {3:4.4f} {4:4.4f} \n'.format((jj*2)+1, pix0, pix1, 0, 0, width4=4, width6=6))
#                    pix0=dstG[jj][0]
#                    pix1=dstG[jj][1]
#                    outfile.write(' {0:4.0f} {1:4.4f} {2:4.4f} {3:4.4f} {4:4.4f} \n'.format((jj*2)+2, pix0, pix1, 0, 0, width4=4, width6=6))
#            
#            root, name = os.path.split(self.pks_fn)
#            pks_fn=os.path.join(root,name[:-4]+'-P2.pks') 
#            with open(pks_fn, 'w') as outfile:
#                for jj in range(len(ptsG2)):
#                    pix0=ptsG2[jj][0]
#                    pix1=ptsG2[jj][1]
#                    outfile.write(' {0:4.0f} {1:4.4f} {2:4.4f} {3:4.4f} {4:4.4f} \n'.format((jj*2)+1, pix0, pix1, 0, 0, width4=4, width6=6))
#        
##$$$$$$ BLACK BOX NUMBER #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$        
#        sizeGauss=11
#        ALL_GAUSS=makeGaussian(sizeGauss, fwhm=3, center=(sizeGauss//2, sizeGauss//2))          
##$$$$$$ BLACK BOX NUMBER #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#          
#        return bg, pts_number, dstG, ptsG, im_mean20_correct, ALL_GAUSS, ptsG2

    def show_selected_spots(self):
    #make a plot with selected spots
        #make image with found spots
        PL=plt.figure(14,figsize=(40,40))    
        plt.imshow(self.im_mean20_correct,vmax=np.amin(self.im_mean20_correct)+5)
        for ii in range((np.amax(np.shape(self.dstG)))): 
            plt.plot(self.ptsG[ii][0],self.ptsG[ii][1], 'wo',markerfacecolor='none', markersize=8)
            plt.plot(self.dstG[ii][0],self.dstG[ii][1], 'wv',markerfacecolor='none', markersize=8)
        for ii in range((np.amax(np.shape(self.ptsG2)))):
            plt.plot(self.ptsG2[ii][0],self.ptsG2[ii][1],'y^',markerfacecolor='none', markersize=8)
        PL.savefig(self.filepath[:-4]+'-P data found spots.tif')
        
        PL=plt.figure(15,figsize=(40,40))
        if self.choice_channel=='d': 
            for ii in range((np.amax(np.shape(self.ptsG2)))):
                plt.plot(self.ptsG2[ii][0]-len(self.im_mean20_correct)//2,self.ptsG2[ii][1], 'r^')#,markerfacecolor='none', markersize=8)
        else:
            for ii in range((np.amax(np.shape(self.ptsG2)))):
                plt.plot(self.ptsG2[ii][0],self.ptsG2[ii][1], 'r^')#,markerfacecolor='none', markersize=8)
        for ii in range((np.amax(np.shape(self.dstG)))): 
            plt.plot(self.ptsG[ii][0],self.ptsG[ii][1], 'ko',markerfacecolor='none', markersize=8)
            plt.plot(self.dstG[ii][0]-len(self.im_mean20_correct)//2,self.dstG[ii][1], 'kv',markerfacecolor='none', markersize=8)
       
        
        PL.savefig(self.filepath[:-4]+'-P data location spots.tif')  
  
#    def subtract_background(self, im):
#        im_correct = im - self.background
#        im_correct[im_correct < 0] = 0
#        return remove_background(im_correct, self.threshold)

    def get_image(self, idx):
        img= read_one_page(self.filepath, idx, self.movie_file_object, self.ii)
        #img = self.subtract_background(img)
        return Image(img, self.height_pixels, self.mapping._tf2_matrix, self.ptsG, self.dstG, self.pts_number, self.Gauss)
    
    def get_image_show(self, idx, hs,ws,siz): # example hs=650,ws=950,siz=20
        img= read_one_page(self.filepath, idx,self.movie_file_object,self.ii)
        plt.figure(idx)
        ax1=plt.subplot(1,2,1)
        ax1.imshow(img)
        ax1.set_xlim(hs,hs+siz)
        ax1.set_ylim(ws, ws+siz)
        
        img = self.subtract_background(img)
        ax2=plt.subplot(1,2,2)
        ax2.imshow(img)
        ax2.set_xlim(hs,hs+siz)
        ax2.set_ylim(ws, ws+siz)
        return Image(img, self.height_pixels, self.mapping._tf2_matrix, self.ptsG, self.dstG, self.pts_number, self.Gauss)
    
    def get_all_traces(self):
    # reutnr donor and acceptor for the full data set
        root, name = os.path.split(self.filepath)
        traces_fn=os.path.join(root,name[:-4]+'-P.traces') 
        Ncolours=2
        if os.path.isfile(traces_fn):
        # load if traces file already exist
             with open(traces_fn, 'r') as infile:
                 Nframes = np.fromfile(infile, dtype = np.int32, count = 1).item()
                 Ntraces = np.fromfile(infile, dtype = np.int16, count = 1).item()
                 rawData = np.fromfile(infile, dtype = np.int16, count = Ncolours*Nframes * Ntraces)
             orderedData = np.reshape(rawData.ravel(), (Ncolours, Ntraces//Ncolours, Nframes), order = 'F') 
             donor=orderedData[0,:,:]   
             acceptor=orderedData[1,:,:]
             donor=np.transpose(donor)
             acceptor=np.transpose(acceptor)
        else:
        # go through all images, extract donor and acceptor signal
            donor=np.zeros(( self.number_of_frames,self.pts_number))
            acceptor=np.zeros((self.number_of_frames,self.pts_number))
           
            t0 = time.time()  
            for ii in range(self.number_of_frames): #self.number_of_frames also works for pm, len(self.movie_file_object.filelist) not
                print(ii)
                img=self.get_image(ii)
                donor[ii,:]=img.donor # will multiply with gaussian, spot location is not drift compensated
                acceptor[ii,:]=img.acceptor # will multiply with gaussian, spot location is not drift compensated
            t1=time.time()
            elapsed_time=t1-t0; print(elapsed_time)    
            
            #root, name = os.path.split(self.filepath)
            
            #if os.path.isfile(trace_fn):
               
            with open(traces_fn, 'w') as outfile:
                 off = np.array([self.number_of_frames], dtype=np.int32)
                 off.tofile(outfile)
                 off = np.array([2*self.pts_number], dtype=np.int16)
                 off.tofile(outfile)
                 time_tr=np.zeros((self. number_of_frames,2*self.pts_number))
                 Ncolours=2
                 for jj in range(2*self.pts_number//Ncolours):
                     time_tr[:,jj*2] = donor[:,jj]
                     time_tr[:,jj*2+1]=  acceptor[:,jj]
                 off = np.array((time_tr), dtype=np.int16)
                 off.tofile(outfile)
        
        return donor, acceptor
    