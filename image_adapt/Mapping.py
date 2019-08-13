# -*- coding: utf-8 -*-
"""
Created on Jun 2019
@author: carlos

a seperate file for doing mapping. 
this should be adapted to start with manual 3 points, then automated numerous points.
like the idl code, and save .coeff and .map files
"""
from ..autopick.pick_spots_akaze_ALL import mapping_manual, mapping_automatic
import numpy as np
import os
import tifffile as TIFF
import matplotlib.pyplot as plt
import cv2
from .find_threshold import remove_background, get_threshold
from ..autopick.pick_spots_akaze_ALL import load_map, load_coeff

class Mapping:#(object):
    def __init__(self, tetra_fname,f=10000, generic=0):
        self._tetra_fn = tetra_fname
        if generic==1:
            print('generic')
           
            save_fn=os.path.join(os.path.split(tetra_fname)[0],'generic.coeff') 
            self._tf1_matrix =load_coeff(save_fn)
                           
            save_fn=os.path.join(os.path.split(tetra_fname)[0],'generic.map')  
            self.P,self.Q,self._tf2_matrix=load_map(save_fn) 
                        
        else:
            self.manual_align()
            self.automatic_align()
        self.visualize()
                

    @property
    def tetra_fn(self):
        return self._tetra_fn

    def manual_align(self):
        self._tf1_matrix, self.points_right,self.points_left, self.fL,self.fR= mapping_manual(self._tetra_fn,
                                                                             show=0,
                                                                             bg=None,
                                                                             tol=0, f=None)
        
       # print(self.fL,self.fR)
        return self._tf1_matrix
    
    #@tetra_fn.setter
    def automatic_align(self):
        self.P,self.Q,self._tf2_matrix,self.position1, self.position2, self.pts1, self.pts2,self.dst2, self.P21,self.Q21 = mapping_automatic(self._tetra_fn,self._tf1_matrix,
                                                                        show=0,
                                                                        bg=None,
                                                                        tol=0, fL=self.fL,fR=self.fR)
        return self._tf2_matrix,self.P,self.Q
    
    
    def visualize(self):
        plt.close('all')
# show raw image        
        import re 
        from image_adapt.load_file import read_one_page, read_header
        if re.search('tif',self._tetra_fn)!=None: 
            image_tetra_raw = TIFF.imread(self._tetra_fn)
        elif re.search('sifx',self._tetra_fn)!=None: 
            try: image_tetra_raw = TIFF.imread(self._tetra_fn[:-5]+'.tif')
            except:
                    hdim, vdim, n_images,A = read_header(self._tetra_fn)
                    im_array = np.dstack([(read_one_page(self._tetra_fn, pageNb=jj,A=A,ii=np.array(range(1024*1024)))).astype(float) for jj in range(20)])
                    image_tetra_raw = np.mean(im_array, axis=2).astype(np.uint16)
                    TIFF.imwrite(self._tetra_fn[:-5]+'.tif',image_tetra_raw.astype(np.uint16))
            
        PL=plt.figure(1,figsize=(40,40)); plt. subplot(1,1,1)
        plt.imshow(image_tetra_raw, vmin=np.amin(image_tetra_raw), vmax=np.amin(image_tetra_raw)+200)
        plt.title('single image tetraspeck')
        PL.savefig(self._tetra_fn[:-4]+'-P VIS image_raw.tif')    
        
# show adapted image + manually mapped points (manual mapping=linear)        
        sh=np.shape(image_tetra_raw)
        thr_donor=get_threshold(image_tetra_raw[:,1:sh[0]//2])
        thr_acceptor=get_threshold(image_tetra_raw[:,sh[0]//2:])
        bg=np.zeros(sh)
        bg[:,1:sh[0]//2]=thr_donor
        bg[:,sh[0]//2:]=thr_acceptor
        image_tetra=remove_background(image_tetra_raw.astype(float),bg)
        
        PL=plt.figure(2,figsize=(40,40)); plt. subplot(1,1,1)
        plt.imshow(image_tetra, vmin=np.amin(image_tetra), vmax=np.amin(image_tetra)+200)
        dstGm = cv2.perspectiveTransform(self.points_right.reshape(-1, 1, 2), np.linalg.inv(self._tf1_matrix))#transform_matrix))
        dstGm = dstGm.reshape(-1, 2)
        for ii in range((np.amax(np.shape(self.points_left)))): 
            plt.plot(self.points_left[ii][0],self.points_left[ii][1], 'wo',markerfacecolor='none', markersize=10)
            plt.plot(self.points_right[ii][0],self.points_right[ii][1], 'ws',markerfacecolor='none', markersize=15)
            plt.plot(self.points_right[ii][0]+len(image_tetra_raw)//2,self.points_right[ii][1], 'ws',markerfacecolor='none', markersize=15)
            plt.plot(dstGm[ii][0],dstGm[ii][1], 'wd',markerfacecolor='none', markersize=10)
                 #plt.plot(dstGm[ii][0]+len(image_tetra_raw)//2,dstGm[ii][1], 'wd',markerfacecolor='none', markersize=10)
        plt.title('tetraspeck+manual mapping')
        PL.savefig(self._tetra_fn[:-4]+'-P VIS manual mapping.tif')   
            
# show adapted image + automatic found spots        
        PL=plt.figure(3,figsize=(40,40)); plt. subplot(1,1,1)
        plt.imshow(image_tetra, vmin=np.amin(image_tetra), vmax=np.amin(image_tetra)+20)
        for ii in range((np.amax(np.shape(self.position1)))): 
            plt.plot(self.position1[ii][0],self.position1[ii][1], 'wo',markerfacecolor='none', markersize=5)
        for ii in range((np.amax(np.shape(self.position2)))): 
            plt.plot(self.position2[ii][0],self.position2[ii][1], 'ws',markerfacecolor='none', markersize=5)
        plt.title('all positions found on tetraspeck')
        PL.savefig(self._tetra_fn[:-4]+'-P VIS positions.tif')

# show difference found and selected spots on tetra image
        PL=plt.figure(4,figsize=(40,40)); plt. subplot(1,1,1)
        plt.imshow(image_tetra, vmin=np.amin(image_tetra), vmax=np.amin(image_tetra)+20)
        for ii in range((np.amax(np.shape(self.position1)))): 
            plt.plot(self.position1[ii][0],self.position1[ii][1], 'wo',markerfacecolor='none', markersize=5)
        for ii in range((np.amax(np.shape(self.position2)))): 
            plt.plot(self.position2[ii][0],self.position2[ii][1], 'ws',markerfacecolor='none', markersize=5)
        PL.savefig(self._tetra_fn[:-4]+'-P VIS positions.tif')
        for ii in range((np.amax(np.shape(self.pts1)))): 
            plt.plot(self.pts1[ii][0],self.pts1[ii][1], 'wo',markerfacecolor='none', markersize=8)
            plt.plot(self.pts2[ii][0],self.pts2[ii][1], 'ws',markerfacecolor='none', markersize=8)
            plt.plot(self.dst2[ii][0],self.dst2[ii][1], 'wd',markerfacecolor='none', markersize=8)
        plt.title('all selected positions found on tetraspeck')
        PL.savefig(self._tetra_fn[:-4]+'-P VIS selected.tif')
                
# show overlap between found and transformed spots
        PL=plt.figure(5,figsize=(20,40)); plt. subplot(1,1,1) 
        for ii in range((np.amax(np.shape(self.pts1)))): 
            plt.plot([self.pts1[ii][0],self.pts2[ii][0]-np.shape(image_tetra)[0]//2],[self.pts1[ii][1],self.pts2[ii][1]],'b-' )
            plt.plot(self.pts1[ii][0],self.pts1[ii][1], 'bo',markerfacecolor='none', markersize=4)
            plt.plot(self.pts2[ii][0]-np.shape(image_tetra)[0]//2,self.pts2[ii][1], 'ks',markerfacecolor='none', markersize=4)
            plt.plot([self.dst2[ii][0]-np.shape(image_tetra)[0]//2,self.pts2[ii][0]-np.shape(image_tetra)[0]//2],[self.dst2[ii][1],self.pts2[ii][1]],'r-' )
            plt.plot(self.dst2[ii][0]-np.shape(image_tetra)[0]//2,self.dst2[ii][1], 'rd',markerfacecolor='none', markersize=4)
        PL.savefig(self._tetra_fn[:-4]+'-P VIS select.tif')  
        PL.set_size_inches(5,10, forward=True)

# show overlap/mismatch between all found spots        
        PL=plt.figure(6,figsize=(20,40)); plt. subplot(1,1,1) 
        for ii in range((np.amax(np.shape(self.position1)))): 
            plt.plot(self.position1[ii][0],self.position1[ii][1], 'ko',markerfacecolor='none', markersize=8)
        for ii in range((np.amax(np.shape(self.position2)))): 
            plt.plot(self.position2[ii][0]-np.shape(image_tetra)[0]//2,self.position2[ii][1], 'ks',markerfacecolor='none', markersize=6)
        PL.savefig(self._tetra_fn[:-4]+'-P VIS pos_overlap.tif')  
        PL.set_size_inches(5,10, forward=True)
        

    
    
   
                 