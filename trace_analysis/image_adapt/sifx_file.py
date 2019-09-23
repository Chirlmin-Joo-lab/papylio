# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:41:47 2019

@author: Ivo Severins, Margreet Doctor, https://github.com/lightingghost/sifreader/blob/master/sifreader/sifreader.py
"""

from pathlib import Path
import os, sys
   

import time
import numpy as np
import matplotlib.pyplot as plt
import tifffile

from trace_analysis.image_adapt.movie import Movie


class SifxFile(Movie):
    def __init__(self, arg,cropping = False, *args, **kwargs):
        self.cropping = cropping
        super().__init__(arg, *args, **kwargs)
        
        self.folderpath = self.filepath.parent
        self.writepath = self.filepath.parent.parent
        self.name = self.filepath.parent.name
        self.bitdepth = 16
        self.read_header()
        self.find_filelist()
        self.threshold = {  'view':             (0,200),
                            'point-selection':  (45,25)
                            }

    def find_filelist(self):
        self.filelist=[p.relative_to(self.filepath.parent) for p in self.filepath.parent.glob('*spool.dat')]
        
        #  correct numerical image name
        self.filelist.sort(key=lambda x: str(x)[9::-1])
       
    def __repr__(self):
        info = (('Original Filename', self.original_filename),
                ('Date', self.date),
                ('Camera Model', self.model),
                ('Temperature (deg.C)', '{:f}'.format(self.temperature)),
                ('Exposure Time', '{:f}'.format(self.exposuretime)),
                ('Cycle Time', '{:f}'.format(self.cycletime)),
                ('Number of accumulations', '{:d}'.format(self.accumulations)),
                ('Pixel Readout Rate (MHz)', '{:f}'.format(self.readout)),
                ("Horizontal Camera Resolution", '{:d}'.format(self.xres)),
                ("Vertical Camera Resolution", '{:d}'.format(self.yres)),
                ("Image width", '{:d}'.format(self.width)),
                ("Image Height", '{:d}'.format(self.height)),
                ("Horizontal Binning", '{:d}'.format(self.xbin)),
                ("Vertical Binning", '{:d}'.format(self.ybin)),
                ("EM Gain level", '{:f}'.format(self.gain)),
                ("Vertical Shift Speed", '{:f}'.format(self.vertical_shift_speed)),
                ("Pre-Amplifier Gain", '{:f}'.format(self.pre_amp_gain)),
                ("Stacksize", '{:d}'.format(self.stacksize)),
                ("Filesize", '{:d}'.format(self.filesize)),
                ("Offset to Image Data", '{:f}'.format(self.m_offset)))
        desc_len = max([len(d) for d in list(zip(*info))[0]]) + 3
        res = ''
        for description, value in info:
            res += ('{:' + str(desc_len) + '}{}\n').format(description + ': ', value)

        res = super().__repr__() + '\n' + res
        return res

    def read_header(self):
        f = open(self.filepath, 'rb')
        headerlen = 32
    #    spool = 0
        for ii in range(50):#headerlen + spool):
            line = f.readline().strip()
         #   print(ii,line)
            if ii == 0:
                if line != b'Andor Technology Multi-Channel File':
                    f.close()
                    raise Exception('{} is not an Andor SIF file'.format(self.filepath))
            # elif ii==1: # line=b'65538 1' , no clue what this means. 2048**2/64=65536
            elif ii == 2:
                tokens = line.split()
                self.temperature = float(tokens[5])
                self.date = time.strftime('%c', time.localtime(float(tokens[4])))
                self.exposuretime = float(tokens[12])
                self.cycletime = float(tokens[13])
                self.accumulations = int(tokens[15])
                self.readout = 1 / float(tokens[18]) / 1e6
                self.gain = float(tokens[21])
                self.vertical_shift_speed = float(tokens[41])
                self.pre_amp_gain = float(tokens[43])
            elif ii == 3:
                self.model = line.decode('utf-8')
            elif ii==4: #nImages is wrong, for test file it should be 5000, Python returns 40
                self.width,self.height,_=[int(ii) for ii in line.decode('utf-8').split()]
            elif ii == 5:
                self.original_filename = line.decode('utf-8') # not so useful if you copy to a different computer
            #elif ii==6: # b'65538 2048'
#            elif ii == 7: # lots of characters, tokens[0] is not Spooled, maybe wrong line?
#                tokens = line.split()
#                if len(tokens) >= 1 and tokens[0] == 'Spooled':
#                    spool = 1
            #elif ii==8: #same junk as ii==7
            #elif i == 9: #similar junk characters as ii==7
            #    wavelength_info = line.split() # example [b'\x00\x00\x00\x00\x00\x00\x00\xfb\x97\x1e2\x05\x00\x00\x00\x00\x00\x00\x00']
#                self.center_wavelength = float(wavelength_info[3])
#                self.grating = float(wavelength_info[6])
#                self.grating_blaze = float(wavelength_info[7])
            #elif 13: # (b'65538 \x01 \x02 \x03 \x00 0 0',)
            #elif ii==14: # (b'65540 0 0 500 0 0 1200 1200',)
            #elif ii==17: # (b'0 SR303i',) 
#            elif i == 19: #(b'0 10',)
#                self.wavelength_coefficients = [float(num) for num in line.split()][::-1]
#                self.wavelength_coefficients = 0
            #elif 21: # (b'65537 1 500 200',)
            #elif 7 < ii < headerlen - 12:
###            elif len(line) == 37 and line[0:6] == b'65539 ':#len(line) == 17
###                   # and line[7] == b'x01' and line[8] == b'x20' \
###                   # and line[9] == b'x00':
###                    headerlen = headerlen + 12
#                   
###            elif ii == 43: #headerlen - 2:
            elif line[:12] == b'Pixel number' and len(line)>14:
                line = line[12:]
                tokens = line.split()
                if len(tokens) < 6:
                    raise Exception('Not able to read stacksize.')
                self.yres = int(tokens[2])
                self.xres = int(tokens[3])
                self.stacksize = int(tokens[5])
                self.number_of_frames = self.stacksize
###            elif ii == 44: #headerlen - 1:  ( b'65538 1 2048 2048 1 1 1 0')
                #continue with next line
                line = f.readline().strip()
    #            print(ii),print(line)
                tokens = line.decode('utf-8').split()
#                print(tokens)
                if len(tokens) < 7:
                   raise Exception("Not able to read Image dimensions.")
                self.left = int(tokens[1])
                self.top = int(tokens[2])
                self.right = int(tokens[3])
                self.bottom = int(tokens[4])
                self.xbin = int(tokens[5])
                self.ybin = int(tokens[6])
#                 self.left=0
#                 self.right=self.left+self.width
#                 self.bottom=0
#                 self.top=self.bottom+self.height
#                 self.xbin=1
#                 self.ybin=1
                break
     
        f.close()

#        width = self.right - self.left + 1
        if self.cropping == True:
            self.width=int(self.width/4)
            self.height=int(self.height/4)
        width=self.width
        mod = width % self.xbin
        self.width = int((width - mod) / self.ybin)
#        height = self.top - self.bottom + 1
        height=self.height
        mod = height % self.ybin
        self.height = int((height - mod) / self.xbin)

        self.filesize = os.path.getsize(self.filepath)
        self.datasize = self.width * self.height * 4 * self.stacksize
        self.m_offset = self.filesize - self.datasize - 8
    
       
    def read_frame(self, frame_number):
        if self.cropping == True:
            self.height=int(self.height*4)
            self.width=int(self.width*4)
        if (self.xbin == 2) and (self.ybin == 2):
             count=self.height*self.width*4
             #name should follow from A.filelist
             with self.folderpath.joinpath(self.filelist[frame_number]).open('rb') as fid:
                 raw=np.uint16(np.fromfile(fid,np.uint8,count))
             ALL = raw[0::4]+raw[1::4]*256
            
        else:
             count=self.height*self.width*3//2
             #name should follow from A.filelist
             with self.folderpath.joinpath(self.filelist[frame_number]).open('rb') as fid:
                  raw=np.uint16(np.fromfile(fid,np.uint8,count))
             
             if not hasattr(self, 'ii'):
                 self.ii = np.array(range(int(self.width/2)*int(self.height/2)))
                
             #print([A.height,A.width,A.stacksize,np.shape(raw)])        
             AA=raw[self.ii*6+0]*16 + (raw[self.ii*6+1]%16)
             BB=raw[self.ii*6+2]*16 + (raw[self.ii*6+1]//16)
             CC=raw[self.ii*6+3]*16 + (raw[self.ii*6+4]%16)
             DD=raw[self.ii*6+5]*16 + (raw[self.ii*6+4]//16) 
                  
             ALL=np.uint16(np.zeros(self.height*self.width))
             ALL[0::4] = AA
             ALL[1::4] = BB
             ALL[2::4] = CC
             ALL[3::4] = DD
                  
        im=np.reshape(ALL,(self.height, self.width))
        if self.cropping == True:
            xstart_green = int((1024 - 256) / 2)
            xend_green = int(1024 - xstart_green)
            xstart_red = int((1024 - 256) / 2)+1024
            xend_red = int(1024 - xstart_red)
            ystart = int((2048 - 512) / 2)
            yend = int(2048 - ystart)
            self.height=int(self.height/4)
            self.width=int(self.width/4)
            im=np.concatenate((im[xstart_green:xend_green,ystart:yend],im[xstart_red:xend_red,ystart:yend]))
        im = np.rot90(im)
        if 0: # for testing match real data
            plt.imshow(im)
            tifffile.imwrite(self.writepath.joinPath(f'{self.name}_fr{frame_number}.tif') , im ,  photometric='minisblack')
        
        return im