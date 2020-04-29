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

from trace_analysis.movie.movie import Movie


class TifMovie(Movie):
    def __init__(self, arg, *args, **kwargs):
        super().__init__(arg, *args, **kwargs)
        
        self.writepath = self.filepath.parent
        self.name = self.filepath.with_suffix('').name
        
        #determine 8 bits or 16 bits
        # self.bitdepth = 16 if (self.filepath.name[-7:-4]=='_16') else 8

        self.threshold = {  'view':             (0,200),
                            'point-selection':  (45,25)
                            }


        self.read_header()
#        self.find_filelist()
#
#    def find_filelist(self):
#        self.filelist=[p.relative_to(self.filepath.parent) for p in self.filepath.parent.glob('*spool.dat')]
#        
#        #  correct numerical image name
#        self.filelist.sort(key=lambda x: str(x)[9::-1])
#       
#    def __repr__(self):
#        info = (('Original Filename', self.original_filename),
#                ('Date', self.date),
#                ('Camera Model', self.model),
#                ('Temperature (deg.C)', '{:f}'.format(self.temperature)),
#                ('Exposure Time', '{:f}'.format(self.exposuretime)),
#                ('Cycle Time', '{:f}'.format(self.cycletime)),
#                ('Number of accumulations', '{:d}'.format(self.accumulations)),
#                ('Pixel Readout Rate (MHz)', '{:f}'.format(self.readout)),
#                ("Horizontal Camera Resolution", '{:d}'.format(self.xres)),
#                ("Vertical Camera Resolution", '{:d}'.format(self.yres)),
#                ("Image width", '{:d}'.format(self.width)),
#                ("Image Height", '{:d}'.format(self.height)),
#                ("Horizontal Binning", '{:d}'.format(self.xbin)),
#                ("Vertical Binning", '{:d}'.format(self.ybin)),
#                ("EM Gain level", '{:f}'.format(self.gain)),
#                ("Vertical Shift Speed", '{:f}'.format(self.vertical_shift_speed)),
#                ("Pre-Amplifier Gain", '{:f}'.format(self.pre_amp_gain)),
#                ("Stacksize", '{:d}'.format(self.stacksize)),
#                ("Filesize", '{:d}'.format(self.filesize)),
#                ("Offset to Image Data", '{:f}'.format(self.m_offset)))
#        desc_len = max([len(d) for d in list(zip(*info))[0]]) + 3
#        res = ''
#        for description, value in info:
#            res += ('{:' + str(desc_len) + '}{}\n').format(description + ': ', value)
#
#        res = super().__repr__() + '\n' + res
#        return res
#
    def read_header(self):
        # im = self.read_frame(0)
        # width, height = im.shape
        with tifffile.TiffFile(self.filepath) as tif:
            tif_tags = {}
            for tag in tif.pages[0].tags.values():
               name, value = tag.name, tag.value
               tif_tags[name] = value
            self.width = tif_tags['ImageWidth']
            self.height = tif_tags['ImageLength']
            self.number_of_frames = len(tif.pages)
            self.bitdepth = tif_tags['BitsPerSample']
            ## hdim,vdim=tif.pages[0].shape

#
#        f = open(self.filepath, 'rb')
#
#
#                self.temperature = float(tokens[5])
#                self.date = time.strftime('%c', time.localtime(float(tokens[4])))
#                self.exposuretime = float(tokens[12])
#                self.cycletime = float(tokens[13])
#                self.accumulations = int(tokens[15])
#                self.readout = 1 / float(tokens[18]) / 1e6
#                self.gain = float(tokens[21])
#                self.vertical_shift_speed = float(tokens[41])
#                self.pre_amp_gain = float(tokens[43])
#            elif ii == 3:
#                self.model = line.decode('utf-8')
#            elif ii==4: #nImages is wrong, for test file it should be 5000, Python returns 40
#                self.width,self.height,_=[int(ii) for ii in line.decode('utf-8').split()]
#            elif ii == 5:
#                self.original_filename = line.decode('utf-8') # not so useful if you copy to a different computer
#
#            elif line[:12] == b'Pixel number' and len(line)>14:
#                line = line[12:]
#                tokens = line.split()
#                if len(tokens) < 6:
#                    raise Exception('Not able to read stacksize.')
#                self.yres = int(tokens[2])
#                self.xres = int(tokens[3])
#                self.stacksize = int(tokens[5])
#                self.number_of_frames = self.stacksize
####            elif ii == 44: #headerlen - 1:  ( b'65538 1 2048 2048 1 1 1 0')
#                #continue with next line
#                line = f.readline().strip()
#    #            print(ii),print(line)
#                tokens = line.decode('utf-8').split()
##                print(tokens)
#                if len(tokens) < 7:
#                   raise Exception("Not able to read Image dimensions.")
#                self.left = int(tokens[1])
#                self.top = int(tokens[2])
#                self.right = int(tokens[3])
#                self.bottom = int(tokens[4])
#                self.xbin = int(tokens[5])
#                self.ybin = int(tokens[6])
##                 self.left=0
##                 self.right=self.left+self.width
##                 self.bottom=0
##                 self.top=self.bottom+self.height
##                 self.xbin=1
##                 self.ybin=1
#                break
#     
#        f.close()
#
##        width = self.right - self.left + 1
#        width=self.width
#        mod = width % self.xbin
#        self.width = int((width - mod) / self.ybin)
##        height = self.top - self.bottom + 1
#        height=self.height
#        mod = height % self.ybin
#        self.height = int((height - mod) / self.xbin)
#
#        self.filesize = os.path.getsize(self.filepath)
#        self.datasize = self.width * self.height * 4 * self.stacksize
#        self.m_offset = self.filesize - self.datasize - 8
    


    def read_frame(self, frame_number):
        # t = time.time()

        with tifffile.TiffFile(self.filepath) as tif:


            tifpage = tif.pages
            if self.number_of_frames == 1:
                # return -1,0,0,0
                im = tifpage[0].asarray()
            elif (self.number_of_frames - 1) >= frame_number:
                im = tifpage[frame_number].asarray()
            else:
                im = tifpage[self.number_of_frames - 1].asarray()
                print('pageNb out of range, printed image {0} instead'.format(self.number_of_frames))
        # elapsed = time.time() - t
        # print(elapsed),
        return im

    # def read_frame(self, frame_number,ii=0):
    #     with self.filepath.open('rb') as fid:
    #         np.fromfile(fid, np.uint16,count=1)
    #         np.fromfile(fid, np.uint16,count=1)
    #
    #         if self.bitdepth == 8: #8 bits
    # #        with open(root+'\\'+name, 'rb') as fid: #did the offset reset?    # is already open
    #             # for image pageNb, 4 for skipping header, plus certain amount of images to read image pageNb
    #             fid.seek(4 + (frame_number*(self.width*self.height)), os.SEEK_SET)
    #             im = np.reshape(np.fromfile(fid,np.uint8,count=self.width*self.height),(self.width,self.height))
    #         else:
    # #        with open(root+'\\'+name, 'rb') as fid: #did the offset reset?  #is already open
    #             fid.seek(4+ 2*frame_number*(self.width*self.height), os.SEEK_SET)
    #             msb=np.reshape(np.fromfile(fid,np.uint8,count=(self.width*self.height)),(self.width,self.height))
    #             lsb=np.reshape(np.fromfile(fid,np.uint8,count=(self.width*self.height)),(self.width,self.height))
    # #            msb = np.core.records.fromfile(fid, 'int8', offset=4+ 2*pageNb*(hdim*vdim), shape=(hdim,vdim)) # for first image
    # #            lsb = np.core.records.fromfile(fid, 'int8', offset=4+ (1+2*pageNb)*(hdim*vdim), shape=(hdim,vdim)) # for first image
    #             im=256*msb+lsb;
    #
    #     if 0: # for testing match real data
    #         plt.imshow(im)
    #         tifffile.imwrite(self.writepath.joinPath(f'{self.name}_fr{frame_number}.tif') , im ,  photometric='minisblack')
    #
    #     return im # still need to convert im


if __name__ == "__main__":
    print('test')