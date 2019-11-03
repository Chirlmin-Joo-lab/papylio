# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 21:35:43 2019

@author: pimam
"""

import traceAnalysisCode as trace_ana
import interactiveAnalysis as interact_ana
import dwellAnalysis as dwell_ana
import dwellFinder as dwell_find
import distributionPlot as dist_ana
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import PureWindowsPath
from PIL import Image


def plotROI(exp_file, mole):
    img = exp_file.avgImage
    #img = Image.open('./hel23.tif')

    coor = exp_file.molecules[mole].coordinates
    print(coor)
    bnd = 5
    totwidth = 4*bnd+2
    totheight = 2*bnd+1

    ROI1 = img.crop((coor[0]-bnd-1, coor[1]-bnd-1, coor[0]+bnd, coor[1]+bnd))
    ROI2 = img.crop((coor[2]-bnd-1, coor[3]-bnd-1, coor[2]+bnd, coor[3]+bnd))
    ROI_mrg = Image.new("P", (totwidth, totheight))
    ROI_mrg.paste(ROI1, (0, 0))
    ROI_mrg.paste(ROI2, (2*bnd+1, 0))
    
    
    pixels = ROI_mrg.load()

    for x in range(2*bnd+1):
        for y in range(2*bnd+1):
            if (x-bnd)**2 + (y-bnd)**2 == (bnd)**2:
                pixels[x, y] = 30
                pixels[x+2*bnd+1, y] = 30
    #print(ROI1.getcolors())
    print(ROI_mrg.getbands())
    
    return img, ROI1, ROI2, ROI_mrg


if __name__ == '__main__':
    mainPath = PureWindowsPath('C:\\Users\\pimam\\Documents\\MEP\\tracesfiles')
    exp = trace_ana.Experiment(mainPath)
    file = exp.files[0]

    img, ROI1, ROI2, ROI = plotROI(file, 5)

    plt.figure(1)
    plt.imshow(img)
    plt.figure(2)
    plt.imshow(ROI1)
    plt.figure(3)
    plt.imshow(ROI2)

    fig, ax = plt.subplots(2, 1)
    ax[0].axis('off')
    ax[0].imshow(ROI)
