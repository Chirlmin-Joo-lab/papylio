# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:10:04 2020

@author: mwdocter
"""
#import os
#os.getcwd()
#os.chdir('..') # one directory up

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import math
import logging

logger = logging.getLogger('ftpuploader')
hdlr = logging.FileHandler('ftplog.llog')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
FTPADDR = "some ftp address"

# steps in GUI:
from papylio.experiment import Experiment
from papylio.movie.movie import Movie
from papylio.movie.background_correction import get_threshold
from skimage import measure

def autoconfig_AND_perform_mapping(mapping_file_index, main_path):
    exp = Experiment(main_path)
    mapping_file = exp.files[mapping_file_index]
    print(mapping_file.name)
    try : 
        mapping_file.perform_mapping()
        print('no need for autoconfig')
    # if it does not workL autoconfig
    except BaseException as e:
        logger.error('Failed to do something 1: ' + str(e))
        autoconfig(exp, mapping_file, opt='mapping')
        try : 
            #mapping_file.restartreset # doe snot work, full reset of exp works. There should be a simpler way !!!!!
            del exp
            exp = Experiment(main_path)
            mapping_file = exp.files[mapping_file_index]
            mapping_file.perform_mapping()
            print('ran autoconfig once') 
        except BaseException as e:
            logger.error('Failed to do something 2: ' + str(e))
            print('#$%^&*(*&^%$')
     #still not enough points: lower the minimum difference, and raise the factor for mapping
            while True:
                exp.configuration['mapping']['peak_finding']['donor']['minimum_intensity_difference']= \
                  0.9*exp.configuration['mapping']['peak_finding']['donor']['minimum_intensity_difference']
                exp.configuration['mapping']['peak_finding']['acceptor']['minimum_intensity_difference']= \
                  0.9*exp.configuration['mapping']['peak_finding']['acceptor']['minimum_intensity_difference']
                exp.configuration['mapping']['coordinate_optimization']['coordinates_without_intensity_at_radius']['fraction_of_peak_max']=\
                  1.05*exp.configuration['mapping']['coordinate_optimization']['coordinates_without_intensity_at_radius']['fraction_of_peak_max']
                exp.export_config_file()
    
                try : 
                    del exp
                    exp = Experiment(main_path)
                    mapping_file = exp.files[mapping_file_index]
                    mapping_file.perform_mapping()
                    print('ran autoconfig again') 
                    break
                except BaseException as e:
                    logger.error('Failed to do something 3: ' + str(e))
                    if exp.configuration['mapping']['peak_finding']['donor']['minimum_intensity_difference']<1 or\
                       exp.configuration['mapping']['peak_finding']['acceptor']['minimum_intensity_difference']<1 or\
                       exp.configuration['mapping']['coordinate_optimization']['coordinates_without_intensity_at_radius']['fraction_of_peak_max']>1:
                           print('give up, this mapping file is of too low quality')
                           break  
                    continue
               
                
def autoconfig(exp, input_file, opt='mapping'): 
# step 1. find recommended threshold for donor and acceptor channel
    show=0
    average_image = input_file.average_image
    donor_image=input_file.movie.get_channel(channel='d')
    acceptor_image=input_file.movie.get_channel(channel='a')
    thr_donor=get_threshold(donor_image,show=show)
    thr_acceptor=get_threshold(acceptor_image,show=show)
    print('recommended value for mapping - donor - threshold: {0}'.format(thr_donor))
    print('recommended value for mapping - acceptor -  threshold: {0}'.format(thr_acceptor))

# step 2 find recommended thresholds dor the minimum intensity differences for donor&acceptor
    filter_neighbourhood_size=10
    image_max = filters.maximum_filter(donor_image, filter_neighbourhood_size)
    image_min = filters.minimum_filter(donor_image, filter_neighbourhood_size)
    image_diff_donor= image_max-image_min
    image_diff_donor_thr=get_threshold(image_diff_donor)

    image_max = filters.maximum_filter(acceptor_image, filter_neighbourhood_size)
    image_min = filters.minimum_filter(acceptor_image, filter_neighbourhood_size)
    image_diff_acceptor= image_max-image_min
    image_diff_acc_thr=get_threshold(image_diff_acceptor)

    print('recommended value for mapping - donor - min.int.diff: {0}'.format(image_diff_donor_thr))
    print('recommended value for mapping - acceptor - min.int.diff: {0}'.format(image_diff_acc_thr))

# step 3. find the recommended radius and fraction
    label_D2 = measure.label(image_diff_donor>image_diff_donor_thr, 8) 

    imm=donor_image

    D2min=np.zeros([np.max(label_D2), 1])
    D2max=np.zeros([np.max(label_D2), 1])
    area=np.zeros([np.max(label_D2), 1])
    crop_peak=np.zeros((np.max(label_D2), 2))
    crop_peak=np.zeros((np.max(label_D2), 2))
    factors1=np.zeros(np.shape(imm))
    factors0=np.zeros((np.max(label_D2), 2))
    
    # for each labeled spot, find the area, intensity in center and at radius,  
    for ii in range(np.max(label_D2)): 
        if ii>0:
            D2min[ii]=np.min(imm[label_D2==ii])
            D2max[ii]=np.max(imm[label_D2==ii])
            area[ii]=np.sum(label_D2==ii)
            
            location=np.mean(np.where(label_D2==ii),axis=1)
            center = np.round(location[::-1]).astype(int)
            width=4*2+1
            cropped_peak=imm[(center[1]-width//2):(center[1]+width//2+1),(center[0]-width//2):(center[0]+width//2+1)]
            if np.shape(cropped_peak)==(9,9):
                r=4
                d = 2*r + 1
                x, y = np.indices((d, d))
                circle_matrix= (np.abs(np.hypot(r - x, r - y)-r) < 0.5).astype(int)
                cutoff=np.median(imm)
                crop_peak[ii,:]=[np.max(cropped_peak * circle_matrix), np.max(cropped_peak)]
            factors1[label_D2==ii]=(crop_peak[ii,0]-cutoff)/(crop_peak[ii,1]-cutoff)
            factors0[ii,:]=[image_diff_donor[label_D2==ii][0], factors1[label_D2==ii][0]]
    radius=np.ceil(np.sqrt(np.ceil(np.median(area)/(4*np.pi))) )
    print('recommended value for the radius is : {0}'.format(radius) )  

    a,b=np.histogram(factors0[:,1], bins=50)
    factor_threshold= np.sum(b[1:-1]*a[:-1])/np.sum(a[:-1]) #weighted sum, except last entry
    
    print('recommended value for the factor is : {0}'.format(factor_threshold) )  

# step 4, write them in the config file
    print(opt + '\n')
   
    if opt=='mapping':
     #   print('doing mapping')  
        exp.configuration['mapping']['peak_finding']['donor']['minimum_intensity_difference']= \
              int(image_diff_donor_thr  )
        exp.configuration['mapping']['peak_finding']['acceptor']['minimum_intensity_difference']= \
              int(image_diff_acc_thr  )
        exp.configuration['mapping']['coordinate_optimization']['coordinates_without_intensity_at_radius']['radius']=\
              int( radius)
        exp.configuration['mapping']['coordinate_optimization']['coordinates_without_intensity_at_radius']['fraction_of_peak_max']=\
              float(   factor_threshold)
    else:
      #  print('doing find coordinate') 
        exp.configuration['find_coordinates']['peak_finding']['minimum_intensity_difference']= \
              int(image_diff_donor_thr  )
     #   print(exp.configuration['find_coordinates']['peak_finding']['minimum_intensity_difference'])
     #   print(exp.configuration['find_coordinates']['peak_finding']['maximum_intensity_difference'])
          
        if  exp.configuration['find_coordinates']['peak_finding']['minimum_intensity_difference'] > \
            exp.configuration['find_coordinates']['peak_finding']['maximum_intensity_difference']:
        #    print('minimum is above maximum?!')
            exp.configuration['find_coordinates']['peak_finding']['maximum_intensity_difference']=\
            int(image_diff_donor_thr  )+11111
        #    print(exp.configuration['find_coordinates']['peak_finding']['minimum_intensity_difference'])
        #    print(exp.configuration['find_coordinates']['peak_finding']['maximum_intensity_difference'])
       
        exp.configuration['find_coordinates']['coordinate_optimization']['coordinates_without_intensity_at_radius']['radius']=\
               int( radius)
        exp.configuration['find_coordinates']['coordinate_optimization']['coordinates_without_intensity_at_radius']['fraction_of_peak_max']=\
               float( factor_threshold)
    exp.export_config_file()
    