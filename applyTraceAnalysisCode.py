# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:50:59 2018

@author: ivoseverins
"""


import os

import numpy as np
from matplotlib import pyplot as plt

# Change path to traceAnalysis directory (assuming this file is in that directory)
#os.chdir(os.path.dirname(os.path.abspath(__file__)))
from trace_analysis import Experiment
from trace_analysis.mapping.mapping import Mapping2

# Define path to data, replace by your own directory
# (Now it is set to the twoColourExampleData folder in the traceAnalysis directory)
mainPath = r'D:\pathToDataFolder'

# Initialize an experiment
exp = Experiment(mainPath, cropping = True)

# Select mapping file
map_file = exp.files[-2]
map_file.movie.show_average_tif()
map_file.movie.threshold['point-selection']=(400,200)
map_file.use_for_mapping()
map_file.movie.mapping.show_mapping_transformation()

for file in exp.files:
    file.movie.threshold['point-selection'] = (20, 20)
    file.movie.generate_pks_file(channel='d')

    file.importExtension('.pks')
    file.movie.show_coordinates(file.movie.average_image, file.coordinates)

    all_coordinates = np.concatenate([molecule.coordinates[:, :] for molecule in file.molecules])
    traces = file.movie.get_all_traces(all_coordinates)
    file.movie.write_traces_to_traces_file(traces)

    file.importExtension('.traces')

    file.background = np.array([1000, 1000])

exp.files[0].molecules[0].plot()

exp.files[0].select()





