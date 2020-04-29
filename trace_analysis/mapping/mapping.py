# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:55:53 2019

@author: ivoseverins
"""

import numpy as np
import matplotlib.pyplot as plt

from trace_analysis.mapping.icp import icp
from trace_analysis.coordinate_transformations import transform
from trace_analysis.image_adapt.polywarp import polywarp, polywarp_apply #required for nonlinear

class Mapping2:
    def __init__(self, source = None, destination = None, method = None,
                 transformation_type = 'linear', initial_translation = None):
        self.source_name = 'source'
        self.source = source #source=donor=left side image
        self.destination_name = 'destination'
        self.destination = destination #destination=acceptor=right side image
        self.method = method
        self.transformation_type = transformation_type
        self.initial_translation = initial_translation
        self.transformation = None
        self.transformation_inverse = None

        if (source is not None) and (destination is not None):
            if self.method is None: self.method = 'icp'
            self.perform_mapping()

#    @property
#    def transformation_inverse(self):
#        return np.linalg.inv(self.transformation)

    @property
    def transform_source_to_destination(self): 
        return self.transform_coordinates(self.source)

    def perform_mapping(self):
        print(self.transformation_type)
        if self.method == 'icp': #icp should be default
            self.transformation, distances, iterations, self.transformation_inverse = \
                icp(self.source, self.destination, initial_translation=self.initial_translation,
                    transformation_type=self.transformation_type)
        else: print('Method not found')

    def show_mapping_transformation(self, figure=None): 
        if not figure: figure = plt.figure()
        destination_from_source = self.transform_coordinates(self.source)

        axis = figure.gca()

        axis.scatter(self.source[:, 0], self.source[:, 1], c='g')
        axis.scatter(self.destination[:, 0], self.destination[:, 1], c='r')
        axis.scatter(destination_from_source[:, 0], destination_from_source[:, 1], c='y')

    def transform_coordinates(self, coordinates, inverse=False, direction=None):
        if direction is not None:
            if direction == self.source_name + '2' + self.destination_name:
                inverse = False
            elif direction == self.destination_name + '2' + self.source_name:
                inverse = True
            else:
                raise ValueError('Wrong direction')

        if not inverse:
            current_transformation = self.transformation
        else:
            current_transformation = self.transformation_inverse

        if self.transformation_type == 'linear':
            # Maybe we should rename transform to linear_transform [IS 05-03-2020]
            # transform(pointSet, transformationMatrix=None, **kwargs):
            return transform(coordinates[:, :2], current_transformation)
        elif self.transformation_type == 'nonlinear':
            return polywarp_apply(current_transformation[0], current_transformation[1], coordinates)

