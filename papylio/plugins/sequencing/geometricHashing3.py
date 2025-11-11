import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as pth
from pathlib import Path
import itertools
from scipy.spatial import cKDTree
import random
import matchpoint as mp
from papylio.plugins.sequencing.geometricHashing2 import polygon_area
import matchpoint as mp
from skimage.transform import AffineTransform
import time



class GeometricHashTable:
    def __init__(self, destinations=None, source_vertices=None, initial_source_transformation=AffineTransform(),
                 number_of_source_bases=20, number_of_destination_bases='all', load=False):
        # self.tile = tile
        # self.files = files # List of coordinate sets
        # self.dataPath = Path(dataPath)

        if load:
            self.load()
        else:

            self.initial_source_transformation = initial_source_transformation

            # self.mode = mode
            # if mode == 'translation': self.hashTableRange = [-10000, 10000]
            # else: self.hashTableRange = [-1,1]
            # self.nBins = nBins

            #
            # self.number_of_source_bases = number_of_source_bases
            # self.number_of_destination_bases = number_of_destination_bases

            # self._hashTable = None
            # self._matches = None
            #
            # self.rotationRange = rotationRange
            # self.magnificationRange = magnificationRange
            #


            self.destinations = destinations
            self.destination_KDTrees = [cKDTree(destination) for destination in destinations]
            self.source_vertices = source_vertices

            self.number_of_hashtable_entries_per_destination = []
            self.number_of_hashtable_entries_per_basis = []
            self.number_of_bases_per_destination = []

            self.basis_index_edges_for_destination = None
            self.entry_index_edges_for_destination = None
            self.entry_index_edges_for_basis = None

            self.hashtable = None

            self.create_hashtable()

    def save(self):
        # destinations_stacked = np.vstack([np.c_[i*np.ones(destination.shape[0], dtype=int), destination]
        #                                   for i, destination in enumerate(self.destinations)])
        destinations_stacked = np.vstack(self.destinations)
        destinations_length = np.array([len(destination) for destination in self.destinations])

        # number_of_hashtable_entries_per_basis_stacked = \
        #     np.vstack([np.c_[i * np.ones(len(n), dtype=int), n]
        #                for i, n in enumerate(self.number_of_hashtable_entries_per_basis)])

        number_of_hashtable_entries_per_basis_stacked = np.hstack(self.number_of_hashtable_entries_per_basis)
        number_of_hashtable_entries_per_basis_length = np.array([len(n) for n in self.number_of_hashtable_entries_per_basis])

        np.savez('geometric_hashtable.npz',
                    initial_source_transformation=self.initial_source_transformation.params,
                    destinations_stacked=destinations_stacked,
                    destinations_length=destinations_length,
                    hashtable=self.hashtable.data,
                    source_vertices=self.source_vertices,
                    number_of_hashtable_entries_per_destination=np.array(self.number_of_hashtable_entries_per_destination),
                    number_of_hashtable_entries_per_basis_stacked=number_of_hashtable_entries_per_basis_stacked,
                    number_of_hashtable_entries_per_basis_length=number_of_hashtable_entries_per_basis_length,
                    number_of_bases_per_destination=np.array(self.number_of_bases_per_destination),
                    basis_index_edges_for_destination=self.basis_index_edges_for_destination,
                    entry_index_edges_for_destination=self.entry_index_edges_for_destination,
                    entry_index_edges_for_basis=self.entry_index_edges_for_basis
                    )

    def load(self):
        file = np.load('geometric_hashtable.npz')
        self.initial_source_transformation = AffineTransform(file['initial_source_transformation'])
        self.destinations = np.split(file['destinations_stacked'], np.cumsum(file['destinations_length'])[:-1])
        self.destination_KDTrees = [cKDTree(destination) for destination in self.destinations]

        self.hashtable = cKDTree(file['hashtable'])
        self.source_vertices = file['source_vertices']

        self.number_of_hashtable_entries_per_destination = file['number_of_hashtable_entries_per_destination'].tolist()

        number_of_hashtable_entries_per_basis=\
            np.split(file['number_of_hashtable_entries_per_basis_stacked'],
                 np.cumsum(file['number_of_hashtable_entries_per_basis_length'])[:-1])
        self.number_of_hashtable_entries_per_basis = [n.tolist() for n in number_of_hashtable_entries_per_basis]

        self.number_of_bases_per_destination = file['number_of_bases_per_destination'].tolist()

        self.basis_index_edges_for_destination = file['basis_index_edges_for_destination']
        self.entry_index_edges_for_destination = file['entry_index_edges_for_destination']
        self.entry_index_edges_for_basis = file['entry_index_edges_for_basis']

    def create_hashtable(self):
        hashtable_entries_per_destination = []
        for destination in self.destinations:
            destination_bases = destination

            hashtable_entries_per_basis = [destination - basis for basis in destination_bases]

            if self.source_vertices is not None:
                # Make the area for which the hashtable is constructed twice the area of the source.
                # The area is still centered on the original source area
                center = np.mean(self.source_vertices, axis=0)
                crop_vertices_in_source = (self.source_vertices - center) * 2 + center
                crop_vertices_in_destination = self.initial_source_transformation(crop_vertices_in_source)
                hashtable_entries_per_basis = [mp.point_set.crop_coordinates(entries, crop_vertices_in_destination) for entries in hashtable_entries_per_basis]

            self.number_of_hashtable_entries_per_basis.append([len(entries) for entries in hashtable_entries_per_basis])

            self.number_of_bases_per_destination.append(len(destination))
            hashtable_entries_per_destination.append(np.vstack(hashtable_entries_per_basis))

        self.number_of_hashtable_entries_per_destination = [np.sum(c) for c in self.number_of_hashtable_entries_per_basis]

        self.basis_index_edges_for_destination = np.cumsum(np.hstack([[0], self.number_of_bases_per_destination]))
        self.entry_index_edges_for_destination = np.cumsum(np.hstack([[0], self.number_of_hashtable_entries_per_destination]))
        self.entry_index_edges_for_basis = np.cumsum(np.hstack([[0], np.hstack(self.number_of_hashtable_entries_per_basis)]))

        # self.destination_bases_index_bin_edges = np.cumsum([0]+[len(c) for i, c in enumerate(hash_table_entries_per_basis)])
        # or self.destination_bases_start_indices
        self.hashtable = cKDTree(np.vstack(hashtable_entries_per_destination))




    # def geometric_hash(point_sets, maximum_distance=100, tuple_size=4):
    #     # TODO: Add minimum_distance and implement
    #     # TODO: Make invariant to mirroring
    #     # TODO: Make usable with multiple point-sets in a single hash table
    #     # TODO: Implement names of point_sets, possibly through a dictionary and adding a attribute to each KDtree
    #
    #     if type(point_sets) is not list:
    #         point_sets = [point_sets]
    #
    #     point_set_KDTrees = [cKDTree(point_set) for point_set in point_sets]
    #
    #     hash_table = []
    #     point_tuple_sets = []
    #     for point_set_KDTree in point_set_KDTrees:
    #         point_tuple_generator = generate_point_tuples(point_set_KDTree, maximum_distance, tuple_size)
    #
    #         point_tuples = []
    #         for point_tuple, hash_code in geometric_hash_table(point_set_KDTree, point_tuple_generator, tuple_size):
    #             point_tuples.append(point_tuple)
    #             hash_table.append(hash_code)
    #
    #         point_tuple_sets.append(point_tuples)
    #
    #     hash_table_KDTree = cKDTree(np.array(hash_table))
    #
    #     return point_set_KDTrees, point_tuple_sets, hash_table_KDTree


    def query(self, source, distance=15, alpha=0.9, sigma=10, K_threshold=10e2):
        best_destination_basis_for_source_basis = []
        source_bases = source
        for basis in source_bases:
            hash_table_queries_in_source = source - basis
            hash_table_queries_in_destination = self.initial_source_transformation(hash_table_queries_in_source)
            found_destination_indices_per_query = self.hashtable.query_ball_point(hash_table_queries_in_destination, distance)
            # hash_table_queries_in_destination = cKDTree(hash_table_queries_in_destination)
            # hash_table_queries_in_destination = self.hashtable.query_ball_tree(hash_table_queries_in_destination, 15)
            found_destination_indices = np.hstack(found_destination_indices_per_query)

            count_per_destination_basis = np.histogram(found_destination_indices, bins=self.entry_index_edges_for_basis)[0]
            best_matching_destination_basis_index = np.argmax(count_per_destination_basis)
            max_count = np.max(count_per_destination_basis)
            best_destination_basis_for_source_basis.append((best_matching_destination_basis_index, max_count))

        # base_transformed = self.initial_image_transformation(np.array([[0,0]]))

        best_destination_basis_for_source_basis = \
            np.hstack([np.atleast_2d(np.arange(len(best_destination_basis_for_source_basis))).T,
                       best_destination_basis_for_source_basis])

        best_destination_basis_for_source_basis = best_destination_basis_for_source_basis[np.flip(np.argsort(best_destination_basis_for_source_basis[:, 2]))]

        for source_basis_index, destination_basis_index, count in best_destination_basis_for_source_basis:
        # i=-1
        # best_matching_source_basis_index = best_destination_basis_for_source_basis[i,0]
        # best_matching_destination_basis_index = best_destination_basis_for_source_basis[i,1]
            destination_index = np.where(destination_basis_index<self.basis_index_edges_for_destination[1:])[0][0]

            source_basis_in_destination = self.initial_source_transformation(source[source_basis_index])
            translation = self.destinations[destination_index][[destination_basis_index-self.basis_index_edges_for_destination[destination_index]]]\
                          - source_basis_in_destination

            found_transformation = self.initial_source_transformation + AffineTransform(translation=translation)

            # plt.figure()
            # mp.icp.scatter_coordinates([self.destination, found_transformation(source)])

            if self.test_transformation(source, self.destination_KDTrees[destination_index], found_transformation,
                                        alpha, sigma, K_threshold):
                match = mp.MatchPoint(source=source, destination=self.destinations[destination_index], method='Geometric hashing',
                                 transformation_type='linear', initial_transformation=None)
                match.transformation = found_transformation
                match.transformation_inverse = AffineTransform(matrix=found_transformation._inv_matrix)
                # match.calculate_inverse_transformation()
                match.destination_index = destination_index
                match.initial_transformation = self.initial_source_transformation
                match.source_vertices = self.source_vertices

                # match.hash_table_distance = distance
                #match.hash_table_distances_checked = hash_table_distances_checked
                #match.tuples_checked = tuples_checked
                return match

        else:
            return None


    def test_transformation(self, source, destination, found_transformation, alpha=0.9, sigma=10, K_threshold=10e2):
        if not type(destination) is cKDTree:
            destination_KDTree = cKDTree(destination)
        else:
            destination_KDTree = destination

        source_vertices_transformed = found_transformation(self.source_vertices)
        destination_cropped = mp.point_set.crop_coordinates(destination_KDTree.data, source_vertices_transformed)

        source_transformed_area = polygon_area(source_vertices_transformed)

        pDB = 1 / source_transformed_area

        K=1

        # TODO: Remove basis points?
        for coordinate in found_transformation(source):
            distance, index = destination_KDTree.query(coordinate)

            # 2d Gaussian
            pDF = alpha / source_transformed_area + \
                  (1 - alpha) / (2 * np.pi * sigma ** 2) * \
                  np.exp(-(distance ** 2) / (2 * sigma ** 2)) / len(destination_cropped)

            K = K * pDF/pDB
            if K > K_threshold:
                print("Found match")
                return True

        return False


if __name__ == '__main__':
    # # source = np.loadtxt(r'D:\SURFdrive\Promotie\Code\Python\papylio\papylio\plugins\sequencing\source.txt')
    # source = np.loadtxt(r'D:\SURFdrive\Promotie\Code\Python\papylio\papylio\plugins\sequencing\source3.txt')
    # destination1 = np.loadtxt(r'D:\SURFdrive\Promotie\Code\Python\papylio\papylio\plugins\sequencing\destination.txt')
    # destination2 = np.loadtxt(r'D:\SURFdrive\Promotie\Code\Python\papylio\papylio\plugins\sequencing\destination2.txt')
    # destinations = [destination1, destination2]
    #
    # initial_magnification = np.array([ 3.67058194, -3.67058194])
    # initial_rotation = 0.6285672733195177 # degrees
    #
    # initial_source_transformation = AffineTransform(matrix=None, scale=initial_magnification,
    #                                                 rotation=initial_rotation/360*np.pi*2,
    #                                                 shear=None, translation=None)
    # source_vertices = np.array([[256,   0], [512,   0], [512, 512], [256, 512]])
    #
    # ht = GeometricHashTable(destinations, source_vertices, initial_source_transformation=initial_source_transformation)
    #
    # match = ht.query(source, 15)

    from matchpoint.point_set_simulation import simulate_mapping_test_point_set

    # Simulate source and destination point sets
    number_of_source_points = 4000
    transformation = AffineTransform(translation=[128, 128], rotation=0 / 360 * 2 * np.pi, scale=[1, 1])
    source_bounds = np.array([[0, 0], [512, 512]])
    source_crop_bounds = np.array([[0, 0], [50, 50]])
    fraction_missing_source = 0.8
    fraction_missing_destination = 0.6
    maximum_error_source = 0
    maximum_error_destination = 0
    shuffle = True

    destination, source = simulate_mapping_test_point_set(number_of_source_points, transformation,
                                                          source_bounds, source_crop_bounds,
                                                          fraction_missing_source, fraction_missing_destination,
                                                          maximum_error_source, maximum_error_destination, shuffle)
    destinations = [destination]

    perfect = mp.MatchPoint(source, destination)
    perfect.transformation = AffineTransform(matrix=transformation._inv_matrix)
    perfect.show_mapping_transformation()

    # mp.icp.scatter_coordinates([source, destination])

    source_vertices = np.array([source_crop_bounds[0], source_crop_bounds.T[0],
                                source_crop_bounds[1], np.flip(source_crop_bounds.T[1])])
    ht = GeometricHashTable(destinations, source_vertices)

    distance = 5
    alpha = 0.5
    sigma = 2
    K_threshold = 10e8

    test = ht.query(source, distance, alpha, sigma, K_threshold)
    if test:
        test.show_mapping_transformation()
