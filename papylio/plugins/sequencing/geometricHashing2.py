import numpy as np
import matplotlib.pyplot as plt

import itertools
from scipy.spatial import cKDTree
import random
import matchpoint as mp
from skimage.transform import AffineTransform
import time


class GeometricHashTable:
    def __init__(self, destinations, source_vertices=None, initial_source_transformation=AffineTransform(),
                 number_of_source_bases=20, number_of_destination_bases='all',
                 tuple_size=4, maximum_distance_source=None, maximum_distance_destination=None):
        self.initial_source_transformation = initial_source_transformation

        self.tuple_size = tuple_size
        self.maximum_distance_source = maximum_distance_source
        self.maximum_distance_destination = maximum_distance_destination

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

        self.create_hashtable()

    def create_hashtable(self):
        destination_hash_data = geometric_hash(self.destinations, self.maximum_distance_destination, self.tuple_size)
        self.destination_KDTrees, self.destination_tuple_sets, self.destination_hash_table_KDTree, \
        self.destination_transformation_matrices = destination_hash_data


    def query(self, source, distance=15, alpha=0.9, sigma=10, K_threshold=10e9, hash_table_distance_threshold=0.01,
              magnification_range=None, rotation_range=None):

        return find_match_after_hashing(source, self.maximum_distance_source, self.tuple_size, source_vertices,
                                        self.destination_KDTrees, self.destination_tuple_sets,
                                        self.destination_hash_table_KDTree,
                                        hash_table_distance_threshold, alpha, sigma, K_threshold,
                                        magnification_range, rotation_range)

    def query_tuple_transformations(self, sources, hash_table_distance_threshold=0.01, parameters=['rotation', 'scale'],
                                    bins=200):
        # np.vstack(sources)

        source_hash_data = geometric_hash(sources, self.maximum_distance_source, self.tuple_size)
        _, _, source_hash_table_KDTree, source_transformation_matrices = source_hash_data

        return compare_tuple_transformations(source_hash_table_KDTree, source_transformation_matrices,
                                              self.destination_hash_table_KDTree, self.destination_transformation_matrices,
                                              hash_table_distance_threshold, parameters, bins)

    def test(self, sources, hash_table_distance_threshold=0.01, bins=50):
        # np.vstack(sources)

        source_hash_data = geometric_hash(sources, self.maximum_distance_source, self.tuple_size)
        source_KDTrees, source_tuple_sets, source_hash_table_KDTree, source_transformation_matrices = source_hash_data

        tuple_matches = source_hash_table_KDTree.query_ball_tree(self.destination_hash_table_KDTree,
                                                                 hash_table_distance_threshold)

        destination_basis_points = np.array([self.destination_tuple_sets[0][i][0:2] for tuple_match in tuple_matches for i in tuple_match])
        basis_centers = self.destinations[0][destination_basis_points].mean(axis=1)
        plt.scatter(basis_centers[:, 0], basis_centers[:, 1], marker='.')

        plt.figure()
        h, edges = np.histogramdd(basis_centers, bins=bins)
        bin_centers = [(e[:-1] + e[1:]) / 2 for e in edges]
        plt.imshow(h)

        destination_basis_points = np.array([[i]+list(self.destination_tuple_sets[0][j][0:2]) for i, tuple_match in enumerate(tuple_matches) for j in tuple_match])




        linked_tuple_indices = np.array([[i, j] for i, tuple_match in enumerate(tuple_matches) for j in tuple_match])

        source_tuple_centers = source_KDTrees[0].data[np.array(source_tuple_sets[0])].mean(axis=1)
        destination_tuple_centers = self.destination_KDTrees[0].data[np.array(self.destination_tuple_sets[0])].mean(axis=1)

        linked_tuple_centers = np.hstack([source_tuple_centers[linked_tuple_indices[:,0]],destination_tuple_centers[linked_tuple_indices[:,1]]])

        ttt = cKDTree(linked_tuple_centers)
        a = np.unique([j for i in ttt.query_pairs(r=10) for j in i])

        tt1 = cKDTree(linked_tuple_centers[:,:2])
        tt2 = cKDTree(linked_tuple_centers[:,2:])
        a1 = tt1.query_pairs(1)
        a2 = tt2.query_pairs(1)
        ta = np.array(list(a1.intersection(a2)))

        plt.figure()
        c_before = destination_tuple_centers[linked_tuple_indices[:,1]]
        c_after = destination_tuple_centers[linked_tuple_indices[a,1]]
        plt.scatter(self.destination_KDTrees[0].data[:,0], self.destination_KDTrees[0].data[:,1])
        plt.scatter(c_before[:,0], c_before[:,1], c='g')
        plt.scatter(c_after[:,0], c_after[:,1], c='r')



        transformation_matrices = []
        for source_index, destination_indices in linked_tuple_indices[a]:
            source_transformation_matrix = source_transformation_matrices[source_index]
            # source_transformation_matrix = np.linalg.inv(source_transformation_matrix)
            for destination_index in destination_indices:
                destination_transformation_matrix = self.destination_transformation_matrices[destination_index]
                destination_transformation_matrix_inverse = np.linalg.inv(destination_transformation_matrix)
                transformation_matrices.append(destination_transformation_matrix_inverse @ source_transformation_matrix)
        transformation_matrices = np.stack(transformation_matrices)

        # plt.figure()
        # plt.hist(transformation_matrices[:, 0, 2], 100)

        transformations = [AffineTransform(transformation_matrix) for transformation_matrix in transformation_matrices]

        parameter_values = [np.array([getattr(t, parameter) for t in transformations]).T for parameter in parameters]


def mapToPoint(pointSet, startPoints, endPoints, returnTransformationMatrix=False, tr=None, di=None, ro=None):
    startPoints = np.atleast_2d(startPoints)
    endPoints = np.atleast_2d(endPoints)
    if len(startPoints) == 1 & len(endPoints) == 1:
        tr = True;
        ro = False;
        di = False

    elif len(startPoints) == 2 & len(endPoints) == 2:
        if tr is None: tr = True
        if di is None: di = True
        if ro is None: ro = True

    transformationMatrix = np.identity(3)

    if tr:
        translationMatrix = mp.coordinate_transformations.translate(endPoints[0] - startPoints[0])
        transformationMatrix = translationMatrix @ transformationMatrix

    if di or ro:
        diffs = np.array([startPoints[0] - startPoints[1], endPoints[0] - endPoints[1]])
        diffLengths = np.linalg.norm(diffs, axis=1, keepdims=True)
        unitDiffs = diffs / diffLengths

        if di:
            dilationMatrix = mp.coordinate_transformations.magnify(diffLengths[1] / diffLengths[0], endPoints[0])
            transformationMatrix = dilationMatrix @ transformationMatrix

        if ro:
            angle = -np.arctan2(np.linalg.det(unitDiffs), np.dot(unitDiffs[0], unitDiffs[1]))
            # angle = np.arccos(np.dot(diffs[0]/endLength,diffs[1]/startLength))
            rotationMatrix = mp.coordinate_transformations.rotate(angle, endPoints[0])
            transformationMatrix = rotationMatrix @ transformationMatrix

    pointSet = np.append(pointSet, np.ones((pointSet.shape[0], 1)), axis=1)
    transformedPointSet = (transformationMatrix @ pointSet.T)[0:2, :].T

    if returnTransformationMatrix:
        return transformedPointSet, transformationMatrix
    else:
        return transformedPointSet


def compare_tuple_transformations(source_hash_table_KDTree, source_transformation_matrices, destination_hash_table_KDTree,
                                  destination_transformation_matrices, hash_table_distance_threshold=0.01,
                                  parameters=['rotation', 'scale'], bins=200):
    tuple_matches = source_hash_table_KDTree.query_ball_tree(destination_hash_table_KDTree, hash_table_distance_threshold)

    # TODO: make this matrix multiplication
    transformation_matrices = []
    for source_index, destination_indices in enumerate(tuple_matches):
        source_transformation_matrix = source_transformation_matrices[source_index]
        # source_transformation_matrix = np.linalg.inv(source_transformation_matrix)
        for destination_index in destination_indices:
            destination_transformation_matrix = destination_transformation_matrices[destination_index]
            destination_transformation_matrix_inverse = np.linalg.inv(destination_transformation_matrix)
            transformation_matrices.append(destination_transformation_matrix_inverse @ source_transformation_matrix)
    transformation_matrices = np.stack(transformation_matrices)

    # plt.figure()
    # plt.hist(transformation_matrices[:, 0, 2], 100)

    transformations = [AffineTransform(transformation_matrix) for transformation_matrix in transformation_matrices]


    parameter_values = [np.array([getattr(t, parameter) for t in transformations]).T for parameter in parameters]

    sample = np.vstack(list(parameter_values)).T
    # sample = transformation_matrices[:, :2, :].reshape(-1, 6)

    h, edges = np.histogramdd(sample, bins=bins)

    bin_centers = [(e[:-1]+e[1:])/2 for e in edges]
    # found_transformation = np.array([bc[h_index] for bc, h_index in zip(bin_centers, np.where(h==h.max()))]).reshape(2,3)
    # t = AffineTransform(np.vstack([found_transformation, [0, 0, 1]]))
    hist_max_index = np.where(h == h.max())
    if len(hist_max_index[0]) > 1:
        raise RuntimeError('No optimal transformation found')
    found_values = [bc[h_index][0] for bc, h_index in zip(bin_centers, hist_max_index)]

    parameter_dict = {parameter: found_values.pop(0) if parameter == 'rotation' else [found_values.pop(0), found_values.pop(0)]
                      for parameter in parameters}

    # found_transformation = AffineTransform(**parameter_dict)

    # Hc = H.copy()
    # Hc[i]=0
    # print(np.max(H)/np.max(Hc)>2)

    # x, y = bin_centers_x, bin_centers_y
    # X, Y = np.meshgrid(x, y)
    # xdata = np.vstack((X.ravel(), Y.ravel()))

    # def twoD_gaussian(M, offset, amplitude, x0, y0, sigma_x, sigma_y):
    #     x, y = M
    #     return offset + amplitude * np.exp(- ((x - x0) / (2 * sigma_x)) ** 2 - ((y - y0) / (2 * sigma_y)) ** 2)
    #
    # from scipy.optimize import curve_fit
    # p0 = [20,20,0,0,1,1]
    # popt, pcov = curve_fit(twoD_gaussian, xdata, H.ravel())#, p0) #input: function, xdata, ydata,p0
    #
    # coeff, var_matrix = curve_fit(gauss.mult_gaussFun_Fit, (bin_centers_x, bin_centers_y), H, p0=p0)

    # plt.figure()
    # plt.scatter(ms, rs)
    #
    # plt.figure()
    # plt.hist(ms, 100)
    #
    # plt.figure()
    # plt.hist(rs, 100)

    return parameter_dict





#
# t0 = time.time()
# from matchpoint.geometricHashing import pointHash, findMatch
# ht = pointHash(destination, bases='all', magnificationRange=[0,10], rotationRange=[-np.pi,np.pi])
# matched_bases = findMatch(source, ht, bases='all', magnificationRange=[0,1], rotationRange=[-np.pi,np.pi])
# source_coordinate_tuple = source[matched_bases['testBasis']]
# destination_coordinate_tuple = destination[matched_bases['hashTableBasis']]
# t1 = time.time()
# plt.close('all')


# centers = np.array([(destination[pair[0]]+destination[pair[1]])/2 for pair in pairs])
#
# centers_KDTree = KDTree(centers)
#
# # Points within the circle containing the two original points in the pair
# indices_in_between_pairs = centers_KDTree.query_ball_tree(destination_KDTree, r=distance/2*0.8)


# def generate_point_tuples(point_set_KDTree, maximum_distance, tuple_size):
#     pairs = list(point_set_KDTree.query_pairs(maximum_distance))
#
#     point_tuples = []
#
#     for pair in pairs:
#         pair_coordinates = point_set_KDTree.data[list(pair)]
#         center = (pair_coordinates[0] + pair_coordinates[1]) / 2
#         distance = np.linalg.norm(pair_coordinates[0] - pair_coordinates[1])
#         internal_points = point_set_KDTree.query_ball_point(center, distance / 2 * 0.99)
#         if len(internal_points) >= (tuple_size - 2):
#             random.shuffle(internal_points)
#             internal_points = internal_points[0:(tuple_size - 2)]
#             #point_tuples.append(pair + tuple(internal_points))
#             yield pair + tuple(internal_points)
#
#     #return point_tuples

def generate_point_tuples(point_set_KDTree, maximum_distance, tuple_size):
    pairs = list(point_set_KDTree.query_pairs(maximum_distance))

    point_tuples = []

    for pair in pairs:
        pair_coordinates = point_set_KDTree.data[list(pair)]
        center = (pair_coordinates[0] + pair_coordinates[1]) / 2
        distance = np.linalg.norm(pair_coordinates[0] - pair_coordinates[1])
        internal_points = point_set_KDTree.query_ball_point(center, distance / 2 * 0.99)
        for internal_point_combination in itertools.combinations(internal_points, tuple_size-2):
            point_tuples.append(pair + tuple(internal_point_combination))
            # yield pair + tuple(internal_point_combination)

    return point_tuples

def geometric_hash_table(point_set_KDTree, point_tuples):
    pt = np.array(point_tuples)
    d = point_set_KDTree.data[pt]
    d0 = np.array([[0,0],[1,1]])


    T = np.repeat(np.expand_dims(np.diag([1.,1.,1.],k=0), axis=0), len(d), axis=0)
    T[:, :2, 2] = d0[0] - d[:, 0, :]


    diffs_d = d[:,1,:] - d[:,0,:]
    diffs_d0 = d0[1] - d0[0]
    m_d = np.linalg.norm(diffs_d, axis=1)
    m_d0 = np.linalg.norm(diffs_d0)

    unit_diffs_d = np.divide(diffs_d, m_d[:, None])
    unit_diffs_d0 = diffs_d0 / m_d0

    D = np.repeat(np.expand_dims(np.diag([1., 1., 1.], k=0), axis=0), len(d), axis=0)
    D[:,0,0] = D[:,1,1] = 1 / m_d * m_d0

    dot_product = np.dot(unit_diffs_d, unit_diffs_d0) # for mapping to (0,0),(1,1) could be replaced by sum to increase speed
    cross_product = np.cross(unit_diffs_d, unit_diffs_d0) # for mapping to (0,0),(1,1) could be replaced by subtraction to increase speed
    R = np.repeat(np.expand_dims(np.diag([1., 1., 1.], k=0), axis=0), len(d), axis=0)
    R[:,0,0] = R[:,1,1] = dot_product
    R[:,1,0] = cross_product
    R[:,0,1] = -cross_product

    transformation_matrices = R@D@T

    # transformation_matrices @ np.dstack([d, np.ones((d.shape[0], d.shape[1], 1))]).swapaxes(1,2)
    hash_code = transformation_matrices[:,:2,:] @ np.dstack([d[:,len(d0):,:], np.ones((d.shape[0], d.shape[1]-len(d0), 1))]).swapaxes(1, 2)

    # Break similarity of pair
    break_similarity = hash_code.swapaxes(1,2)[:, :, 0].sum(axis=1) > ((pt.shape[1] - d0.shape[0]) / 2)
    # this is not yet suited for using more than two basis points
    switch_basis_points_matrix = np.array([[-1,0,1],[0,-1,1],[0,0,1]])
    transformation_matrices[break_similarity] = switch_basis_points_matrix[None,:,:]@transformation_matrices[break_similarity]
    hash_code[break_similarity] = transformation_matrices[break_similarity, :2, :] @ np.dstack(
        [d[break_similarity, len(d0):, :], np.ones((d[break_similarity].shape[0], d.shape[1] - len(d0), 1))]).swapaxes(1, 2)

    hash_code = hash_code.swapaxes(1, 2)

    # Break similarity of internal points
    s = hash_code[:, :, 0].argsort()
    hash_code = np.take_along_axis(hash_code, s[:, :, None], axis=1)

    return hash_code.reshape(len(hash_code), -1), transformation_matrices

    # x = np.hstack([d[0][0:2], [[1],[1]]])
    # y = np.hstack([d0, [[1],[1]]])
    #
    # x = np.vstack([x, x[1]-x[0]])
    # y = np.vstack([y, y[1]-y[0]])
    #
    # A = np.linalg.lstsq(x.T, y.T, rcond=None)[0]
    #
    # scale = np.sqrt(x * x).sum(axis=0)
    # coeff, r, rank, s = np.linalg.lstsq(x / scale, y[:, 0:2], rcond=None)
    # A = (coeff.T / scale).T

### Old version
# def geometric_hash_table(point_set_KDTree, point_tuples, tuple_size):
#     #hash_table = []
#     for point_tuple in point_tuples:
#
#         pair = point_tuple[:2]
#         internal_points = point_tuple[2:]
#         pair_coordinates = point_set_KDTree.data[list(pair)]
#         internal_coordinates = point_set_KDTree.data[list(internal_points)]
#         # Sort internal points based on x coordinate
#         # internal_point_order = np.argsort(internal_coordinates[:, 0])
#         # internal_points = [internal_points[i] for i in internal_point_order]
#         # internal_coordinates = internal_coordinates[internal_point_order]
#
#         end_points = np.array([[0, 0], [1, 1]])
#         hash_coordinates = mapToPoint(internal_coordinates, pair_coordinates, end_points)
#         # Break similarity of pair
#         if np.sum(hash_coordinates[:, 0]) > ((tuple_size - 2) / 2):
#             pair = pair[::-1]
#             pair_coordinates = pair_coordinates[::-1]
#             hash_coordinates = mapToPoint(internal_coordinates, pair_coordinates, end_points)
#
#         # Break similarity of internal points
#         internal_point_order = np.argsort(hash_coordinates[:, 0])
#         internal_points = [internal_points[i] for i in internal_point_order]
#         internal_coordinates = internal_coordinates[internal_point_order]
#         hash_coordinates = hash_coordinates[internal_point_order]
#
#         point_tuple = pair + tuple(internal_points)
#         hash_code = hash_coordinates.flatten()
#         # hash_table.append(hash_code)
#
#         yield point_tuple, hash_code
#
#         # plot_tuple(np.vstack([pair_coordinates, internal_coordinates])
#         # plot_tuple(np.vstack([end_points, hash_coordinates]))

def geometric_hash(point_sets, maximum_distance=100, tuple_size=4):
    # TODO: Add minimum_distance and implement
    # TODO: Make invariant to mirroring
    # TODO: Make usable with multiple point-sets in a single hash table
    # TODO: Implement names of point_sets, possibly through a dictionary and adding a attribute to each KDtree
    start_time = time.time()

    if type(point_sets) is not list:
        point_sets = [point_sets]

    point_set_KDTrees = [cKDTree(point_set) for point_set in point_sets]

    hash_tables = []
    point_tuple_sets = []
    for point_set_KDTree in point_set_KDTrees:
        point_tuples = generate_point_tuples(point_set_KDTree, maximum_distance, tuple_size)
        hash_table, transformation_matrices = geometric_hash_table(point_set_KDTree, point_tuples)

        hash_tables.append(hash_table)
        point_tuple_sets.append(point_tuples)

    hash_table_KDTree = cKDTree(np.vstack(hash_tables))

    print("--- %s seconds ---" % (time.time() - start_time))

    return point_set_KDTrees, point_tuple_sets, hash_table_KDTree, transformation_matrices

### Old version
# def geometric_hash(point_sets, maximum_distance=100, tuple_size=4):
#     # TODO: Add minimum_distance and implement
#     # TODO: Make invariant to mirroring
#     # TODO: Make usable with multiple point-sets in a single hash table
#     # TODO: Implement names of point_sets, possibly through a dictionary and adding a attribute to each KDtree
#     start_time = time.time()
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
#     print("--- %s seconds ---" % (time.time() - start_time))
#
#     return point_set_KDTrees, point_tuple_sets, hash_table_KDTree




def find_match_after_hashing(source, maximum_distance_source, tuple_size, source_vertices,
                             destination_KDTrees, destination_tuple_sets, destination_hash_table_KDTree,
                             hash_table_distance_threshold=0.01,
                             alpha=0.1, test_radius=10, K_threshold=10e9,
                             magnification_range=None, rotation_range=None):

    if type(destination_KDTrees) is not list:
        destination_KDTrees = [destination_KDTrees]

    source_KDTree = cKDTree(source)
    source_tuple_generator = generate_point_tuples(source_KDTree, maximum_distance_source, tuple_size)

    hash_table_distances_checked = 0
    tuples_checked = 0
    # for source_tuple_index in np.arange(len(source_tuples)):
    for source_tuple, source_hash_code in geometric_hash_table(source_KDTree, source_tuple_generator, tuple_size):
        # TODO: Get all destination tuple indices within a range
        distance, destination_tuple_index = destination_hash_table_KDTree.query(source_hash_code)

        # We can also put a threshold on the distance here possibly
        #print(distance)
        hash_table_distances_checked += 1
        # if hash_table_distances_checked > 500:
        #     return

        if distance < hash_table_distance_threshold:
            # source_coordinate_tuple = source[list(source_tuples[source_tuple_index])]
            # destination_coordinate_tuple = destination[list(destination_tuples[destination_tuple_index])]

            # source_tuple = source_tuples[source_tuple_index]
            # Find the destination tuple by determining in which destination pointset the destination tuple index is located
            # and by determining what the index is within that destination pointset.
            cumulative_tuples_per_destination = np.cumsum([0]+[len(point_tuples) for point_tuples in destination_tuple_sets])
            destination_index = np.where((cumulative_tuples_per_destination[:-1] <= destination_tuple_index) &
                                         (cumulative_tuples_per_destination[1:] > destination_tuple_index))[0][0]
            tuple_index_in_destination_set = destination_tuple_index - cumulative_tuples_per_destination[destination_index]
            destination_tuple = destination_tuple_sets[destination_index][tuple_index_in_destination_set]

            #Or list(itertools.chain.from_iterable(destination_tuple_sets))[destination_tuple_index]

            tuples_checked += 1

            destination_KDTree = destination_KDTrees[destination_index]
            found_transformation = tuple_match(source, destination_KDTree, source_vertices, source_tuple, destination_tuple,
                                alpha, test_radius, K_threshold, magnification_range, rotation_range)
            if found_transformation:
                match = mp.MatchPoint(source=source, destination=destination_KDTree.data, method='Geometric hashing',
                                 transformation_type='linear', initial_transformation=None)
                match.transformation = found_transformation
                match.destination_index = destination_index

                match.hash_table_distance = distance
                match.hash_table_distances_checked = hash_table_distances_checked
                match.tuples_checked = tuples_checked
                return match

def polygon_area(vertices):
    x = vertices[:,0]
    y = vertices[:,1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def tuple_match(source, destination_KDTree, source_vertices, source_tuple, destination_tuple,
                alpha=0.1, test_radius=10, K_threshold=10e9, scaling_range=None, rotation_range=None):
    source_coordinate_tuple = source[list(source_tuple)]
    destination_coordinate_tuple = destination_KDTree.data[list(destination_tuple)]

    source_indices_without_tuple = [i for i in range(len(source)) if i not in source_tuple]
    source = source[source_indices_without_tuple]

    source_transformed, transformation_matrix = mapToPoint(source, source_coordinate_tuple[:2], destination_coordinate_tuple[:2], returnTransformationMatrix=True)
    # mp.icp.scatter_coordinates([source_transformed])

    found_transformation = AffineTransform(transformation_matrix)

    if rotation_range:
        rotation = found_transformation.rotation/(2*np.pi)*360
        if not (rotation_range[0] < rotation < rotation_range[1]):
            return
    if scaling_range:
        scaling = np.mean(found_transformation.scale)
        if not (scaling_range[0] < scaling < scaling_range[1]):
            return

    # if not (-1 < rot < 1) & (3.3 < sca < 3.4):
    #     return
    # if (-1 < rot < 1) & (3.3 < sca < 3.4):
    #     print('found')
    # mp.icp.scatter_coordinates([destination_KDTree.data, source_transformed])

    source_vertices_transformed = found_transformation(source_vertices)
    destination_cropped = mp.point_set.crop_coordinates(destination_KDTree.data, source_vertices_transformed)

    #source_transformed_area = np.linalg.norm(source_vertices_transformed[0] - source_vertices_transformed[1])
    # source_transformed_area = np.abs(np.cross(source_vertices_transformed[1] - source_vertices_transformed[0],
    #                                           source_vertices_transformed[3] - source_vertices_transformed[0]))

    source_transformed_area = polygon_area(source_vertices_transformed)

    # pDB = 1 / source_transformed_area
    # pDB = 1 / len(destination_cropped)
    # pDB = 1
    # pDB = 1 / len(source)
    pDB = 1 / source_transformed_area

    # alpha = 0.1
    # # test_radius = 5
    # test_radius = 10
    K=1
    # K_threshold = 10e9
    # bayes_factor6 = []
    for coordinate in source_transformed:
        # points_within_radius = destination_KDTree.query_ball_point(coordinate, test_radius)
        # pDF = alpha/source_transformed_area + (1-alpha)*len(points_within_radius)/len(destination_cropped)
        # pDF = alpha/len(destination_cropped) + (1-alpha)*len(points_within_radius)/len(destination_cropped)
        # pDF = alpha + (1-alpha)*len(points_within_radius)
        # pDF = alpha/len(source) + (1-alpha)*len(points_within_radius)/len(destination_cropped)
        # pDF = alpha/source_transformed_area + (1-alpha)*len(points_within_radius)/(len(destination_cropped)*np.pi*test_radius**2)

        sigma = test_radius
        # sigma = 10
        distance, index = destination_KDTree.query(coordinate)
        # pDF = alpha/source_transformed_area + (1-alpha)/(2*np.pi*sigma**2)*np.exp(-distance**2/(2*sigma**2))/len(destination_cropped)
        # 2d Gaussian
        pDF = alpha / source_transformed_area + \
              (1 - alpha) / (2 * np.pi * sigma ** 2) * \
              np.exp(-(distance ** 2) / (2 * sigma ** 2)) / len(destination_cropped)

        # print(len(points_within_radius))
        # bayes_factor6.append(pDF/pDB)
        K = K * pDF/pDB
        if K > K_threshold:
            print("Found match")
            return found_transformation

    # print(pDF)
    # print(K)



def find_match(source, destination, source_vertices,
               tuple_size=4, maximum_distance_source=4, maximum_distance_destination=40,
               hash_table_distance_threshold = 0.01,
               alpha = 0.1, test_radius = 10, K_threshold = 10e9
            ):
    # destination_KDTree, destination_tuples, destination_hash_table_KDTree = geometric_hash(destination, 40, 4)
    # source_KDTree, source_tuples, source_hash_table_KDTree = geometric_hash(source, 4, 4)
    # 200 points 200,20
    # 10000 points 10,1
    # return find_match_after_hashing(*geometric_hash(source, 4, 4), source_vertices, *geometric_hash(destination, 40, 4))


    return find_match_after_hashing(source, maximum_distance_source, tuple_size, source_vertices,
                                    *geometric_hash(destination, maximum_distance_destination, tuple_size),
                                    hash_table_distance_threshold,
                                    alpha, test_radius, K_threshold)


if __name__ == '__main__':
    from papylio.plugins.sequencing.point_set_simulation import simulate_mapping_test_point_set

    # Simulate source and destination point sets
    number_of_source_points = 4000
    transformation = AffineTransform(translation=[128, 128], rotation=0 / 360 * 2 * np.pi, scale=[1, 1])
    source_bounds = np.array([[0, 0], [512, 512]])
    source_crop_bounds = np.array([[0, 0], [50, 50]])
    fraction_missing_source = 0
    fraction_missing_destination = 0
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
    perfect.show_mapping_transformation(show_source=True)

    # mp.icp.scatter_coordinates([source, destination])

    source_vertices = np.array([source_crop_bounds[0], source_crop_bounds.T[0],
                                source_crop_bounds[1], np.flip(source_crop_bounds.T[1])])




    # destination_KDTree, destination_tuples, destination_hash_table_KDTree = geometric_hash(destination, 3000, 4)
    # source_KDTree, source_tuples, source_hash_table_KDTree = geometric_hash(source, 100, 4)





    hash_table_distance_threshold = 0.01
    alpha = 0.5
    test_radius = 10
    K_threshold = 10e9

    # match = find_match_after_hashing(source, maximum_distance_source, tuple_size, source_vertices,
    #                                  *destination_hash_data2,
    #                                  hash_table_distance_threshold=0.01,
    #                                  alpha=0.1, test_radius=10, K_threshold=10e9)

    maximum_distance_destination = 20
    maximum_distance_source = 20


    # ht = GeometricHashTable(destinations, source_vertices, initial_source_transformation=AffineTransform(),
    #                  number_of_source_bases=20, number_of_destination_bases='all',
    #                  tuple_size=4, maximum_distance_source=maximum_distance_source, maximum_distance_destination=maximum_distance_destination)
    #
    #
    # # test = ht.query_tuple_transformations(source, hash_table_distance_threshold=0.1, parameters=['rotation'], bins=50)
    # plt.figure()
    # t = ht.test(source, hash_table_distance_threshold=0.01, bins=25)




    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import numpy as np

    triang_source = tri.Triangulation(source[:, 0], source[:, 1])

    triang_destination = tri.Triangulation(destination[:, 0], destination[:, 1])





    def angles_from_triangles(pointset, triangles):
        triangle_points = pointset[triangles.triangles]

        side_lengths = np.linalg.norm(triangle_points[:, [0, 1, 2], :] - triangle_points[:, [1, 2, 0], :], axis=2)

        a = side_lengths[:, 0]
        b = side_lengths[:, 1]
        c = side_lengths[:, 2]

        # R = a * b * c / np.sqrt((a + b + c) * (a - b + c) * (a + b - c) * (b + c - a))
        # angles = np.arcsin(side_lengths / (2 * R[:, np.newaxis]))

        A = np.arccos((-a ** 2 + b ** 2 + c ** 2) / (2 * b * c))
        B = np.arccos((a ** 2 - b ** 2 + c ** 2) / (2 * c * a))
        C = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))

        A = (-a ** 2 + b ** 2 + c ** 2) / (2 * b * c)
        B = (a ** 2 - b ** 2 + c ** 2) / (2 * c * a)
        C = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)

        angles = np.vstack([A, B, C]).T

        return angles

    angles_source = angles_from_triangles(source, triang_source)#/(2*np.pi)*360
    angles_destination = angles_from_triangles(destination, triang_destination)#/(2*np.pi)*360

    angles_source.sort()
    angles_destination.sort()

    hash_table_source = cKDTree(angles_source[:, :2])
    hash_table_destination = cKDTree(angles_destination[:,:2])

    matches = hash_table_source.query_ball_tree(hash_table_destination, 0.01)

    [len(i) for i in test]

    triang_source.triangles[51]

    triangle_index = 0

    source_neighbours = triang_source.neighbors[triangle_index]
    source_neighbours_in_destination = [matches[source_triangle_index] if source_triangle_index >= 0 else [] for source_triangle_index in source_neighbours]
    destination_triangles = matches[triangle_index]
    destination_neighbours = triang_destination.neighbors[destination_triangles]

    t2 = np.array(
        [[dn in source_neighbours_in_destination[i] for dn in destination_neighbours[:, i]] for i in range(3)]).T

    # fig1 = plt.figure()
    fig1.clf()
    plot_triangles(triang_source, highlight=[0], figure=fig1)
    plot_triangles(triang_source, highlight=source_neighbours,
                   highlight_kwargs={'c': 'r'}, figure=fig1)

    # fig2 = plt.figure()
    fig2.clf()
    plot_triangles(triang_destination, highlight=matches[0], figure=fig2)
    plot_triangles(triang_destination, highlight=destination_neighbours.flatten(),
                   highlight_kwargs={'c': 'r'}, figure=fig2)

    plot_triangles(triang_destination, highlight=np.hstack(source_neighbours_in_destination),
                   highlight_kwargs={'c': 'k', 'marker': 'o'}, figure=fig2)




    t2s = []
    t2cs = []
    for triangle_index, _ in enumerate(triang_source.triangles):
        source_neighbours = triang_source.neighbors[triangle_index]
        source_neighbours_in_destination = [matches[i] if i >= 0 else [] for i in source_neighbours]
        destination_neighbours = triang_destination.neighbors[matches[triangle_index]]

        t2 = np.array([[dn in source_neighbours_in_destination[i] for dn in destination_neighbours[:, i]] for i in range(3)]).T

        t2 = t2.sum(axis=1)
        t2u = np.unique(t2, return_counts=True)
        t2c = np.zeros(4).astype(int)
        print(t2u)
        t2c[t2u[0]] = t2u[1]
        print(t2)
        t2cs.append(t2c)
        if t2c[3]==1:
            t2s.append((triangle_index, test[triangle_index][np.where(t2==3)[0][0]]))
    t2s = np.vstack(t2s)
    t2cs = np.array(t2cs)


    tr = np.array(t2s)

    mask = np.ones(len(triang_source.triangles)).astype(bool)
    mask[tr[:,0]] = False

    triang_source.set_mask(mask)

    mask = np.ones(len(triang_destination.triangles)).astype(bool)
    mask[tr[:,1]] = False

    triang_destination.set_mask(mask)




    fig2, ax2 = plt.subplots()
    ax2.set_aspect('equal')
    ax2.triplot(triang_destination, 'o-', lw=1)



    def plot_triangles(triangulation, highlight=[], highlight_kwargs={'c': 'b'}, figure=None):
        if not figure:
            figure = plt.figure()
        ax = figure.gca()
        ax.set_aspect('equal')
        ax.triplot(triangulation, '-', lw=1)

        triangle_indices = highlight
        point_indices = triangulation.triangles[triangle_indices]
        x_mean = triangulation.x[point_indices].mean(axis=1)
        y_mean = triangulation.y[point_indices].mean(axis=1)
        ax.scatter(x_mean, y_mean, **highlight_kwargs)

        return figure


    match.show_mapping_transformation(show_source=True)

    match = find_match_after_hashing(*source_hash_data, source_vertices, *destination_hash_data)

    match = find_match(source, destination, source_vertices)
    if match:
        mp.icp.scatter_coordinates([source, destination, match(source), source_vertices, match(source_vertices)])







    #
    # import papylio as pp
    # exp = pp.Experiment(r'D:\SURFdrive\Promotie\Data\Sung hyun and Frank')
    #
    # import numpy as np
    # tile1101 = np.loadtxt(r'D:\SURFdrive\Promotie\Data\Sung hyun and Frank\1101.loc')
    # tile1102 = np.loadtxt(r'D:\SURFdrive\Promotie\Data\Sung hyun and Frank\1102.loc')
    #
    #
    # tuple_size = 4
    # maximum_distance_destination = 1000
    # tile1101_hash_data = geometric_hash2(tile1101, maximum_distance_destination, tuple_size)
    # tile1102_hash_data = geometric_hash2(tile1102, maximum_distance_destination, tuple_size)
    #
    # destination_hash_data = tile1102_hash_data
    # destination_KDTree, destination_tuples, destination_hash_table_KDTree, destination_transformation_matrices = destination_hash_data
    #
    # Ts = []
    # for file in exp.files:
    #
    #     maximum_distance_source = 200
    #     file_coordinates = file.coordinates[1::2]
    #     file_coordinates[:,0] = -file_coordinates[:,0]
    #     source_hash_data = geometric_hash2(file_coordinates, maximum_distance_source, tuple_size)
    #
    #     source_KDTree, source_tuples, source_hash_table_KDTree, source_transformation_matrices = source_hash_data
    #
    #
    #     htmatch = source_hash_table_KDTree.query_ball_tree(destination_hash_table_KDTree, 0.005)
    #
    #     Tss=[]
    #     for i, htm in enumerate(htmatch):
    #         sT = source_transformation_matrices[i]
    #         sTi = np.linalg.inv(sT)
    #         for dm in htm:
    #             dT = destination_transformation_matrices[dm]
    #             dTi = np.linalg.inv(dT)
    #             Tss.append(dTi@sT)
    #             print('appended')
    #     Ts.append(Tss)
    #
    # for Tss in Ts:
    #     Tss = np.stack(Tss)
    #
    #
    #     ms = np.sqrt(Tss[:, 0, 0] ** 2 + Tss[:, 1, 0]**2)
    #     plt.figure()
    #     h, b, _ = plt.hist(ms, 100)
    #
    # m = np.mean(b[h.argmax():h.argmax() + 2])
    #
    # sel = np.all(np.vstack([ms>2.4, ms<2.5]), axis=0)
    # test = Ts[sel]
    #
    # h, b, _ = plt.hist(np.arccos(test[:,0,0]/m), 100)
    # r = np.mean(b[h.argmax():h.argmax() + 2])
    #
    #
    #
    # trs=[]
    # for Tss in Ts:
    #     for tr in Tss:
    #         trs.append(AffineTransform(tr))
    #
    # ms = np.array([tr.scale[0] for tr in trs])
    # rs = np.array([tr.rotation for tr in trs])
    #
    # plt.scatter(ms, rs)
    #
    # plt.hist(ms, 100)
    # plt.hist(rs, 100)
    #
    # plt.close('all')
    # for file in exp.files:
    #     file_coordinates = file.coordinates[1::2]
    #     file_coordinates[:,0] = -file_coordinates[:,0]
    #     source_vertices = np.array([[256,0],[256,512],[512,512],[512,0]])
    #     match = find_match_after_hashing(file_coordinates, maximum_distance_source, tuple_size, source_vertices,
    #                                      *tile1101_hash_data,
    #                                      hash_table_distance_threshold=0.01,
    #                                      alpha=0.5, test_radius=50, K_threshold=10e9)
    #     if match is not None:
    #         match.show_mapping_transformation()
















    # Make random source and destination dataset
    # np.random.seed(42)
    # destination = np.random.rand(1000,2)*1000
    # destination2 = np.random.rand(1000, 2) * 1000
    # source_vertices_in_destination = np.array([[300, 300], [450, 300], [450, 600], [300,600]])
    #
    #
    # transformation = AffineTransform(scale=(0.1, 0.1), rotation=np.pi, shear=None, translation=(-100,350))
    # source = transformation(mp.point_set.crop_coordinates(destination, source_vertices_in_destination))
    # source_vertices = transformation(source_vertices_in_destination)
    # plt.figure()
    # plt.scatter(destination[:,0],destination[:,1])
    # plt.scatter(source[:,0],source[:,1])
    # mp.icp.scatter_coordinates([source,destination,mp.point_set.crop_coordinates(destination, source_vertices_in_destination)])
    #
    # # destination_KDTree, destination_tuples, destination_hash_table_KDTree = geometric_hash(destination, 40, 4)
    # # source_KDTree, source_tuples, source_hash_table_KDTree = geometric_hash(source, 4, 4)
    # #
    # match = find_match(source, [destination2, destination], source_vertices,
    #                    tuple_size=4, maximum_distance_source=4, maximum_distance_destination=40,
    #                    hash_table_distance_threshold = 0.01,
    #                    alpha = 0.1, test_radius = 10, K_threshold = 10e9)
    # plt.figure()
    # mp.icp.scatter_coordinates([source, destination, match(source), source_vertices,source_vertices_in_destination])



    # file_coordinates = np.loadtxt(
    #     r'C:\Users\Ivo Severins\Desktop\seqdemo\20190924 - Single-molecule setup (TIR-I)\16L\spool_6.pks')[:, 1:3]
    # tile_coordinates = np.loadtxt(r'C:\Users\Ivo Severins\Desktop\seqdemo\20190926 - Sequencer (MiSeq)\2102.loc')
    #
    # source = file_coordinates
    # destination = tile_coordinates
    # source_vertices = np.array([[1024,0],[2048,0],[2048,2048],[1024,2048]])
    #
    # # source = file_coordinates[[1,5,9,7]]
    # # destination = tile_coordinates[[73,82,85,84]]
    #
    # source[:,0] = -source[:,0]
    # source_vertices[:,0] = -source_vertices[:,0]
    #
    # # destination_KDTree, destination_tuples, destination_hash_table_KDTree = geometric_hash(destination, 3000, 4)
    # # source_KDTree, source_tuples, source_hash_table_KDTree = geometric_hash(source, 100, 4)
    #
    # tuple_size = 4
    # maximum_distance_source = 1000
    # maximum_distance_destination = 3000
    #
    # source_hash_data = geometric_hash(source, maximum_distance_source, tuple_size)
    # destination_hash_data = geometric_hash(destination, maximum_distance_destination, tuple_size)
    #
    # hash_table_distance_threshold = 0.01
    # alpha = 0.1
    # test_radius = 10
    # K_threshold = 10e9
    #
    # match = find_match_after_hashing(source, maximum_distance_source, tuple_size, source_vertices,
    #                                  *destination_hash_data,
    #                                  hash_table_distance_threshold=0.01,
    #                                  alpha=0.1, test_radius=10, K_threshold=10e9)
    #
    #
    # source_KDTree, source_tuples, source_hash_table_KDTree = source_hash_data
    # destination_KDTree, destination_tuples, destination_hash_table_KDTree = destination_hash_data
    #
    # #match = find_match_after_hashing(*source_hash_data, source_vertices, *destination_hash_data)
    #
    # #match = find_match(source, destination, source_vertices)
    # if match:
    #     mp.icp.scatter_coordinates([source, destination, match(source), source_vertices, match(source_vertices)])

    #
    #
    # def scatter_with_number(coordinates):
    #     # plot the chart
    #     plt.scatter(coordinates[:,0], coordinates[:,1])
    #
    #     # zip joins x and y coordinates in pairs
    #     for i, coordinate in enumerate(coordinates):
    #         label = str(i)
    #
    #         # this method is called for each point
    #         plt.annotate(label,  # this is the text
    #                      (coordinate[0], coordinate[1]),  # this is the point to label
    #                      textcoords="offset points",  # how to position the text
    #                      xytext=(0, 10),  # distance from text to points (x,y)
    #                      ha='center')
    #
    # scatter_with_number(source)
    # scatter_with_number(destination)
    #
    # for i, t in enumerate(source_tuples):
    #     if ((1 in t) & (5 in t) & (9 in t) & (7 in t)):
    #         print(i)
    #
    # for i, t in enumerate(destination_tuples):
    #     if ((73 in t) & (82 in t) & (85 in t) & (84 in t)):
    #         print(i)
    #
    # tuple_match(source, destination_KDTree, source_vertices, source_tuples[1116], destination_tuples[5365])
    #
    # source_hash_code = source_hash_table_KDTree.data[1116]
    #
    # distance, destination_tuple_index = destination_hash_table_KDTree.query(source_hash_code)

    #
    #
    # def show_match(self, match, figure = None, view='destination'):
    #     if not figure: figure = plt.gcf()
    #     figure.clf()
    #     ax = figure.gca()
    #
    #     #ax.scatter(ps2[:,0],ps2[:,1],c='g',marker = '+')
    #
    #     ax.scatter(match.destination[:,0],match.destination[:,1], marker = '.', facecolors = 'k', edgecolors='k')
    #     ax.scatter(match.transform_source_to_destination[:,0],match.transform_source_to_destination[:,1],c='r',marker = 'x')
    #
    #     destination_basis_index = match.best_image_basis['hashTableBasis']
    #     source_basis_index = match.best_image_basis['testBasis']
    #     ax.scatter(match.destination[destination_basis_index, 0], match.destination[destination_basis_index, 1], marker='.', facecolors='g', edgecolors='g')
    #     ax.scatter(match.transform_source_to_destination[source_basis_index, 0], match.transform_source_to_destination[source_basis_index, 1], c='g',
    #                marker='x')
    #
    #     ax.set_aspect('equal')
    #     ax.set_title('Tile:' + self.tile.name +', File: ' + str(self.files[self.matches.index(match)].relativeFilePath))
    #
    #     if view == 'source':
    #         maxs = np.max(match.transform_source_to_destination, axis=0)
    #         mins = np.min(match.transform_source_to_destination, axis=0)
    #         ax.set_xlim([mins[0], maxs[0]])
    #         ax.set_ylim([mins[1], maxs[1]])
    #     elif view == 'destination':
    #         maxs = np.max(match.destination, axis=0)
    #         mins = np.min(match.destination, axis=0)
    #         ax.set_xlim([mins[0], maxs[0]])
    #         ax.set_ylim([mins[1], maxs[1]])
    #         # ax.set_xlim([0, 31000])
    #         # ax.set_ylim([0, 31000])
    #
    #     name = str(self.files[self.matches.index(match)].relativeFilePath)
    #     print(name)
    #     n = name.replace('\\', '_')
    #
    #     figure.savefig(self.dataPath.joinpath(n + '_raw.pdf'), bbox_inches='tight')
    #     figure.savefig(self.dataPath.joinpath(n + '_raw.png'), bbox_inches='tight', dpi=1000)
    #
    # def loop_through_matches(self, figure=plt.figure()):
    #     plt.ion()
    #     for match in self.matches:
    #         self.show_match(match, figure=figure)
    #         plt.show()
    #         plt.pause(0.001)
    #         input("Press enter to continue")





# destination_coordinate_tuples = [destination[list(t)] for t in destination_tuples]
# source_coordinate_tuples = [source[list(t)] for t in source_tuples]
# source_coordinate_tuple = source_coordinate_tuples[source_tuple_index]
# destination_coordinate_tuple = destination_coordinate_tuples[destination_tuple_index]


#
# mp.icp.scatter_coordinates([source_coordinate_tuple, destination_coordinate_tuple])
#
# def connect_pairs(pairs):
#     for pair in pairs:
#         plt.plot(pair[:,0], pair[:,1], color='r')
#
# def plot_tuple(tuple_coordinates):
#     pair_coordinates = tuple_coordinates[:2]
#     internal_coordinates = tuple_coordinates[2:]
#     plt.figure()
#     center = (pair_coordinates[0] + pair_coordinates[1]) / 2
#     distance = np.linalg.norm(pair_coordinates[0] - pair_coordinates[1])
#     connect_pairs([pair_coordinates])
#     mp.icp.([pair_coordinates, np.atleast_2d(center), internal_coordinates])
#     plt.gca().set_aspect('equal')
#     circle = plt.Circle(center, distance / 2, fill=False)
#     plt.gcf().gca().add_artist(circle)
#     axis_limits = np.array([center-distance/2*1.2, center+distance/2*1.2])
#     plt.xlim(axis_limits[:, 0])
#     plt.ylim(axis_limits[:, 1])
#     plt.xlabel('x')
#     plt.ylabel('y')







