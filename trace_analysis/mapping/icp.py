import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.neighbors import NearestNeighbors
from skimage.transform import AffineTransform, PolynomialTransform

from trace_analysis.plotting import scatter_coordinates, show_point_connections
from trace_analysis.mapping.polywarp import PolywarpTransform

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    #assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def nearest_neighbor_pair(pointset1, pointset2):
    distances2, indices2 = nearest_neighbor(pointset1, pointset2)
    distances1, indices1 = nearest_neighbor(pointset2, pointset1)

    i1 = indices1[indices2[indices1] == np.arange(len(indices1))]
    i2 = np.where(indices2[indices1] == np.arange(len(indices1)))[0]

    return distances1[i2], i1, i2


def direct_match(source, destination, transform=AffineTransform, return_inverse=False, **kwargs):
    transformation = transform()
    transformation.estimate(source, destination, **kwargs)

    if return_inverse:
        transformation_inverse = transform()
        transformation_inverse.estimate(destination, source, **kwargs)
    else:
        transformation_inverse = None

    error = mean_squared_error(source, destination, transformation)

    return transformation, transformation_inverse, error


def nearest_neighbour_match(source, destination, transform=AffineTransform, initial_transformation=None,
                            distance_threshold=None, return_inverse=False, **kwargs):

    if initial_transformation:
        source_after_initial_transformation = initial_transformation(source)
    else:
        source_after_initial_transformation = source

    if distance_threshold == 'auto':
        auto_distance_threshold = True
    else:
        auto_distance_threshold = False

    distances, source_indices, destination_indices = \
        nearest_neighbor_pair(source_after_initial_transformation, destination)

    if auto_distance_threshold:
        distance_threshold = np.median(distances) + np.std(distances)

    if type(distance_threshold) in (float, int):
        source_indices = source_indices[distances < distance_threshold]
        destination_indices = destination_indices[distances < distance_threshold]

    transformation, transformation_inverse, error = direct_match(source[source_indices], destination[destination_indices],
                                                                 transform, return_inverse=return_inverse, **kwargs)

    return transformation, transformation_inverse, source_indices, destination_indices, error


def mean_squared_error(source, destination, transformation):
    distances = np.linalg.norm(destination - transformation(source), axis=1)
    return np.mean(distances**2)


def icp(source, destination, max_iterations=20, tolerance=0.001, distance_threshold=None, distance_threshold_final=None,
        initial_transformation=None, transform=AffineTransform, transform_final=None, show_plot=False, **kwargs):
    """Iterative closest point algorithm for mapping a source point set on a destination point set.

    Parameters
    ----------
    source : Nx2 numpy.ndarray
        Coordinates of the source point set
    destination : Nx2 numpy.ndarray
        Coordinates of the destination point set
    max_iterations : int
        Maximum number of iterations to be performed by the algorithm.
    tolerance : float
        If difference in mean squared error between two iterations is smaller than the tolerance,
        the algorithm finishes.
    distance_threshold : float or None
        If set only nearest-neighbours with distances smaller than the distance_threshold will be used for estimating
        the transformation.
        If set to 'auto' the distance_threshold value will be calculated each iteration by taking the sum of the median
        and the standard deviation of the distances between the nearest neighbours.
    distance_threshold_final : float or None
        Distance threshold applied in the final match.
        If set to None, then the regular distance threshold will be used.
    initial_transformation : skimage.transform._geometric.GeometricTransform
        Initial transformation to apply to the source before start of iterations.
        If no initial_transformation is given then an initial transformation is calculated so that the centers of
        the source and destination coincide.
    transform : type
        Transform type used during iteration.
        Note: the class is passed, not the instance.
    transform_final : type or None
        Transform type used during for final estimation.
        If no value is given for transform_final then transform will be used.
        Note: the class is passed, not the instance.
    show_plot : bool
        If True an interactive plot will open that shows the pointsets in the different iterations.
    kwargs
        Additional keyword arguments are passed to the kwargs are passed to the transform.estimate.
        This should be used when using PolynomialTransform and the order value needs to be changed from the default value.

    Returns
    -------
    transformation_final : skimage.transform._geometric.GeometricTransform
        Obtained transformation
    transformation_final_inverse : skimage.transform._geometric.GeometricTransform
        Obtained inverse transformation.
    error
        Mean squared error of the neirest-neighbour pairs used for estimating the transformation.
    i : int
        Number of performed iterations

    """

    if transform_final is None:
        transform_final=transform
    if distance_threshold_final is None:
        distance_threshold_final = distance_threshold

    # Initialize plotting object
    plot = icp_plot()
    plot.append_data(source, destination, title='Start')

    if initial_transformation is None:
        # Initial translation to overlap both point-sets
        initial_transformation = AffineTransform(translation=(np.mean(destination, axis=0) - np.mean(source, axis=0)))
    current_transformation = initial_transformation

    plot.append_data(current_transformation(source), destination, title='Initial transformation')

    previous_error = 0

    for i in range(max_iterations):
        current_transformation, _, source_indices, destination_indices, error = \
            nearest_neighbour_match(source, destination, transform, current_transformation,
                                    distance_threshold, return_inverse=False, **kwargs)

        if show_plot:
            plot.append_data(current_transformation(source), destination, source_indices, destination_indices, error,
                             title=f'Iteration {i}')

        print(f'Iteration: {i} \t Mean squared error: {error} \t Number of pairs: {len(source_indices)}')

        if np.abs(previous_error - error) < tolerance:
            break
        previous_error = error

    # Perform final transformation, possibly with a different transformation type
    transformation_final, transformation_final_inverse, source_indices, destination_indices, error = \
        nearest_neighbour_match(source, destination, transform_final, current_transformation, distance_threshold_final,
                                return_inverse=True, **kwargs)

    print(f'Final \t\t Mean squared error: {error} \t Number of pairs: {len(source_indices)}')

    if show_plot:
        plot.append_data(transformation_final(source), destination, source_indices, destination_indices, error,
                         title=f'Final')
        plot.plot()

    return transformation_final, transformation_final_inverse, error, i


class icp_plot:
    def __init__(self):
        self.data = []
        self.xlim = [0,512]
        self.ylim = [0,512]

    def append_data(self, source, destination, source_indices=None, destination_indices=None, error=None,
                    title=None):
        self.data.append((source, destination, source_indices, destination_indices, error, title))

    def set_step(self, step_index):
        step_index = int(step_index)
        self.axis.cla()
        self.axis.set_xlim(self.xlim)
        self.axis.set_ylim(self.ylim)
        plot_icp_step(*self.data[step_index], self.axis)
        self.figure.canvas.draw_idle()
        # plt.title('#donor=' + str(len(source)) + '#acceptor=' + str(len(destination)) + '#overlap=' + str(
        #     len(source_indices)))

    def plot(self):
        self.figure, self.axis = plt.subplots()
        plt.subplots_adjust(bottom=0.15)
        self.axis.set_aspect(1)
        axcolor = 'lightgoldenrodyellow'
        slider_axis = plt.axes([0.1, 0.05, 0.8, 0.025], facecolor=axcolor)

        self.slider = Slider(slider_axis, 'Step', 0, len(self.data)-1,
                             valinit=len(self.data), valstep=1)

        self.slider.on_changed(self.set_step)

        self.set_step(len(self.data)-1)

def plot_icp_step(source, destination, source_indices=None, destination_indices=None,
                  error=None, title=None, axis=None):
    if not axis:
        axis = plt.gca()
    if (source_indices is not None) and (destination_indices is not None):
        show_point_connections(source[source_indices], destination[destination_indices], axis)
    scatter_coordinates([destination, source], axis)
    axis.set_title(f'{title}\nMean squared error: {error}')


if __name__ == '__main__':
    from trace_analysis.plugins.sequencing.point_set_simulation import simulate_mapping_test_point_set

    # Simulate soure and destination point sets
    number_of_source_points = 40
    transformation = AffineTransform(translation=[256,0], rotation=5/360*2*np.pi, scale=[0.98, 0.98])
    source_bounds = ([0, 0], [256, 512])
    source_crop_bounds = None
    fraction_missing_source = 0.1
    fraction_missing_destination = 0.1
    maximum_error_source = 4
    maximum_error_destination = 4
    shuffle = True

    source, destination = simulate_mapping_test_point_set(number_of_source_points, transformation,
                                                          source_bounds, source_crop_bounds,
                                                          fraction_missing_source, fraction_missing_destination,
                                                          maximum_error_source, maximum_error_destination, shuffle)

    # Perform icp on the simulated point sets
    max_iterations = 20
    tolerance = 0.0000001
    cutoff = None
    cutoff_final = 10
    initial_transformation = None
    transform = AffineTransform
    transform_final = PolywarpTransform

    transformation, transformation_inverse, distances, i = icp(source, destination, max_iterations, tolerance,
                                                               cutoff, cutoff_final, initial_transformation,
                                                               transform, transform_final, show_plot=True)
