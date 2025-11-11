import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

def coordinates_within_margin_selection(coordinates,  image = None, bounds = None, margin=10):
    if coordinates.size == 0:
        return np.array([])

    if image is not None:
        bounds = np.array([[0,0], [image.shape[1],image.shape[0]]])

    criteria = np.array([(coordinates[:, 0] > (bounds[0,0] + margin)),
                         (coordinates[:, 0] < (bounds[1,0] - margin)),
                         (coordinates[:, 1] > (bounds[0,1] + margin)),
                         (coordinates[:, 1] < (bounds[1,1] - margin))
                         ])

    return criteria.all(axis=0)

def coordinates_within_margin_selection(coordinates,  image = None, bounds = None, margin=10):
    if coordinates.size == 0:
        return np.array([])

    if image is not None:
        bounds = np.array([[0,0], [image.shape[1],image.shape[0]]])

    if isinstance(margin, int):
        margin = np.array([margin, margin])

    criteria = np.array([(coordinates[:, 0] > (bounds[0, 0] + margin[0])),
                         (coordinates[:, 0] < (bounds[1, 0] - margin[0])),
                         (coordinates[:, 1] > (bounds[0, 1] + margin[1])),
                         (coordinates[:, 1] < (bounds[1, 1] - margin[1]))
                         ])

    return criteria.all(axis=0)


def coordinates_within_margin(coordinates, image=None, bounds=None, margin=10):
    criteria = coordinates_within_margin_selection(coordinates,  image=image, bounds=bounds, margin=margin)
    if criteria.size == 0:
        return np.array([])
    else:
        return coordinates[criteria]


def circle(r):
    d = 2*r + 1
    rx, ry = d/2, d/2
    x, y = np.indices((d, d))
    return (np.abs(np.hypot(r - x, r - y)-r) < 0.5).astype(int)

def coordinates_without_intensity_at_radius(coordinates, image, radius, cutoff, fraction_of_peak_max = 0.25):
    if cutoff == 'image_median': cutoff = np.median(image)
    circle_matrix = circle(radius)
    new_coordinates = []

    coordinates = coordinates_within_margin(coordinates, image=image, margin=radius+1)

    for i, coordinate in enumerate(coordinates):
        # Could use the coordinates_within_margin for this [IS 01-11-2019]
        #if np.all(coordinate > radius+1) and \
        #        np.all(coordinate < (np.array(image.shape)-radius-1)):

        cropped_peak = crop(image, coordinate, radius*2+1)
        #if np.all((cropped_peak * circle_matrix) < (cutoff + fraction_of_peak_max * np.max(cropped_peak))): #OLd version!
        if np.all((cropped_peak * circle_matrix) < (cutoff + fraction_of_peak_max * (np.max(cropped_peak)-cutoff))):
            new_coordinates.append(coordinate)

    return np.array(new_coordinates)

def crop(image, center, width):
    center = np.round(center).astype(int)
    return image[(center[1]-width//2):(center[1]+width//2+1),(center[0]-width//2):(center[0]+width//2+1)]

# def twoD_gaussian(M, offset, amplitude, x0, y0, sigma_x, sigma_y):
#     x, y = M
#     return offset + amplitude * np.exp(- ((x-x0)/(2*sigma_x))**2 - ((y-y0)/(2*sigma_y))**2)

def twoD_gaussian(M, offset, amplitude, x0, y0, sigma):
    x, y = M
    return offset + amplitude * np.exp(- ((x-x0)**2+(y-y0)**2)/(2*sigma**2))

def fit_twoD_gaussian(Z):
    height, width = Z.shape
    x, y = np.arange(width)-width//2, np.arange(height)-height//2
    X, Y = np.meshgrid(x, y)
    xdata = np.vstack((X.ravel(), Y.ravel()))

    # p0 = [20,20,0,0,1,1]
    # p0 = [np.min(Z), np.max(Z) - np.min(Z), 0, 0, 1, 1]
    p0 = [np.min(Z), np.max(Z) - np.min(Z), 0, 0, 1]
    popt, pcov = curve_fit(twoD_gaussian, xdata, Z.ravel(), p0) #, maxfev=3000) #input: function, xdata, ydata,p0

    # The offset can potentially be used for background subtraction
    return popt

def coordinates_after_gaussian_fit(coordinates, image, gaussian_width = 9, return_fit_parameters=False):
    # TODO: fix standard deviation to the psf size, this may improve correct peak localization when two peaks are close.
    new_coordinates = []
    fit_parameters = []
    if len(coordinates) == 0:  # This statement may not be necessary. However, check the code thoroughly before you remove this.
        new_coordinates = coordinates
    else:
        coordinates = coordinates_within_margin(coordinates, image=image, margin=gaussian_width//2+1)

        for i, coordinate in enumerate(coordinates):
            # Could use the coordinates_within_margin for this [IS 01-11-2019]
            #if np.all(coordinate > gaussian_width//2+1) and \
            #        np.all(coordinate < np.array(image.shape)-gaussian_width//2-1):
            cropped_peak = crop(image, coordinate, gaussian_width)
            try:
                coefficients = fit_twoD_gaussian(cropped_peak)
                #new_coordinates.append(coordinate + coefficients[2:4])
                new_coordinate = coordinate + coefficients[2:4]
                if np.sum(np.abs(coefficients[2:4]))<gaussian_width*2:
                    new_coordinates.append(new_coordinate)
                    fit_parameters.append(coefficients)
                # else: #MD do nothing, you don't want to include fits with a center far outside the cropped image
            except RuntimeError:
                pass
    if return_fit_parameters:
        return np.array(new_coordinates), np.array(fit_parameters)
    else:
        return np.array(new_coordinates)


def merge_nearby_coordinates(coordinates, distance_threshold=2, plot=False):
    """Merge nearby coordinates to a single coordinate

    Coordinates are stored in a KD-tree.
    Each pair of points with a distance smaller than the distance threshold is obtained
    Pairs are chained to obtain groups of points
    For each group find the center coordinate and use that as a new coordinate
    (do this only if each member of the group is within the distance threshold from the center coordinate).
    Add individual points to the new coordinate list, i.e. points that do not have other points within the distance threshold.

    Parameters
    ----------
    coordinates : numpy.ndarray of ints or floats OR set of tuples
        Array with each row a set of coordinates
    distance_threshold : int or float
        Points closer than this distance are considered belonging to the same molecule.
    plot : bool
        If True shows a scatter plot of the coordinates and the new coordinates on top. (Only for 2D coordinates)

    Returns
    -------
    new_coordinates : numpy.ndarray of floats
        Coordinate array after merging nearby coordinates

    """

    # Convert to numpy array in case the coordinates are given as a set of tuples
    coordinates = array_from_set_of_tuples(coordinates)

    # Put coordinates in KD-tree for fast nearest-neighbour finding
    coordinates_KDTree = cKDTree(coordinates)

    # Determine pairs of points closer than the distance_threshold
    close_pairs = coordinates_KDTree.query_pairs(r=distance_threshold)
    close_pairs = [set(pair) for pair in close_pairs] # Convert to list of sets

    # Chain the pairs to obtain groups (or clusters) of points
    groups_of_points = combine_overlapping_sets(close_pairs)

    # Calculate the new coordinates by taking the center of all the neighbouring points.
    # A threshold for the total group is applied, i.e. all points must lie within the distance_threshold
    # from the center coordinate.
    new_coordinates = []
    for group in groups_of_points:
        group_coordinates = coordinates[list(group)]
        center_coordinate = np.mean(group_coordinates, axis=0)
        distances_to_center = np.sqrt(np.sum((group_coordinates-center_coordinate)**2, axis=1))
        if not (np.max(distances_to_center) > distance_threshold): # This could be another threshold
            new_coordinates.append(center_coordinate)

    # Obtain individual points, i.e. those that do not have another point within the distance_threshold.
    # This is done by taking the difference from all points and the ones that are present in any of the groups.
    all_points_in_groups = set(point for group in groups_of_points for point in group)
    all_points = set(range(len(coordinates)))
    individual_points = all_points.difference(all_points_in_groups)

    # Add individual points to new_coordinates list
    for point in individual_points:
        new_coordinates.append(coordinates[point])

    # Convert to numpy array
    new_coordinates = np.array(new_coordinates)

    if plot:
        axis = plt.figure().gca()
        axis.scatter(coordinates[:,0],coordinates[:,1])
        axis.scatter(new_coordinates[:,0],new_coordinates[:,1])

    return new_coordinates


def combine_overlapping_sets(old_list_of_sets):
    """ Combine sets that have overlap

    Go through each set, if it has overlap with one of the sets in the new list of sets, then combine it with this set
    If there is no overlap, append the set to the new list of sets.
    Perform this function recursively until the new_list_of_sets does not change anymore.

    Parameters
    ----------
    old_list_of_sets : list of sets
        List of sets of which overlapping ones should be combined

    Returns
    -------
    new_list_of_sets : list of sets
        Combined list of sets

    """

    # test_set1 = [set((1,2)),set((3,4)),set((5,2)),set((5,6)),set((4,10))]
    # test_set2 = [set((1,2)),set((3,4)),set((2,3)),set((5,6)),set((4,10))]

    new_list_of_sets = []
    for old_set in old_list_of_sets:
        append = True
        for new_set in new_list_of_sets:
            if not (old_set.isdisjoint(new_set)):
                new_set.update(old_set)
                append = False
        if append:
            new_list_of_sets.append(old_set.copy())

    if not (new_list_of_sets == old_list_of_sets):
        new_list_of_sets = combine_overlapping_sets(new_list_of_sets)

    return new_list_of_sets


def set_of_tuples_from_array(array):
    return set([tuple(a) for a in array])


def array_from_set_of_tuples(set_of_tuples):
    return np.array([t for t in set_of_tuples])



if __name__ == '__main__':
    from papylio import Experiment
    from papylio.peak_finding import find_peaks

    exp = Experiment(r'D:\ivoseverins\SURFdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\20141017 - Holliday junction - Copy\HJC-50pM')
    file = exp.files[2]
    movie = file.movie
    image = movie.make_average_image(write=False)

    coordinates = find_peaks(image=image, method='adaptive-threshold', minimum_area=5, maximum_area=15)

    plt.imshow(image)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='b')

    coordinates = coordinates_within_margin(coordinates, bounds = np.array([[0,255], [0,511]]), margin=10)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='g')

    coordinates = coordinates_after_gaussian_fit(coordinates, image)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='y')

    coordinates = coordinates_without_intensity_at_radius(coordinates,
                                                          image,
                                                          radius=4,
                                                          cutoff=np.median(image),
                                                          fraction_of_peak_max=0.35) # was 0.25 in IDL code
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='r')
