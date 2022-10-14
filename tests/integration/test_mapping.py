import pytest
import tifffile
import numpy as np
from skimage.transform import SimilarityTransform, AffineTransform
from trace_analysis.mapping.mapping import Mapping2


def test_geometric_hash_table():
    translation = np.array([256, 10])
    rotation = 125 / 360 * 2 * np.pi
    scale = np.array([10, 10])
    transformation = SimilarityTransform(translation=translation, rotation=rotation, scale=scale)
    mapping = Mapping2.simulate(number_of_points=200, transformation=transformation,
                                bounds=([0, 0], [256, 512]), crop_bounds=((50, 200), None), fraction_missing=(0.1, 0.1),
                                error_sigma=(0.5, 0.5), shuffle=True, seed=10252)
    mapping.geometric_hashing(method='test_one_by_one', tuple_size=4, maximum_distance_source=100, maximum_distance_destination=1000,
                              distance=15, alpha=0.9, sigma=10, K_threshold=10e9, hash_table_distance_threshold=0.01,
                              magnification_range=None, rotation_range=None)

    assert mapping.transformation_is_similar_to_correct_transformation(translation_error=50, rotation_error=0.01,
                                                                       scale_error=0.2)

    mapping.transformation = mapping.transformation_correct
    mapping.geometric_hashing(method='abundant_transformations', tuple_size=4, maximum_distance_source=100, maximum_distance_destination=1000,
                              hash_table_distance_threshold=0.01,
                              parameters=['translation', 'rotation', 'scale']
                              )

    assert mapping.transformation_is_similar_to_correct_transformation(translation_error=10, rotation_error=0.01, scale_error=0.01)

def test_kernel_correlations():
    translation = np.array([10, -10])
    rotation = 1 / 360 * 2 * np.pi
    scale = [0.98, 0.98]
    transformation = AffineTransform(translation=translation, rotation=rotation, scale=scale)
    mapping = Mapping2.simulate(number_of_points=10000, transformation=transformation,
                                bounds=([0, 0], [256, 512]), crop_bounds=(None, None), fraction_missing=(0.95, 0.6),
                                error_sigma=(0.5, 0.5), shuffle=True, seed=10252)

    minimization_bounds = ((0.97, 1.02), (-0.05, 0.05), (-20, 20), (-20, 20))
    mapping.kernel_correlation(minimization_bounds, sigma=1, crop=False, plot=False,
                               strategy='best1bin', maxiter=1000, popsize=50, tol=0.01, mutation=0.25, recombination=0.7,
                               seed=None, callback=None, disp=False, polish=True, init='sobol', atol=0,
                               updating='immediate', workers=1, constraints=())

    assert mapping.transformation_is_similar_to_correct_transformation(translation_error=1, rotation_error=0.001, scale_error=0.001)