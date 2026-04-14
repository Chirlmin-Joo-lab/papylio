"""Basic shading correction helpers for movie images.

Contains lightweight utilities used to compute and apply simple shading/darkfield
corrections to movie frames.
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import tqdm

from papylio.movie.shading_correction import BaSiC


def BaSiCfun(directory, output_directory, extension='.tif', estimate_darkfield=False, apply_correction=False,
             use_flatfield=None, use_darkfield=None, epsilon=1e-6, l_s=None, l_d=None,
             output_flatfield_filename=None, output_darkfield_filename=None, verbose=False):
    """Run BaSiC shading correction on image stack.

    Executes the BaSiC (Background and Shading Correction) algorithm for
    correcting uneven illumination and background in microscopy images.
    Can estimate or use provided flatfield and darkfield corrections.

    Parameters
    ----------
    directory : str or Path
        Directory containing image stack to process
    output_directory : str or Path
        Directory to save output corrected images and correction fields
    extension : str, optional
        Image file extension to process (default: '.tif')
    estimate_darkfield : bool, optional
        If True, estimate darkfield; if False use provided or skip (default: False)
    apply_correction : bool, optional
        If True, apply correction to images and save results (default: False)
    use_flatfield : str or Path, optional
        Path to pre-computed flatfield image to use (default: None)
    use_darkfield : str or Path, optional
        Path to pre-computed darkfield image to use (default: None)
    epsilon : float, optional
        Small value for numerical stability (default: 1e-6)
    l_s : float, optional
        Regularization parameter for flatfield (default: None, auto-set)
    l_d : float, optional
        Regularization parameter for darkfield (default: None, auto-set)
    output_flatfield_filename : str or Path, optional
        Custom path for saving flatfield correction (default: None)
    output_darkfield_filename : str or Path, optional
        Custom path for saving darkfield correction (default: None)
    verbose : bool, optional
        If True, print progress information (default: False)

    Notes
    -----
    - Saves flatfield.tif and darkfield.tif to output_directory
    - Output files are saved as float32 TIFF format
    - If perform_estimation=False, still creates optimizer for corrections
    """
    # Create the BaSiC shading correction object
    optimizer = BaSiC(directory, estimate_darkfield=estimate_darkfield, extension=extension, verbose=verbose)

    # Set some optimizer parameters
    optimizer.l_s = l_s
    optimizer.l_d = l_d
    # Prepare the optimization
    optimizer.prepare()

    # Extract the flat and dark fields
    perform_estimation = True
    if use_flatfield is not None:
        img = cv2.imread(use_flatfield, cv2.IMREAD_ANYDEPTH)
        optimizer.set_flatfield(img)
        perform_estimation = False

    if use_darkfield is not None:
        img = cv2.imread(use_darkfield, cv2.IMREAD_ANYDEPTH)
        optimizer.set_flatfield(img)
        perform_estimation = False

    # Perform the estimation
    if perform_estimation:
        optimizer.run()

        # Save the estimated fields (only if the profiles were estimated)
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        if output_flatfield_filename is not None:
            flatfield_name = Path(output_flatfield_filename).resolve()
            flatfield_name.parent.mkdir(parents=True, exist_ok=True)
        else:
            flatfield_name = directory / "flatfield.tif"
        if output_darkfield_filename is not None:
            darkfield_name = Path(output_darkfield_filename).resolve()
            darkfield_name.parent.mkdir(parents=True, exist_ok=True)
        else:
            darkfield_name = directory / "darkfield.tif"

        cv2.imwrite(str(flatfield_name), optimizer.flatfield_fullsize.astype(np.float32))
        cv2.imwrite(str(darkfield_name), optimizer.darkfield_fullsize.astype(np.float32))

    # Apply shading correction.
    if apply_correction:
        optimizer.write_images(output_directory, epsilon=epsilon)




def squeeze_channel_from_frames(frames):
    """Reorganize channel dimension in frame xarray DataArray.

    Converts channel-separated frames into spatially adjacent channels by
    combining channels along x and y pixel dimensions. Useful for preparing
    data for channel-specific processing.

    Parameters
    ----------
    frames : xr.DataArray
        Frame data with dimensions (frame, channel, x_pixel, y_pixel)

    Returns
    -------
    xr.DataArray
        Reorganized frames with channels combined into spatial dimensions
    """
    return xr.combine_by_coords(
        [frames.sel(channel=channel).set_index(x='x_pixel', y='y_pixel') for channel in frames.channel])

    # xr.combine_by_coords([frames.sel(channel=channel.index).set_index(x='x_pixel', y='y_pixel') for channel in self.channels])
    #
    # test = frames.assign_coords(channel_index_x=('channel',[0,1]),channel_index_y=('channel',[0,0]))
    # test2 = test.drop('channel').set_index(channel=('channel_index_x','channel_index_y'))
    # test3 = test2.unstack('channel').stack(x_pixel=('channel_index_x','x')).stack(y_pixel=('channel_index_y','y'))
    #
    # test2.unstack('channel').stack(channel=('channel_index_x','channel_index_y'))
    # test2.unstack('channel').stack(x_pixel=('channel_index_x','x'), y_pixel=('channel_index_y','y'))

def spatial_shading_correction(movies, method='BaSiC', illumination_index=0, frame_index=0, estimate_darkfield=True, **kwargs):
    """Compute spatial shading corrections for a collection of movies.

    Applies BaSiC or averaging-based correction to estimate flatfield and
    darkfield for specific illumination pattern, across multiple movies.

    Parameters
    ----------
    movies : Collection of Movie
        Movie objects or collection thereof to analyze
    method : str, optional
        Correction method: 'BaSiC' or 'average' (default: 'BaSiC')
    illumination_index : int, optional
        Index of illumination pattern to correct (default: 0)
    frame_index : int, optional
        Frame index within illumination group to use (default: 0)
    estimate_darkfield : bool, optional
        If True, estimate darkfield; otherwise use existing (default: True)
    **kwargs
        Additional keyword arguments passed to BaSiC optimizer

    Returns
    -------
    darkfield : np.ndarray
        Estimated darkfield correction array
    flatfield : np.ndarray
        Estimated flatfield correction array

    Notes
    -----
    - Reads frames only from movies containing the specified illumination_index
    - Uses first frame of each movie with the illumination pattern
    - Processes each channel independently
    - Output arrays match movie channel dimensions
    - BaSiC method provides more sophisticated correction than averaging
    """
    selected_movies = [movie for movie in movies if illumination_index in movie.illumination_indices_in_movie]
    frame_with_illumination = [np.where(movie.illumination_index_per_frame == illumination_index)[0][frame_index] for movie in selected_movies]
    # frames = xr.concat([movie.read_frames([frame], apply_corrections=False, xarray=True, flatten_channels=False)
    #                     for movie, frame in tqdm.tqdm(zip(selected_movies, first_frame_with_illumination))], dim='frame')
    # frames = frames.reset_index('frame', drop=True)
    frames = np.vstack([movie.read_frames([frame], apply_corrections=False, xarray=False, flatten_channels=False)
                        for movie, frame in tqdm.tqdm(zip(selected_movies, frame_with_illumination),
                                            'Read frames', len(selected_movies))])

    flatfield = np.ones_like(frames[0], dtype=float)
    if estimate_darkfield:
        darkfield = np.zeros_like(frames[0], dtype=float)
    else:
        darkfield = selected_movies[0].corrections.darkfield_correction.sel(illumination=illumination_index).values
        frames = frames - darkfield[None, :, :, :]

    if method == 'BaSiC':
        for channel_index in movies[0].channel_indices:
            optimizer = BaSiC(frames[:, channel_index, :, :], estimate_darkfield=estimate_darkfield, extension=None, verbose=True, **kwargs)
            optimizer.prepare()
            optimizer.run()
            flatfield[channel_index,:,:] = optimizer.flatfield_fullsize
            if estimate_darkfield:
                darkfield[channel_index,:,:] = optimizer.darkfield_fullsize
    elif method == 'average':
        flatfield = frames.mean(axis=0)
        flatfield = flatfield / flatfield.mean(axis=(-2,-1), keepdims=True)

    # flatfield = squeeze_channel_from_frames(flatfield)
    # darkfield = squeeze_channel_from_frames(darkfield)

    flatfield = movies[0].flatten_channels(flatfield)
    darkfield = movies[0].flatten_channels(darkfield)

    return darkfield, flatfield

#
# movies = files_green_laser[0::25].movie
# darkfield, flatfield = spatial_shading_correction(movies)
#
# import tifffile
# save_path = Path(r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20220602 - Objective-type TIRF (BN)')
# tifffile.imwrite(save_path.joinpath('flatfield.tif'), flatfield)
# tifffile.imwrite(save_path.joinpath('darkfield.tif'), darkfield)

# plt.figure()
# plt.imshow(flatfield)
#
# plt.figure()
# plt.imshow(darkfield)

#
# frames = files_green_laser[0].movie.read_frames_raw()
# frames = squeeze_channel_from_frames(frames)

if __name__ == '__main__':
    import tifffile
    from pathlib2 import Path
    import numpy as np

    pth = Path(r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20220602 - Objective-type TIRF (BN)\Analysis\Test')

    img_stack = tifffile.imread(pth.joinpath('TIRF 561 0001.tif'))
    img_stack = np.rot90(img_stack, 1, axes=(1,2))
    flatfield = tifffile.imread(pth.joinpath('flatfield.tif'))
    darkfield = tifffile.imread(pth.joinpath('darkfield.tif'))

    from papylio.movie.shading_correction import get_photobleach

    import time
    start = time.time()
    c1 = get_photobleach(img_stack[:,:,0:256], flatfield[:,0:256], darkfield[:,0:256], size=(32,64))
    print(time.time()-start)
    start = time.time()
    c2 = get_photobleach(img_stack[:,:,256:512], flatfield[:,256:512], darkfield[:,256:512], size=(32,64))
    print(time.time()-start)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(c1.T/c1.max())
    plt.plot(c1.T*c2.max()/c2.T)
    plt.plot(c2.T-c1.T*c2.max()/c2.T)
    plt.plot(c2.max()/c2.T)
    plt.plot(c2.T/c2.max())
    plt.plot(test2/test2.max())




    from papylio.movie.shading_correction import get_photobleach
    import time
    start = time.time()
    test = get_photobleach(frames.values, flatfield, darkfield=darkfield)
    print(time.time()-start)

    import scipy.ndimage
    start = time.time()
    # test2 = np.array([scipy.ndimage.minimum_filter(((frame)), size=15, mode='wrap').sum() for frame in img_stack])
    test2 = np.array([scipy.ndimage.gaussian_filter(((frame-darkfield)/flatfield)[:,0:256], sigma=50, mode='wrap').mean() for frame in img_stack])
    test2 = np.array([scipy.ndimage.minimum_filter(((frame-darkfield)/flatfield)[:,0:256], size=15, mode='wrap').mean() for frame in img_stack])
    # test2 = np.array([frame.sum() for frame in frames])
    # test2 = np.array([scipy.ndimage.minimum_filter(frame, size=15, mode='wrap').sum() for frame in img_stack])
    print(time.time()-start)

    test = test.squeeze()
    test = test/test.sum()
    test2 = test2/test2.sum()

    plt.figure()
    plt.plot(test)
    plt.plot(test2)





