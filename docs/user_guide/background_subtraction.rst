Background subtraction
======================

To obtain correct fluorescence and FRET values it is important to perform background correction. There are three types of background subtraction that can be performed. A temporal background subtraction that corrects variations in background over time (but not in x and y); a spatial background correction that corrects variations over the x and y (but not over time); and a general background correction, which is a single value for the entire movie. Any of these can be applied. If more than one are used, they have to be determined in the order temporal, spatial, general, as this is also the order in which they are applied. Determining the corrections in a different order, will delete the corrections that should have been determined later. The background corrections are stored in an ``_corrections.nc`` file for each movie.

Temporal background correction
------------------------------
If there are fluorophores in solutions, the background may change slightly over time due to e.g. bleaching.
In this case a temporal background correction can be applied. The following line calculates a single background subtraction value per frame by taking the median.

.. code-block:: python

    file.movie.determine_temporal_background_correction(method='median')


Spatial background correction
-----------------------------
To compensate for variations in background over the image, a spatial correction can be applied. For example by estimating the background using a median filter, which is then applied to all frames.

.. code-block:: python

    file.movie.determine_spatial_background_correction(method='median_filter', size=20)


General background correction
-----------------------------
Because the median filter usually overestimates the signal, this can be compensated by applying a more accurate general background correction, a single value for each entire channel over all frames. The correction can be estimated for example by fitting the background peak in the intensity histogram.

.. code-block:: python

     file.movie.determine_general_background_correction(method='fit_background_peak')
