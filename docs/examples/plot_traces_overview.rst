Plot traces overview
====================
To plot an overview of the traces of several molecules, the following code can be used.

.. code-block:: python

    molecule_indices = np.arange(0,50)
    file.intensity.sel(molecule=molecule_indices).plot(x='time', hue='channel', row='molecule', col_wrap=5)

This plots the first molecule 0 to 50 in rows of 5.