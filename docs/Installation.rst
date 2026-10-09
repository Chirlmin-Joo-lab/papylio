Installation
============

Standalone installation
-----------------------
The easiest way to install Papylio is as a standalone application.
It provides a graphical user interface and additionally allows running custom Python code through Marimo notebooks.
Note that only packages bundled with Papylio can be used in the notebooks.
If you want to use additional packages, it is recommended to install Papylio through `conda or pip <Conda and pip_>`_ instead.

The standalone application is available through the links below:

- `Download for Windows`_
- `Download for macOS`_

All versions can be found on the `releases`_ page of the GitHub repository.

.. _releases: https://github.com/Chirlmin-Joo-lab/papylio/releases

Conda and pip
-------------

Papylio can be installed through conda.

.. code-block::

   conda install papylio -c conda-forge

or through using pip

.. code-block::

   pip install papylio

where it is recommended to make a separate virtual environment or conda environment for `papylio`.

Detailed conda installation steps
---------------------------------
For installation with conda, it is recommended to use `miniforge`_.

1. Download and install `miniforge`_ (e.g. v26.5.3).
2. Open Miniforge Prompt from the Windows start menu.
3. Create a new environment by typing

   .. code-block:: bash

      conda create -n papylio

   and hit enter.

4. Activate the environment

   .. code-block:: bash

      conda activate papylio

5. Install papylio

   .. code-block:: bash

      conda install papylio -c conda-forge

6. Test the installation by running

   .. code-block:: bash

      python -m papylio

   or find the papylio icon in the start menu and click it.


Anaconda Navigator installation
-------------------------------
Papylio can also be installed through the Anaconda Navigator gui.

1. Download and install `Anaconda Navigator`_ (e.g. v2.6.3).
2. Create a new environment

   - Go to the `Environments` tab
   - Click `Create`
   - Enter the environment name, e.g. `papylio`
   - Select Python with version 3.9.x
   - Click `Create`
   - Select the `papylio` environment.

3. Add the `conda-forge` channel.

   - Click `Channels`
   - Click `Add`
   - Type `conda-forge` and hit `Enter`.
   - Click `Update channels`.

4. Install papylio

   - Change `Installed` to `Not installed`
   - Search for `papylio` in the `Search Packages` search box.
   - Click the small square on the left side of `papylio`.
   - Click `Apply`


.. _Anaconda Navigator: https://www.anaconda.com/products/navigator
.. _Miniforge: https://conda-forge.org/download/