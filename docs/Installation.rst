Installation
============

Conda and pip
-------------

Papylio can be installed through conda.

.. code-block::

   conda install papylio -c conda-forge

or through using pip

.. code-block::

   pip install papylio

where it is recommended to make a separate virtual environment or conda environment for `papylio`.

.. ATTENTION::

   Currently, installing Papylio through conda on Windows is giving an dll error when loading an Experiment.
   For now it is recommended to either install python and then use pip to install papylio.

   .. code-block::

      conda create -n papylio
      conda install python=3.9 -c conda-forge
      pip install papylio

   Or use the download the environment file in `installer/windows/environment_windows.yml` from the github repository and
   use it to create a conda environment with all the papylio dependencies.
   After that Papylio can be installed in the regular way using conda.

   .. code-block::

      conda create -n papylio -f environment_windows.yml
      conda install papylio -c conda-forge


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