Getting started
===============
A regular analysis consists of mapping the emission channels, localizing molecules and
extracting the intensity traces.

First import the library and create an :class:`.Experiment`.

.. code-block:: python

    import papylio as pp
    exp = pp.Experiment()
    exp.files.print()

Calling :py:class:`.Experiment` without any arguments will open a dialog box where you can select the main folder containing
the single-molecule movies.

Creating the experiment object will either load an existing configuration (config) file that is present in the selected path,
or it will create a default config file which can then be adapted.



