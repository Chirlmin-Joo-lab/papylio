Single channel experiment
====================================
Papylio also supports a single emission channel, as opposed to a donor and acceptor channel.
However, it requires a short configuration step.

.. code-block:: python

    for file in exp.files:
        file.movie.channels = file.movie.channels[0:1]
        file.movie.channel_arrangement = [[[0]]]

These settings are currently not saved, so this needs to be run every time the experiment is loaded.