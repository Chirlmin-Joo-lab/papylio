SPARXS
======

Papylio can be used for analyzing data from a Single-molecule Parallel Analysis for Rapid eXploration of Sequence space (SPARXS) experiment,
as described in the research article `Single-molecule structural and kinetic studies across sequence space`_ and
the protocol `Single-molecule parallel analysis for rapid exploration of sequence space`_.

Obtaining the sequence characteristics from the data is divided into 6 step.

1. :doc:`Single-molecule data analysis<1_single_molecule_data_analysis>`
2. :doc:`Sequence data analysis<2_sequence_data_analysis>`
3. :doc:`Aligning datasets<3_aligning_datasets>`
4. :doc:`Coupling datasets<4_coupling_datasets>`
5. :doc:`Single-molecule data analysis 2<5_single_molecule_data_analysis_2>`
6. :doc:`Properties per sequence<6_properties_per_sequence>`

When doing SPARXS for the first time or when using a new combination of microscope and sequencer, the :doc:`global alignment<global_alignment>` will need to be determined using a separate experiment.

The documentation was made using the `SPARXS example dataset`_ which is freely available at Zenodo.

.. toctree::
    :maxdepth: 2
    :hidden:

    1_single_molecule_data_analysis
    2_sequence_data_analysis
    3_aligning_datasets
    4_coupling_datasets
    5_single_molecule_data_analysis_2
    6_properties_per_sequence
    global_alignment


.. _Single-molecule structural and kinetic studies across sequence space: https://www.science.org/doi/10.1126/science.adn5968

.. _Single-molecule parallel analysis for rapid exploration of sequence space: https://www.nature.com/articles/s41596-025-01196-y

.. _SPARXS example dataset: https://doi.org/10.5281/zenodo.13841177