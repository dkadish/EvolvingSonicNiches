Examples
=================

Experiment
--------------------------------------------

Experiments can be run using the python scripts in :py:mod:`evolvingniches.runs`.
:py:mod:`evolvingniches.runs.evolve_1_species_with_noise` is the module used in "Kadish, D., & Risi, S. (2020). Adapting to a changing environment: Simulating the effects of noise on animal sonification. The 2020 Conference on Artificial Life, 687–695. https://doi.org/10.1162/isal_a_00320".

Its usage is documented in :doc:`experiments/evolve_1_species_with_noise`, but can be invoked with default options as::

    python -m evolvingniches.runs.evolve_1_species_with_noise

Config
------

``config.ini`` sets preferences for logdna which isn't really used anymore, so it can pretty much be ignored.

Experiment Analysis using a Jupyter Notebook
--------------------------------------------

Take a look at ``examples/experiment_analysis_notebook.ipynb`` for an example of analysis using a jupyter notebook.
You need to have generated the data files first.

SLURM Job - Experiment
--------------------------------------------
Job submission to a SLURM server.

SLURM Job - Combine Dataframes
--------------------------------------------
Combining the datafiles on the SLURM server (my MacBook ran out of memory at some point).