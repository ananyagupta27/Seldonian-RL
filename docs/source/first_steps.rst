First-steps
===========
To get started with the library follow these steps-

Pre-requisites
^^^^^^^^^^^^^^

Install the following python libraries

.. code-block:: python

   pip install numpy
   pip install pandas
   pip install scipy
   pip install cma
   pip install ray
   pip install torch
   pip install sklearn

Running Experiments
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   python experiments.py --environment {0,1,2,3,4} --delta (0,1) --discrete {0,1} --workers <int> --trials <int> --split_ratio (0,1) --is_estimator <IS, PDIS, WIS, DR, DR_hat>  --cis <ttest, Anderson, MPeB, Phil, Hoeffding> --optimizer <Powell, CMA, CMAES, BFGS, CEM>



Creating Plots
^^^^^^^^^^^^^^

.. code-block:: python

   python utils_dir/create_plots.py


Running experiments on swarm cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   ./slurm/slurm_starter.sh


Using QSA library
^^^^^^^^^^^^^^^^^
Create an object of the class QSA class and use its methods.
The usage of this library is also shown in experiments.py, this performs multiple experiments with QSA over multiple trials.

For Example Notebooks and customizing QSA library refer to
https://github.com/ananyagupta27/Seldonian-RL/tree/main/examples