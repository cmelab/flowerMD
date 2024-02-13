============
Installation
============

Install from anaconda
---------------------------------------
flowerMD is available on `conda-forge <https://anaconda.org/conda-forge/flowermd>`_
::

    $ conda install -c conda-forge flowermd


Install from source
---------------------------------------

1. Clone this repository:
::

    $ git clone git@github.com:cmelab/flowerMD.git
    $ cd flowerMD

2. Set up and activate environment:
::

    $ conda env create -f environment.yml
    $ conda activate flowermd
    $ python -m pip install .

.. note::

    To install a GPU compatible version of HOOMD-blue in your flowerMD environment, you need to manually set the cuda version **before installing flowermd**.
    This is to ensure that the HOOMD build pulled from conda-forge is compatible with your cuda version.
    To set the cuda version, run the following command before installing flowermd::

        $ export CONDA_OVERRIDE_CUDA="[YOUR_CUDA_VERSION]"

    Please see the `HOOMD-blue installation instructions <https://hoomd-blue.readthedocs.io/en/stable/installation.html>`_ for more information.
