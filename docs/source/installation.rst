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
