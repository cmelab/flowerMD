# Contributing to flowerMD

Thank you for considering contributing to flowerMD! We welcome all contributions
in the form of bug reports, feature requests, code contributions, suggestions, etc.

The default branch for this repository is `main`. Please submit all pull requests
to the `main` branch.

## Bug reports and feature requests
For bug reports and feature requests, please open an issue on the
[GitHub issue tracker](https://github.com/cmelab/flowerMD/issues).

If you are reporting a bug, please include the following information:
- A short description of the bug
- A minimal working example that reproduces the bug
- The expected behavior
- The actual behavior
- The version of flowerMD you are using

For feature requests, please include a short description of the feature you would
like to see added to flowerMD.

## Code contributions
We welcome all code contributions in the form of pull requests. If you are
interested in contributing code, please follow the steps below:
- Fork the repository (default branch is `main`)
- Create a new branch for your feature/bug fix
- Create the `flowermd` conda environment using the `environment-cpu.yml` or `environment-gpu.yml` file
- Make your changes
- Install the latest version of pre-commit using `conda install -c conda-forge pre-commit` and run `pre-commit install` in the root directory of the repository
- Commit your changes and make sure all the pre-commit hooks pass
- Push your changes to your fork and create a pull request
- Assign the pull request to a reviewer and wait for the reviewers feedback

New functionalities should be covered by unit tests. If you are adding a new
function/feature/class please follow the existing style for docstrings.

## Questions and comments

Please feel free to create an issue if you have any questions or comments about
flowerMD.