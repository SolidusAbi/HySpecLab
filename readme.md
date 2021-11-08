# HySpecLab
Laboratory for the use of algorithms in HyperSpectral imagery.

## Prerequisites

* [Anaconda](https://www.anaconda.com/distribution/)
* [Git](https://git-scm.com/)

## Submodules
This repository contains the following submodules:

* [IPDL](git@github.com:SolidusAbi/IPDL.git)

In order to **download the submodules** in the cloning process, use the following instruction:
``` Bash
git clone --recurse-submodules git@github.com:mt4sd/ThermalAnalysis.git
```

## Dependencies
1. [PyTorch](https://anaconda.org/pytorch/pytorch) 
    * version 1.9 or above
1. [Torchvision](https://anaconda.org/pytorch/torchvision)
    * version 0.10 or above
1. [Statsmodels](https://anaconda.org/anaconda/statsmodels)
    * version 0.12.2
1. [Scikit-Learn](https://anaconda.org/anaconda/scikit-learn)
    * version 0.23
1. [Scikit-Optimize](https://anaconda.org/conda-forge/scikit-optimize)
    * version 0.8.1
1. Others:
    * Matplotlib...

## Export Environment
If you have included a new library in your environment, please update the *environment* using the following command:
### Linux
``` Bash
conda env export --from-history | grep -v "^prefix: " > environments/Linux.yml
```

### Windows
``` Bash
TODO
```

## Resources
* [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

## TODO
- [ ] Setup library.
- [ ] Fix HyperSpectralUnderSampler.