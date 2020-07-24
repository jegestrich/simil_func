simil_func
============

Tool for automated fitting of the large-scale-turbulence (LST) and fine-scale-turbulence (FST) spectra based on Tam et al., 1996 to volcanic eruption infrasound.

Included functions in `simil_func.py`:

**Basic functions**:
* `simil_func()`: Function to calculate the similarity spectrum for combined contributions of fine scale turbulence (FST) and large scale turbulence (LST) from *Tam et al., 1996*, equation (2)
* `simil_FST_func()`: Function to calculate the similarity spectrum for fine scale turbulence (FST) from *Tam et al., 1996*, equation (4)
* `simil_LST_func()`: Function to calculate the similarity spectrum for large scale turbulence (LST) from *Tam et al., 1996*, equation (3)

**Fitting**:
* `simil_fit()`: Tool for automated fitting of similarity spectra (LST & FST) to a given spectrum using non-linear least squares fitting and root-mean-square error as misfit function (`misfit()`)

**Plotting**:
* `simil_plot()`: Plotting tool for the results of `simil_fit()`. Plots spectrogram of original (unfiltered) data and associated (filtered) waveform and spund pressure level (SPL) as well as time series of similarity spectra misfits.

**References**
*Tam, C. K. W., Golebiowski, M., & Seiner, J. M. (1996). On the Two Components of Turbulent Mixing Noise from Supersonic Jets. American Institute of Aeronautics and Astronautics.*

<!---*If this package accompanies your paper or uses specific results from a paper,
reference it here...*-->


Installation
------------

<!--*Here are install instructions for an example conda environment. For
consistency, we encourage all interfacing packages in uafgeotools to use conda
environments.*-->

We recommend using conda and creating a new conda environment such as:
```
conda create -n simil_func -c conda-forge obspy
```
Or use an existing environment with installed required packages (see Dependencies).
Information on conda environments (and more) is available
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

This package is dependent on [_waveform_collection_](https://github.com/uafgeotools/waveform_collection) and has to be installed in the environment. If not downloaded yet execute:
```
$ conda activate simil_func  # Or your pre-existing env
$ git clone https://github.com/uafgeotools/waveform_collection.git
$ cd waveform_collection
$ pip install -e .
$ cd ..
```
```
$ git clone https://github.com/uafgeotools/lts_array.git
$ cd lts_array
$ pip install --editable .
$ cd ..
```

```
$ git clone https://github.com/uafgeotools/array_processing.git
$ cd array_processing
$ pip install --editable .
$ cd ..
```

After setting up the conda environment,
[install](https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs)
the package by executing the following commands:
```
$conda activate simil_func
$git clone https://github.com/jegestrich/simil_func.git
$cd simil_func
$pip install -e .
```
The final command installs the package in "editable" mode, which means that you
can update it with a simple `git pull` in your local repository. This install
command only needs to be run once.


Dependencies
------------

*For example:*

_uafgeotools_ packages:
* [_waveform_collection_](https://github.com/uafgeotools/waveform_collection)
* [_lts_array_](https://github.com/uafgeotools/lts_array)
* [_array_processing_](https://github.com/uafgeotools/array_processing)

Python packages:
* [ObsPy](http://docs.obspy.org/)
* [OpenPyXL](https://openpyxl.readthedocs.io/en/stable/#) (only necessary for [*example.py*](simil_func/example.py))


Example
-------

See the included [*example.py*](simil_func/example.py).


Usage
-----

Import the package like any other Python package, ensuring the correct
environment is active. For example,
```
$ conda activate simil_func
$ python
>>> import simil_func
```

<!--*Mention documentation here. Perhaps point to the example file.*-->


Authors
-------

(_Alphabetical order by last name._)

Julia Gestrich

<!--stackedit_data:
eyJwcm9wZXJ0aWVzIjoiZXh0ZW5zaW9uczpcbiAgcHJlc2V0Oi
BnZm1cbiAgbWFya2Rvd246XG4gICAgYnJlYWtzOiBmYWxzZVxu
IiwiaGlzdG9yeSI6WzYxMTk4MTkwMCwxOTg3MzQ1MzEwLDQzMD
M3MzM1OSw0MzAzNzMzNTldfQ==
-->
