Welcome to ronswanson's documentation!
======================================
Ron Swanson builds tables for 3ML.

This tool provides an easy interface to construct 3ML table/ template models from
computational expensive simulations on HPC systems. Why? Because it is your
right to do so!

Table/template models are models built from compuationally expensive functions
or simulations of astrophysical spectra by evaluating them on a grid in their
parameter space. The output is then interpolated on this grid for evaluation in
e.g. a fit to astrophysical spectral data. These models can be used in `3ML as
described here
<https://threeml.readthedocs.io/en/stable/notebooks/spectral_models.html#Template-(Table)-Models>`_


.. image:: https://raw.githubusercontent.com/grburgess/ronswanson/master/docs/media/mad.jpg
   :alt: ronswanson


Installation
____________

$ pip install ronswanson


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   notebooks/ronswanson
   api/API.rst
	     
