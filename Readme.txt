===============
Digital Embryos
===============

This is the code repository for the digital embryos projects. To learn more about the code repository as well as the project,
please visit the corresponding wiki page at
https://gorgonzola.iwr.uni-heidelberg.de/intern/wiki/index.php/Digital_Embryos#SVN_Repository.

Dependencies
============
Vigra
-----
If Vigra is not installed at a standard location, please set the env
variables CPLUS_INCLUDE_PATH, LD_LIBRARY_PATH, and LIBRARY_PATH
accordingly. Alternatively, you can point the env variable VIGRA_ROOT
to a Vigra project or installation directory (make sure, that Vigra is
compiled inside the project directory)

We depend on vigranumpy. Make sure, that the PYTHONPATH points to a
directory, where "vigra/vigranumpycore.so" resides.

Boost
-----
If Boost is not installed at a standard location, please set the env
variable BOOST_ROOT to the project directory.

HDF5
----
* version >=1.8.5
* To provide the module with a hint about where to find your HDF5
  installation, you can set the environment variable HDF5_ROOT. 
