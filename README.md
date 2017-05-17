# NOTE!!
This program requires the CUDA toolkit associated with visual studio to run. Not to mention you need to have a NVIDIA gpu plugged in.

The main solver code can be found in SPH/SPH/kernel.cu
There are two configuration files in the above directory: config.h and parameters.h, You can modify the parameters in the parameters.h file to fine tune the simulation. config.h includes stuff that is more implementatuin specific.

Initial configuration of SPH particles in the domain are read from 'grid.txt'. You can modify this file to change the various ascpects of the simulation.
Schema description of the above file: *coming soon*

For any particluar problem, you will have to input these three files: parameters.h, config.h and grid.txt

For starters you can load the root folder into visual studio and run the program. A simple dam break scenario is already set up.
