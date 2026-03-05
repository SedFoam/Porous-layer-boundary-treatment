Procedure for calculating shoreline and stack of k and tau :
=========

To calculate the shoreline:
----------------------------

Once the simulation is complete, run the script ‘save_shoreline_position.py’ to calculate and save the shoreline in a netcdf file (save file name: ‘shoreline.nc’).
The code requires nothing more than the simulation result (alpha.water) to run.
It uses a sub-module called ‘intersect.py’ which calculates the intersection point of two curves (found in the Python codes).


To calculate the stacks of k and tau:
----------------------------

Once the simulation is complete, start by running the ‘gradU’ post-processing with OpenFOAM (command ‘postProcess -func gradU’), which calculates the gradient of U at each time step.
Next, run ‘extract_k_nut_gradU.py’, which extracts the values of alpha.water, nut, k, grad(U) in the median plane of the channel and saves them in a netcdf file (file name: ‘k_nut_gradU.nc’).
Then we run the scripts ‘interpol_k.py’ and ‘interpol_tau.py’ which calculate the interpolations of k and tau on probes orthogonal to the beach every centimetre in order to make the stacks, and save them in the files ‘interp_k_6m.nc’ and ‘interp_tau.nc’.
Finally, run the codes ‘stack_tau.py’, ‘stack_TKE.py’ or ‘stack_tau_Shields.py’ depending on what you want, which calculate and save the stacks as such.
This code requires that the shoreline has already been calculated in order to work (i.e. that the ‘shoreline.nc’ file already exists, see previous section).