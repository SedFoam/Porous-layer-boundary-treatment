Porous layer boundary treatment
=========

This repository contains the OpenFOAM numerical setups and python post-processing codes used in the Coastal Engineering paper "Numerical study of swash dynamics: Development of a porous layer boundary treatment for rough surfaces modelling".


What is this repository for?
----------------------------

* Supported OpenFoam Versions : v2412plus to latest
* Supported Python Versions : >= 3.13


Download and Installation of the roughWallFunctions library
-----------------------

```bash
cd $FOAM_RUN
git clone --recurse-submodules https://github.com/sedfoam/Porous-layer-boundary-treatment
cd Porous-layer-boundary-treatment/wallFunctions
./Allwmake
```

Usage
---------------

swak4foam is required for running the numerical cases involving a porous layer. If you have difficulties installing swak4foam, we invite you to use the docker environment provided by sedfoam: https://sedfoam.github.io/sedfoam/howtodocker.html.

The python package fluidfoam is required for running the post-processing codes for the Kikkert et al. [2012} experiments. Deployment instructions can be found here: https://fluidfoam.readthedocs.io/en/latest/.

The 'roughWallFunctions' library, which contains the rough wall boundary condition introduced by Wilcox [2006] and revised by Fuhrman et al. [2010], is located in the /wallFunctions folder. This library is used as a reference in this work. See the section above for installation instructions.

The 'Fuhrman 2010' folder contains the numerical setups for simulating the fully developed boundary layer flow over a rough surface experiments conducted by Fuhrman et al. [2010] (see section 3 of the article).   
The '/Kikkert2012' folder contains the numerical setups for simulating the bore-driven swash experiments of Kikkert et al. [2012] in 2D and 3D (see section 4 of the article).   
The '/python' folder contains the post-processing codes used to generate the figures presented in the article.


Acknowledgments
---------------
This work was made possible thanks to a Contrat Doctoral Spécifique Normalien thesis grant from ENS Paris-Saclay, was carried out within the framework of the Oxalia Chair, supported by the Fondation Grenoble INP thanks to the patronage of Artelia, and is affiliated with LEGI.
The authors would like to thank Dr O'Donoghue for providing access to the experimental data reported by Kikkert et al. [2012].


Bibliography
---------------
* Fuhrman et al. [2010] : Fuhrman, D., M. Dixen, and N. Jacobsen, Physically-consistent wall boundary conditions for the k−ω turbulence model, Journal of Hydraulic Research, 48, 793–800, 2010.   
* Kikkert et al. [2012] : Kikkert, G., T. O’Donoghue, D. Pokrajac, and N. Dodd, Experimental study of bore-driven swash hydrodynamics onimpermeable rough slopes, Coastal Engineering, 60, 2012.   
* Wilcox [2006] : Wilcox, D., Turbulence modeling for cfd, third edition, DCW Industries, 2006.   

