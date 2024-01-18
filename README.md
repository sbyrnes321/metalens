(C) 2015-2017 Steven Byrnes. This is free software, released under the MIT license - see bottom of this README.

This is software for designing, optimizing, and simulating metasurface lenses (and metasurface beam deflectors). It was written for the purpose of doing calculations described the paper "Designing large, high-efficiency, high-numerical-aperture, transmissive meta-lenses for visible light" at https://doi.org/10.1364/OE.24.005110 Please refer to that paper for motivation and method. I thank the coauthors of that paper for valuable help and feedback, and I thank Osram and Draper for support.

Set up and requirements
=======================

The Python code is for Python 3.4 or later. The easiest way to install Python is Anaconda: https://www.anaconda.com/download

The actual electromagnetic simulation is done using S4, a rigorous coupled-wave analysis (RCWA) code. You need to download S4.exe from http://web.stanford.edu/group/fan/S4/install.html . (UPDATE: Oops, the windows binary file S4.exe is no longer available from that website. I don’t know why. Luckily, I still have the old version of S4.exe that I used myself when writing my paper, and I have posted it at https://sjbyrnes.com/S4.exe .) The program expects `grating.py`, `grating.lua`, `S4.exe`, etc. to all be in the same folder. Then there should be a subfolder called `temp`. The program writes configuration files (for communicating between python and S4) into `temp` and into subfolders of `temp`. I set it up that way so that I could remove `temp` from dropbox. (The config files get changed by the program gizillions of times per minute, so it's silly for dropbox to synchronize them.)

The code is currently windows-only. I'm sure it can be easily modified for mac by downloading the mac version of S4 and changing the subprocess commands etc. It could be modified for linux in theory, but not really in practice, because the S4 website does not provide pre-compiled binaries for linux. You need to compile it yourself. The code has makefiles and instructions for compiling, but I guess the instructions must be full of mistakes, or very tricky. I could not get it to work, and other people have had problems too (see https://github.com/victorliu/S4/issues ). I mean, you're welcome to try, but if you can use the pre-compiled versions you'll potentially save a lot of time.

Using the code
==============

Now, to design a metasurface:

The first step is designing a set of individual "gratings" (i.e. unit cells of metasurface beam deflectors). The code is in `grating.py`. (Which in turn calls `grating.lua`, the script that directly interfaces with S4.) Create a Grating object however you want, then make it better by running `optimize()` and/or `optimize2()`. Read `grating.lua` for the details of what precise figure-of-merit you're optimizing, particularly what wavelength(s) and polarizations. Note that `optimize()` and `optimize2()` are very basic optimization algorithms; lots of room for improvement.

Anyway, now you have some optimized `Grating`s. Run `vary_angle()` (still in `grating.py`) to use this as the starting point for a `GratingCollection`, a group of gradually-varying gratings. You might need a few `GratingCollection`s to fill out a complete lens, especially at high numerical aperture. `vary_angle()` runs from the inside of the lens towards the outside, so if you want a `GratingCollection` that runs from incident angle=20° to incident angle=30°, you start with a grating designed for incident angle=20°, not 30°.

One other ingredient is the center of the lens. If you go to `lens_center.py`, you'll find code to make a `HexGridSet`, i.e. a set of `Grating` objects corresponding to simple periodic nano-pillar structures on a hexagonal grid. The set consists of different choices of diameter for the pillars. Despite the name (`Grating`), these should generally be spaced closely enough that there are no extra propagating diffraction orders. If you run `hgs.characterize(...)` where `hgs` is a `HexGridSet`, then it will calculate and store the phase associated with different diameters, and then you know which pillar to put at which location.

With both the `HexGridSet` for the center, and the `GratingCollection`s for the periphery, we put them together using `design_collimator.py`. This has functions to design a lens and store the design in a compact form, and also to create a list of nano-pillars for the whole lens, or even create DXF and SVG files of the design.

Next, you probably want to calculate the far-field. Run `gc.characterize(...)` for each `GratingCollection` `gc`, and also for the `HexGridSet`. This builds up a database of complex diffraction amplitudes as a function of incident angle, polarization, wavelength, and diffraction order. Run `gc.build_interpolators()` to create interpolating functions related to this database. Then finally use `nearfield.py` to calculate the near-field based on that database, and then `nearfield_farfield.py` to do a nearfield-farfield transform.

Note on saving your work
========================

Any time you find a good Grating or GratingCollection or HexGridSet that you want to remember for later, let's say `x`, then run `print(x)` (or type `x` or in your ipython console) to get a specification of this object, in the form of a string of python code which recreates it. Then using copy-and-paste, you can make a python script file that recreates this grating instantaneously. So you don't have to start from scratch each time.

Note that the specification of a `GratingCollection` may be many megabytes long, particularly if you have run `characterize()` on it. (The `characterize()` data is saved. The interpolators are not saved, but it only takes a second to re-create them.) Anyway it's enough code that some syntax-highlighting text editors (like the one built into Spyder) will take hours to open the file. Sublime handles it pretty well, as does notepad++. 

List of files
=============

grating.py, lens_center.py, grating.lua, design_collimator.py, nearfield.py, nearfield_farfield.py were mentioned above.

grating_lumerical.lsf is a lumerical script file. It simulates the same gratings in lumerical for comparison with S4. (Use grating.py --> run_lumerical()). So far I've found they're pretty consistent, within 5 or 10% absolute efficiency. Warning: This probably doesn't work at the moment, because I changed some function definitions elsewhere in the code recently but didn't re-check this part.

S4conventions.py is some tests to understand the exact definition of "output amplitude" in S4, i.e. all the phase conventions, sign conventions, polarization conventions, etc. To run this you need to uncomment some lines in grating.lua.

refractive_index.py is just a little script where refractive index data for TiO2 and glass is stored.

Getting-started example (but the discussion above is more comprehensive)
========================================================================

**One-time setup:** Clone this repository, then download S4.exe into that same directory (see above), and also add an empty subfolder to that directory called "temp".

**Now open `grating.lua`:** What aspect of the gratings is it set to optimize? This is set mainly in the `display_fom` ("display the figure of merit") function. As of this writing, it is set to optimize a grating that sends 0.580μm light into the -1 order and 0.450μm light into the 0 order. That's random, it's just the last thing I wanted to calculate before I uploaded this file. You probably want to do something different. So edit it! (Other example figures-of-merit are commented out nearby.) The script is in Lua, which is an extremely simple programming language (if you're not sure what a command does, just google-search it), and the S4-specific commands are explained at http://web.stanford.edu/group/fan/S4/lua_api.html . Note that there are default refractive indices for glass and TiO2 in little lists at the top, but they only work for the specific wavelengths on the list; if you're interested in other wavelengths or materials you need to edit that code, or input the refractive indices in Python. Let's say in this example that I'm only interested in 785nm light, so I make some edits to the file to use the 785nm refractive indices and calculate the grating efficiency for 785nm. (Exact edits not shown.)

**Now open Spyder (or other python IDE) and do the following:**

```python
# First run grating.py in your ipython console.
# Then in the same console:
from numericalunits import nm
from numpy import array
from math import pi
degree = pi/180
# This example grating has two nanopillars per period.
# I picked the starting parameters arbitrarily. Note that this particular
# example will end up working very inefficiently; I'm just illustrating the
# commands here.
mygrating=Grating(lateral_period=560*nm, cyl_height=500*nm,
                  target_wavelength=785*nm, angle_in_air=65*degree,
                  xyrra_list_in_nm_deg=array([[0.,   0., 200., 150., 0.],
      	                                      [400., 280., 150., 200., 10.]]))
optimize(mygrating, target_wavelength=785*nm)
```

...this runs and runs, and I can either let it complete, or maybe after a while decide the structure is good enough. Then I can copy-paste the latest configuration from the console output (editing the variable name):

```python
mygrating_optimized=Grating(lateral_period=560.0*nm, grating_period=866.1516663855559*nm, cyl_height=500.0*nm, n_glass=0, n_tio2=0, xyrra_list_in_nm_deg=np.array([[-2.15000000e+02,2.00000000e+00,2.44000000e+02,1.11000000e+02,-1.52666625e-13],[1.96000000e+02,-2.78000000e+02,1.47000000e+02,2.30000000e+02,1.00000000e-01]]), data=None)
```

(Maybe I'll also save this line in a separate text file for safekeeping.)

To see what this looks like:

```python
mygrating_optimized.show_config()
```

Now let's say I want a family of smoothly-varying gratings designed for incident angle spanning the range from 65° to 70°, to be tiled into a round (as opposed to cylindrical) lens. I can run:

```python
vary_angle(start_grating=mygrating_optimized, end_angle=70*degree, lens_type='round', target_wavelength=785*nm)
```

This will run for quite a while, and eventually print a `GratingCollection` object. Again, I copy-paste from the console output into a variable name of my choice, and probably also copy it into a separate text file for safekeeping:

```python
my_gratingcollection_65_to_70 = GratingCollection(target_wavelength=785.0*nm, lateral_period=261.13228856679905*nm, lens_type='round', grating_list= [Grating(lateral_period=565.6*nm, grating_period=864.6262219924635*nm, cyl_height=500.0*nm, n_glass=0, n_tio2=0, xyrra_list_in_nm_deg=np.array([[-2.15907032e+02,2.01782243e+00,2.45133589e+02,1.11028705e+02,1.87512868e-02],[1.91953580e+02,-2.80871711e+02,1.46444938e+02,2.30320164e+02,4.59054597e-02]]), data=None), Grating(lateral_period=560.0*nm, grating_period=866.1516663855559*nm, cyl_height=500.0*nm, n_glass=0, n_tio2=0, xyrra_list_in_nm_deg=np.array([[-2.15000000e+02,2.00000000e+00,2.44000000e+02,1.11000000e+02,-1.52666625e-13],[1.96000000e+02,-2.78000000e+02,1.47000000e+02,2.30000000e+02,1.00000000e-01]]), data=None), ... etc. etc. etc.])
```

If I want to later calculate a lens far-field incorporating this `GratingCollection`, I need to characterize its transmission properties using

```python
my_gratingcollection_65_to_70.characterize(785*nm, numG=30)
```

(`numG` controls the speed and accuracy of the RCWA, I lowered it here because I'm in a hurry.)

Now `my_gratingcollection_65_to_70` has its diffraction efficiency data stored in it. I type into the console `my_gratingcollection_65_to_70` or `print(my_gratingcollection_65_to_70)`, and again probably save the results in a separate text-file for safekeeping, so I don't need to repeat the time-consuming calculation next time. Note that this command may be very long, see "Note on saving your work" above.

OK, hopefully that's enough to get you started; you can email me with further questions, or if you expand this tutorial yourself and send it to me, I'll post it for everyone else's benefit.

License
=======

Copyright (c) 2015-2017 Steven Byrnes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
