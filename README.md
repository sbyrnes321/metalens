(C) 2015 Steven Byrnes

This is software for designing, optimizing, and simulating metasurface lenses (and metasurface beam deflectors). It was written for the purpose of doing calculations described the paper "Designing large, high-efficiency, high-numerical-aperture, transmissive meta-lenses for visible light" at http://arxiv.org/abs/1511.04781 Please refer to that paper for motivation and method. I thank the coauthors of that paper for valuable help and feedback, and I thank Osram and Draper for support.

Set up and requirements
=======================

The Python code was written in Python 3.4. The easiest way to install Python is Anaconda: https://www.continuum.io/downloads For more Python installation and getting-started details: http://sjbyrnes.com/?page_id=67

The actual electromagnetic simulation is done using S4, a rigorous coupled-wave analysis (RCWA) code. You need to download S4.exe from http://web.stanford.edu/group/fan/S4/install.html . The program expects grating.py, grating.lua, S4.exe, etc. to all be in the same folder. Then there should be a subfolder called "temp" (The python code will create it if it's not already there). The program writes configuration files (for communicating between python and S4) into "temp" and into subfolders of "temp". I set it up that way so that I could remove "temp" from dropbox. (The config files get changed by the program gizillions of times per minute, so it's silly for dropbox to try synchonizing them.)

The code is currently windows-only. I'm sure it can be easily modified for mac by downloading the mac version of S4 and changing the subprocess commands etc. It could be modified for linux in theory, but not really in practice, because the S4 website does not provide pre-compiled binaries for linux. You need to compile it yourself. The code has makefiles and instructions for compiling, but I guess the instructions must be full of mistakes, or very tricky. I could not get it to work, and other people have had problems too (see https://github.com/victorliu/S4/issues ). I mean, you're welcome to try, but if you can use the pre-compiled versions you'll potentially save a lot of time.

Using the code
==============

Now, to design a metasurface:

The first step is designing a set of individual "gratings" (i.e. unit cells of metasurface beam deflectors). The code is in grating.py. (Which in turn calls grating.lua, the script that directly interfaces with S4.) Create a Grating object however you want, then make it better by running optimize() and/or optimize2(). Read grating.lua for the details of what precise figure-of-merit you're optimizing, particularly what wavelength(s) and polarizations. Note that optimize() and optimize2() are very basic optimization algorithms; lots of room for improvement.

Anyway, now you have some optimized Gratings. Run vary_angle() (still in grating.py) to use this as the starting point for a GratingCollection, a group of gradually-varying gratings. You might need a few GratingCollections to fill out a complete lens, especially at high numerical aperture. vary_angle() runs from the inside of the lens towards the outside, so if you want a GratingCollection that runs from incident angle=20° to incident angle=30°, you start with a grating designed for incident angle=20°, not 30°.

One other ingredient is the center of the lens. If you go to lens_center.py, you'll find code to make a HexGridSet, i.e. a set of Grating objects corresponding to simple periodic nano-pillar structures on a hexagonal grid. The set consists of different choices of diameter for the pillars. Despite the name ("Grating"), these should generally be spaced closely enough that there are no extra propagating diffraction orders. If you run "hgs.characterize(...)" where hgs is a HexGridSet, then it will calculate and store the phase associated with different diameters, and then you know which pillar to put at which location.

With both the HexGridSet for the center, and the GratingCollections for the periphery, we put them together using design_collimator.py. This has functions to design a lens and store the design in a compact form, and also to create a list of nano-pillars for the whole lens, or even create DXF and SVG files of the design.

Next, you probably want to calculate the far-field. Run gc.characterize(...) for each GratingCollection gc, and also for the HexGridSet. This builds up a database of complex diffraction amplitudes as a function of incident angle, polarization, wavelength, and diffraction order. Run gc.build_interpolators() to create interpolating functions related to this database. Then finally use nearfield.py to calculate the nearfield based on that database, and then nearfield_farfield.py to do a nearfield-farfield transform.

Note on saving your work
========================

Anytime you find a good Grating or GratingCollection or HexGridSet that you want to remember for later, let's say "x", then run "print(x)" (or type "x" or in your ipython console) to get a specification of this object, in the form of a string of python code which recreates it. Then using copy-and-paste, you can make a python script file that recreates this grating instantaneously. So you don't have to start from scratch each time.

Note that the specification of a GratingCollection may be many megabytes long, particularly if you have run characterize() on it. (The characterize() data is saved. The interpolators are not saved, but it only takes a second to re-create them.) Anyway it's enough code that some syntax-highlighting text editors (like the one built into Spyder) will take hours to open the file. Sublime handles it pretty well, as does notepad++. 

List of files
=============

grating.py, lens_center.py, grating.lua, design_collimator.py, nearfield.py, nearfield_farfield.py were mentioned above.

grating_lumerical.lsf is a lumerical script file. It simulates the same gratings in lumerical for comparison with S4. (Use grating.py --> run_lumerical()). So far I've found they're pretty consistent, within 5 or 10% absolute efficiency. Warning: This probably doesn't work at the moment, because I changed some function definitions elsewhere in the code recently but didn't re-check this part.

S4conventions.py is some tests to understand the exact definition of "output amplitude" in S4, i.e. all the phase conventions, sign conventions, polarization conventions, etc. To run this you need to uncomment some lines in grating.lua.

refractive_index.py is just a little script where refractive index data for TiO2 and glass is stored.
