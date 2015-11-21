# -*- coding: utf-8 -*-
"""
(C) 2015 Steven Byrnes

Design and optimize metasurface gratings.

"""
import math, os, matplotlib, shutil
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import subprocess
from subprocess import PIPE
import random
import cmath
import string
from scipy.interpolate import RegularGridInterpolator

# my units package
# https://pypi.python.org/pypi/numericalunits
# just run the command "pip install numericalunits"
import numericalunits as nu
from numericalunits import m, nm, um

pi = math.pi
inf = float('inf')
degree = pi / 180

# we assume that S4 and grating.lua are in the same folder as this python file.
# S4 will look for the inputs files (i.e.,  files specifying cylinder locations
# and other parameters) in a subfolder of the current working directory (cwd)
# called "temp". We can set the cwd to whatever we want in order to run multiple
# optimizations simultaneously (generally, each in its own independent IPython
# interpreter, though some parts of the python code will spawn multiple
# subprocesses that run in parallel.)

def cwd_for_S4(subfolder=None):
    """cwd (current working directory) in which to run S4. Use a subfolder
    if running multiple instances in parallel. Subfolder can be any string."""
    here = os.path.dirname(os.path.realpath('__file__'))
    if subfolder is None:
        return here
    else:
        return os.path.join(here, 'temp', subfolder)

def path_to_temp(subfolder=None):
    """cwd of S4 has a subfolder named "temp" where the data-files go. Also,
    this function creates relevant folders if they don't already exist, and
    copies grating.lua if necessary"""
    path = os.path.join(cwd_for_S4(subfolder), 'temp')
    if subfolder is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        src = os.path.join(cwd_for_S4(), 'grating.lua')
        dst = os.path.join(cwd_for_S4(subfolder), 'grating.lua')
        if (not os.path.isfile(dst) or os.path.getmtime(src) > os.path.getmtime(dst)):
            shutil.copyfile(src=src, dst=dst)
    return path

def remove_subfolder(subfolder):
    assert subfolder not in (None, '')
    shutil.rmtree(cwd_for_S4(subfolder))

def random_subfolder_name():
    return ''.join(random.choice(string.ascii_letters + string.digits)
                    for _ in range(6))
    
def xyrra_filename(subfolder=None, index=None):
    """This file is used to communicate the posititons of the cylinders to
    S4 or lumerical.
    Use a subfolder to run many S4 optimizations at once.
    The index is for running a batch of calculations using lumerical."""
    filename = ('grating_xyrra_list'
                 + (str(index) if index is not None else '')
                 + '.txt')
    return os.path.join(path_to_temp(subfolder), filename)

def setup_filename(subfolder=None, index=None):
    """This file is used to communicate how the simulation is set up to
    S4 or lumerical.
    Use a subfolder to run many S4 optimizations at once.
    The index is for running a batch of calculations using lumerical."""
    filename = ('grating_setup'
                 + (str(index) if index is not None else '')
                 + '.txt')
    return os.path.join(path_to_temp(subfolder), filename)



class Grating:
    """A Grating object has all the information needed to specify a grating.
    including the locations and sizes of the cylinders and the periodicity.
    
    * xyrra_list is a list of [x at center, y at center, semi-axis in x
    direction, semi-axis in y direction, rotation angle (positive is CCW)] for
    each ellipse in the pattern. Internally, it's stored with lengths in
    "numericalunits" (see https://pypi.python.org/pypi/numericalunits) and
    angles are in radians.
    
    * cyl_height is the height of each nano-pillar
    
    * The periodic cell has size grating_period × lateral_period. The former is
    larget and is supposed to bend light to the +1 diffraction order (or -1
    depending on your conventions.)
    
    * n_glass and n_tio2 are refractive index of the substrate and pillars
    respectively. Set to 0 to use measured / literature dispersion curves.
    
    Instead of specifying grating_period directly, you can also specify
    angle_in_air, the angle that light is traveling while in the air
    (relative to normal), before hitting the meta-lens and traveling through
    glass (hopefully) at normal. Then you need to also specify
    target_wavelength.
    """
    def __init__(self, lateral_period, cyl_height, grating_period=None, 
                 target_wavelength=None, angle_in_air=None,
                 n_glass=0, n_tio2=0, xyrra_list_in_nm_deg=None, data=None):
        """supply EITHER grating_period OR (angle_in_air and target_wavelength)
        which allows you to compute grating_period. But grating_period is the
        only parameter saved as a property, not angle_in_air nor target_wavelength.
        
        Set n_glass and/or n_tio2 to zero to use tabulated values"""
        if grating_period is not None:
            assert (target_wavelength is None) and (angle_in_air is None)
            self.grating_period = grating_period
        else:
            self.grating_period = target_wavelength / math.sin(angle_in_air)
        self.n_glass = n_glass
        self.n_tio2 = n_tio2
        self.lateral_period = lateral_period
        self.cyl_height = cyl_height
        
        self.grating_kx = 2*pi / self.grating_period
        
        if xyrra_list_in_nm_deg is not None:
            self.xyrra_list = xyrra_list_in_nm_deg.copy()
            self.xyrra_list[:,0:4] *= nm
            self.xyrra_list[:,4] *= degree
        if data is not None:
            self.data = data
    
    def get_xyrra_list(self, units=None, replicas=None):
        """Get a copy of the xyrra_list.
        
        For units, None means "as stored interally", or 'um,deg' means xyrr
        are in microns and a is in degrees, or 'nm,deg' is similar.
        
        For replicas: For python internal storage and for lua, we only need one
        periodic replica of each ellipse, whereas for lumerical and display we
        need multiple periodic replicas. With replicas=True, include every
        ellipse that sticks into the central unit cell. With replicas=N,
        include every ellipse that is within N unit cells of the center. (To
        allow zooming out in a display.)"""
        if replicas is not None:
            # N is number of unit cells away from center to include
            N = (0 if replicas is True else replicas)
            grating_period = self.grating_period
            lateral_period = self.lateral_period
            new_xyrra_list = []
            for x,y,rx,ry,a in self.xyrra_list:
                for translate_x in range(-(N+1), N+2):
                    for translate_y in range(-(N+1), N+2):
                        x_center = x + translate_x * grating_period
                        y_center = y + translate_y * lateral_period
                        ellipse_pt_list = ellipse_pts(x_center, y_center, rx, ry, a, num_points=120)
                        if any(abs(x) < grating_period/2 + N*grating_period
                                        and abs(y) < lateral_period/2 + N*lateral_period
                                              for x,y in ellipse_pt_list):
                            new_xyrra_list.append([x_center, y_center, rx, ry, a])
            new_xyrra_list = np.array(new_xyrra_list)
        else:
            new_xyrra_list = self.xyrra_list.copy()
        if units is None:
            return new_xyrra_list
        if units == 'nm,deg':
            new_xyrra_list[:,0:4] /= nm
            new_xyrra_list[:,4] /= degree
            return new_xyrra_list
        if units == 'um,deg':
            new_xyrra_list[:,0:4] /= um
            new_xyrra_list[:,4] /= degree
            return new_xyrra_list
        raise ValueError('bad units specification')
    
    @property
    def xyrra_list_in_nm_deg(self):
        """Expresses x,y,r,r in nm and angle in degrees"""
        return self.get_xyrra_list(units='nm,deg')
    
    @property
    def xyrra_list_in_um_deg(self):
        """Expresses x,y,r,r in microns and angle in degrees"""
        return self.get_xyrra_list(units='um,deg')
    
    def get_angle_in_air(self, target_wavelength):
        """If this grating were part of a lens designed for target_wavelength,
        then it would be located at a place where light in air is going at
        this angle. (Then the light would be bent to normal in glass.)"""
        if self.grating_period < target_wavelength:
            raise ValueError('bad inputs!', target_wavelength/nm, self.grating_period/nm)
        return math.asin(target_wavelength / self.grating_period)
    
    def write(self, angle_in_air=None, subfolder=None, index=None, replicas=False,
              ux_min=None, ux_max=None, uy_min=None, uy_max=None, u_steps=None,
              wavelength=None, numG=50):
        """save simulation setup and xyrra_list to a standard file, which
        either lumerical or S4 can read.
        
        The "replicas" parameter is defined as in self.get_xyrra_list(). TODO - need for lumerical
        
        angle_in_air is the angle from which the light is coming at the
        metasurface. In radians.
        
        numG is the number of modes used in RCWA. More is slower but more
        accurate.
        
        index is appended to the filename, this is used for batch lumerical simulations.
        
        The following parameters are used ONLY for far-field calculation, not
        figure-of-merit for optimizations:
        * wavelength is self-explanatory (in numericalunits)
        * ux_min, ux_max, uy_min, uy_max is the range of incident light angles
          to test ... the light is traveling in the direction with unit vector
          (ux,uy,uz).
        * u_steps is the number of increments for varying ux and uy (each is
          the same, at least for now)
        
        """
        filename = setup_filename(subfolder=subfolder, index=index)
        with open(filename, 'w') as f:
            if angle_in_air is not None:
                # calculate figure-of-merit
                assert all(x is None for x in (ux_min, ux_max, uy_min, uy_max,
                                               u_steps, wavelength))
                print(1, file=f)
                print(self.grating_period / m, file=f)
                print(self.lateral_period / m, file=f)
                print(angle_in_air, file=f)
                print(self.n_glass, file=f)
                print(self.n_tio2, file=f)
                print(self.cyl_height / m, file=f)
                print(numG, file=f)
            else:
                assert all(x is not None for x in (ux_min, ux_max, uy_min, uy_max,
                                               u_steps, wavelength))
                print(2, file=f)
                print(self.grating_period / m, file=f)
                print(self.lateral_period / m, file=f)
                print(self.n_glass, file=f)
                print(self.n_tio2, file=f)
                print(self.cyl_height / m, file=f)
                print(numG, file=f)
                print(ux_min, file=f)
                print(ux_max, file=f)
                print(uy_min, file=f)
                print(uy_max, file=f)
                print(u_steps, file=f)
                print(round(wavelength/nm)/1000, file=f)
                
        np.savetxt(xyrra_filename(subfolder=subfolder, index=index),
                   self.xyrra_list_in_um_deg, delimiter=' ')
    
    def __repr__(self):
        """this is important - after we spend 5 minutes finding a nice
        Grating, let's say "mygrating", we can just type
        "mygrating" or "repr(mygrating)" into IPython and copy the
        resulting text. That's the code to re-create that grating
        immediately, without re-calculating it."""
        xyrra_list_str = (np.array2string(self.xyrra_list_in_nm_deg, separator=',')
                                         .replace(' ', '').replace('\n',''))
        return ('Grating(lateral_period=' + repr(self.lateral_period/nm) + '*nm'
                + ', grating_period=' + repr(self.grating_period/nm) + '*nm'
                + ', cyl_height=' + repr(self.cyl_height/nm) + '*nm'
                + ', n_glass=' + repr(self.n_glass)
                + ', n_tio2=' + repr(self.n_tio2)
                + ', xyrra_list_in_nm_deg=np.array(' + xyrra_list_str + ')'
                + ', data=' + (repr(self.data) if hasattr(self,'data') else 'None')
                + ')')
    
    def copy(self):
        return eval(repr(self))
    
    def run_lua(self, target_wavelength=None, subfolder=None, **kwargs):
        """output the parameters for a simulation, then run the lua S4 script and
        return the result.
        Target_wavelength is important for determining what angle the light is
        coming at. (For collimator applications, this angle applies to all
        wavelengths.) This is only used for calculating figure-of-merit; for
        characterize() leave it out
        To run many in parallel, use run_lua_initiate() and run_lua_getresult()
        separately. The former starts the simulation running but doesn't wait
        for it to finish; instead it returns the process (Popen) object. Then
        the latter reads the result.
        kwargs are optional arguments passed through to self.write(). Used for
        self.characterize()"""
        process = self.run_lua_initiate(target_wavelength=target_wavelength,
                                        subfolder=subfolder, **kwargs)
        return self.run_lua_getresult(process)
        
    def run_lua_initiate(self, target_wavelength=None, subfolder=None, **kwargs):
        """See self.run_lua() for information"""
        angle_in_air = None if target_wavelength is None else self.get_angle_in_air(target_wavelength)
        self.write(angle_in_air=angle_in_air, subfolder=subfolder, **kwargs)
        cwd = cwd_for_S4(subfolder=subfolder)
        return subprocess.Popen(['S4', 'grating.lua'],
                                 cwd=cwd, stdout=PIPE, stderr=PIPE)

    def run_lua_getresult(self, process):
        """See self.run_lua() for information"""
        output, error = process.communicate()
        try:
            return float(output)
        except ValueError:
            print('Cannot convert output to float!')
            print('S4 output was:', repr(output.decode()))
            print('S4 error was:', repr(error.decode()))
            return 0

    def run_lumerical(self, target_wavelength):
        """output the parameters for a simulation. Can't actually run the
        simulation though, need to open grating_lumerical.lsf and run it manually,
        as far as I know.
        Target_wavelength is important for determining what angle the light is
        coming at. (For collimator applications, this angle applies to all
        wavelengths.)"""
        angle_in_air = self.get_angle_in_air(target_wavelength)
        self.write(angle_in_air=angle_in_air, index=0, replicas=True)
        if os.path.isfile(xyrra_filename(index=1)):
            os.remove(xyrra_filename(index=1))
            os.remove(setup_filename(index=1))
    
    def standardize(self):
        """ pick the appropriate periodic replica of each item. Overwrite original """
        grating_period = self.grating_period
        lateral_period = self.lateral_period
        xyrra_list = self.xyrra_list
        xyrra_list[:,0] %= grating_period
        xyrra_list[(xyrra_list[:,0]>grating_period/2),0] -= grating_period
        xyrra_list[:,1] %= lateral_period
        xyrra_list[(xyrra_list[:,1]>lateral_period/2),1] -= lateral_period
        xyrra_list[:,4] %= 2*pi
        xyrra_list[(xyrra_list[:,4]>pi),4] -= 2*pi

    def show_config(self):
        grating_period = self.grating_period
        lateral_period = self.lateral_period
        plt.figure()
        plt.xlim(-grating_period / nm, grating_period / nm)
        plt.ylim(-lateral_period / nm, lateral_period / nm)
        # add lotsa periodic replicas to allow zooming out
        for x,y,rx,ry,a in self.get_xyrra_list(replicas=3):
            circle = matplotlib.patches.Ellipse((x / nm, y / nm),
                                                 2*rx / nm, 2*ry / nm,
                                                 angle=a / degree,
                                                 color='k', alpha=0.5)
            plt.gcf().gca().add_artist(circle)
        rect = matplotlib.patches.Rectangle((-grating_period/2 / nm,-lateral_period/2 / nm),
                                            grating_period / nm, lateral_period / nm,
                                            facecolor='none', linestyle='dashed',
                                            linewidth=2, edgecolor='red')
        plt.gcf().gca().add_artist(rect)
        plt.gcf().gca().set_aspect('equal')
    
    def characterize(self, subfolder=None, process=None,
                     ux_min=None, ux_max=None, uy_min=-0.2, uy_max=0.2,
                     u_steps=3, wavelength=580*nm, numG=100, convert_to_xy=True,
                     just_normal=False):
        """Calculate grating output as a function of incoming angle with
        grating.lua, then store the result in self.data.
        Initiate the process separately if you want to run many in parallel
        (See GratingCollection.characterize for an example.)
        
        ampfy, ampfx, ampry, amprx are the outgoing amplitudes for the "x" and
        "y" polarizations (see S4conventions for what that means) in each of
        the forward ("f") and reflected ("r") directions.
        
        convert_to_xy switches the two incoming polarizations from 's' and 'p'
        (used by grating.lua) to 'x' and 'y' (see S4conventions.py for
        definitions). The advantage is that x and y are defined in a smoothly-
        varying way across all propagating directions, whereas s and p switch
        directions sharply near normal, making interpolation tricky.
        
        If just_normal is True, the ux,uy inputs are ignored, and instead we
        only calculate for normal incidence. We put the data in a format so that
        the usual interpolation code still works, in particular we copy the
        same data into (ux,uy)=(0.001,0.001), (0.001,-0.001), etc. straddling the
        normal direction
        """
        if process is None:
            if just_normal is True:
                ux_min = ux_max = uy_min = uy_max = 0.001
                u_steps = 1
            else:
                if ux_min is None:
                    target_ux = self.get_angle_in_air(580*nm)
                    ux_min = max(-0.99, target_ux - 0.2)
                if ux_max is None:
                    target_ux = self.get_angle_in_air(580*nm)
                    ux_max = min(0.99, target_ux + 0.2)
            process = self.run_lua_initiate(subfolder=subfolder, ux_min=ux_min,
                                            ux_max=ux_max, uy_min=uy_min,
                                            uy_max=uy_max, u_steps=u_steps,
                                            wavelength=wavelength, numG=numG)
        
        output, error = (x.decode() for x in process.communicate())
        if error is not '':
            raise ValueError(error)
        all_data = []
        for line in output.split('\n'):
            split = line.split()
            #print(split)
            if len(split) > 0:
                assert len(split) == 14
                all_data.append({'wavelength_in_nm': float(split[0]),
                                 's_or_p': split[1],
                                 'ux':float(split[2]),
                                 'uy':float(split[3]),
                                 'ox':int(split[4]),
                                 'oy':int(split[5]),
                                 'ampfy':float(split[6]) + 1j * float(split[7]),
                                 'ampfx':float(split[8]) + 1j * float(split[9]),
                                 'ampry':float(split[10]) + 1j * float(split[11]),
                                 'amprx':float(split[12]) + 1j * float(split[13])})
        if convert_to_xy is True:
            all_data_xy = []
            for ep in (x for x in all_data if x['s_or_p'] == 'p'):
                temp = [x for x in all_data if x['wavelength_in_nm']==ep['wavelength_in_nm']
                                              and (x['ux'],x['uy'])==(ep['ux'],ep['uy'])
                                              and (x['ox'],x['oy'])==(ep['ox'],ep['oy'])
                                              and x['s_or_p']=='s']
                assert len(temp)==1
                es = temp[0]
                
                # Now I have a matching (s,p) pair. Take a linear combination
                # to find the results I would get from x and y incident
                # polarization.
                # Since this only concerns the incident light, use formulas
                # with n=1, kx = kx_incident, etc.
                k = 2*pi / (es['wavelength_in_nm'] * nm)
                kx = k*ep['ux']
                ky = k*ep['uy']
                # at kx==ky==0, s and p are undefined. In practice, -0.0 is
                # different than +0.0. It's fraught, best to just forbid it.
                assert 0 < kx**2 + ky**2 <= k**2
                kz = (k**2 - kx**2 - ky**2)**0.5
                
                # see S4conventions.py for the following four lines
                x_p_coef = kx/(kx**2+ky**2)**0.5
                x_s_coef = -ky*k/(kz*(kx**2+ky**2)**0.5)
                y_p_coef = -ky/(kx**2+ky**2)**0.5
                y_s_coef = -kx*k/(kz*(kx**2+ky**2)**0.5)
                
                ex = {key:ep[key] for key in ep if key in ('wavelength_in_nm',
                                                           'ux','uy','ox','oy')}
                ey = {key:ep[key] for key in ep if key in ('wavelength_in_nm',
                                                           'ux','uy','ox','oy')}
                ex['x_or_y'] = 'x'
                ey['x_or_y'] = 'y'
                for a in ('ampfy','ampfx','ampry','amprx'):
                    ex[a] = x_p_coef * ep[a] + x_s_coef * es[a]
                    ey[a] = y_p_coef * ep[a] + y_s_coef * es[a]
                all_data_xy.append(ex)
                all_data_xy.append(ey)
            if just_normal:
                assert all(e['ux'] == 0.001 for e in all_data_xy)
                assert all(e['uy'] == 0.001 for e in all_data_xy)
                for entry in all_data_xy.copy():
                    for ux_sign, uy_sign in [(-1,1), (-1,-1), (1,-1)]:
                        entry2 = entry.copy()
                        entry2['ux'] *= ux_sign
                        entry2['uy'] *= uy_sign
                        all_data_xy.append(entry2)
            self.data = all_data_xy
        else:
            # didn't yet code the just_normal, don't-convert-to-xy case
            assert just_normal is False
            
            self.data = all_data
    
def show_characterization(mygrating):
    all_data = mygrating.data
    ux_list = sorted({x['ux'] for x in all_data})
    uy_list = sorted({x['uy'] for x in all_data})
    my_order = (0,0)
    my_pol = 'x'
    my_wavelength = 580
    
    all_data_filtered = [x for x in all_data if x['x_or_y'] == my_pol
                                             and x['ox'] == my_order[0]
                                             and x['oy'] == my_order[1]
                                             and x['wavelength_in_nm'] == my_wavelength]
    data = np.zeros(shape=(len(ux_list), len(uy_list)), dtype=complex) + np.nan
    
    for entry in all_data_filtered:
        ix = ux_list.index(entry['ux'])
        iy = uy_list.index(entry['uy'])
        assert np.isnan(data[ix,iy])
        data[ix,iy] = entry['amprx']
    
    plt.figure()
    #plt.imshow(np.angle(data).T, interpolation='none', extent=((min(ux_list),max(ux_list),min(uy_list),max(uy_list))))
    plt.imshow(np.abs(data).T, interpolation='none', extent=((min(ux_list),max(ux_list),min(uy_list),max(uy_list))))
    plt.xlabel('ux (x-component of unit vector of incoming light direction)')
    plt.ylabel('uy (y-component of unit vector of incoming light direction)')
    plt.colorbar()
    
    

min_diameter = 100*nm
min_distance = 100*nm

def sq_distance_mod(x0,y0,x1,y1,x_period,y_period):
    """squared distance between two points in a 2d periodic structure"""
    dx = min((x0 - x1) % x_period, (x1 - x0) % x_period)
    dy = min((y0 - y1) % y_period, (y1 - y0) % y_period)
    return dx*dx + dy*dy

def distance_mod(x0,x1,period):
    """1d version - distance between two points in a periodic structure"""
    return min((x0 - x1) % period, (x1 - x0) % period)

def validate(mygrating, print_details=False, similar_to=None, how_similar=None):
    """ make sure the structure can be fabricated, doesn't self-intersect, etc.
        If similar_to is provided, it's an xyrra_list describing a configuration
    that we need to resemble, and how_similar should be a factor like 0.02
    meaning the radii etc. should change by less than 2%."""
    xyrra_list = mygrating.xyrra_list
    if xyrra_list[:,[2,3]].min() < min_diameter/2:
        if print_details:
            print('a diameter is too small')
        return False
#    if xyrra_list[:,3].max() > max_y_diameter/2:
#        if print_details:
#            print('a y-diameter is too big')
#        return False
    
    # Check that no two shapes are excessively close
    # points_list[i][j,k] is the x (if k=0) or y (if k=1) coordinate
    # of the j'th point on the border of the i'th shape
    points_per_ellipse = 100
    num_ellipses = xyrra_list.shape[0]
    points_arrays = []
    for i in range(num_ellipses):
        points_arrays.append(ellipse_pts(*xyrra_list[i,:], num_points=points_per_ellipse))
    
    # first, check each shape against its own periodic replicas, in the
    # smaller (y) direction. I assume that shapes will not be big enough to 
    # approach their periodic replicas
    for i in range(num_ellipses):
        i_pt_list = points_arrays[i]
        j_pt_list = i_pt_list.copy()
        j_pt_list[:,1] += mygrating.lateral_period
        for i_pt in i_pt_list:
            for j_pt in j_pt_list:
                if (i_pt[0] - j_pt[0])**2 + (i_pt[1] - j_pt[1])**2 < min_distance**2:
                    if print_details:
                        print('too close, between ellipse', i, 'and its periodic replica')
                        print(i_pt)
                        print(j_pt)
                        mygrating.show_config()
                        plt.plot(i_pt[0]/nm, i_pt[1]/nm, 'r.')
                        plt.plot(j_pt[0]/nm, j_pt[1]/nm, 'r.')
                    return False
    
    for i in range(1,num_ellipses):
        i_pt_list = points_arrays[i]
        for j in range(i):
            j_pt_list = points_arrays[j]
            for i_pt in i_pt_list:
                for j_pt in j_pt_list:
                    if sq_distance_mod(i_pt[0], i_pt[1], j_pt[0], j_pt[1],
                                       mygrating.grating_period, mygrating.lateral_period) < min_distance**2:
                        if print_details:
                            print('too close, between ellipse', j, 'and', i)
                            print(i_pt)
                            print(j_pt)
                            mygrating.show_config()
                            plt.plot(i_pt[0]/nm, i_pt[1]/nm, 'r.')
                            plt.plot(j_pt[0]/nm, j_pt[1]/nm, 'r.')
                        return False
    if similar_to is not None:
        for i in range(num_ellipses):
            if max(abs(xyrra_list[i, 2:4] - similar_to[i, 2:4]) / similar_to[i, 2:4]) > how_similar:
                if print_details:
                    print('A radius of ellipse', i, 'changed too much')
                return False
            if distance_mod(xyrra_list[i,0], similar_to[i,0], mygrating.grating_period) > how_similar * mygrating.grating_period:
                if print_details:
                    print('x-coordinate of ellipse', i, 'changed too much')
                return False
            if distance_mod(xyrra_list[i,1], similar_to[i,1], mygrating.lateral_period) > how_similar * mygrating.lateral_period:
                if print_details:
                    print('y-coordinate of ellipse', i, 'changed too much')
                return False
            if distance_mod(xyrra_list[i,4], similar_to[i,4], 2*pi) > how_similar * (2*pi):
                if print_details:
                    print('rotation of ellipse', i, 'changed too much')
                return False
    return True

def resize(oldgrating, newgrating_shell):
    """the vary_angle() routine gradually changes the periodicity of the
    grating. As a starting condition, the simplest thing would be to use the
    old xyrra_list with the new periodicity. But it's possible in this case
    for the grating to fail validate().
    
    For the cylindrical lens, in vary_angle(), lateral_period is fixed and
    grating_period increases, so there's no possible problem.
    
    For the round lens, in vary_angle(), lateral_period is increasing while
    grating_period is decreasing. The latter could cause a problem. But usually
    there is a big gap somewhere along the x dimension, so we can shrink that
    gap a bit.
    
    oldgrating is the previous grating, which passed validation.
    newgrating_shell is the new grating with no xyrra_list yet. We will fill
    it out and return a fresh grating"""
    oldgrating = oldgrating.copy()
    oldgrating.standardize()
    g = newgrating_shell.copy()
    g.xyrra_list = oldgrating.xyrra_list.copy()
    if validate(g) is True:
        return g
    
    old_grating_period = oldgrating.grating_period
    new_grating_period = g.grating_period
    
    assert new_grating_period < old_grating_period
    assert g.lateral_period >= oldgrating.lateral_period
    
    # look for places to try cutting horizontal space, i.e. x-coordinates with
    # a lot of clearance to the nearest pillar
    try_cutting_list = np.linspace(-old_grating_period/2, old_grating_period/2,
                                   num=100, endpoint=False)
    clearance_list = np.zeros_like(try_cutting_list) + np.inf
    for xc,yc,rx,ry,a in oldgrating.xyrra_list:
        for x,y in ellipse_pts(xc,yc,rx,ry,a,num_points=80):
            for i, xnow in enumerate(try_cutting_list):
                clearance_list[i] = min(clearance_list[i],
                                        distance_mod(xnow, x, old_grating_period))
    x_to_cut_at = try_cutting_list[np.argmax(clearance_list)]
    
    for i in range(g.xyrra_list.shape[0]):
        if g.xyrra_list[i,0] > x_to_cut_at:
            g.xyrra_list[i,0] -= (old_grating_period - new_grating_period)
    
    assert validate(g, print_details=True)
    return g
        


def correct_imshow_extent(array, min_px_center_x, max_px_center_x,
                          min_px_center_y, max_px_center_y):
    """imshow extent specifies the left side of the leftmost pixel etc. I want
    to instead say what the coordinate is at the center of the leftmost pixel etc."""
    nx = array.shape[1]
    ny = array.shape[0]
    px_extent_x = (max_px_center_x - min_px_center_x) / (nx-1)
    px_extent_y = (max_px_center_y - min_px_center_y) / (ny-1)
    return [min_px_center_x - px_extent_x/2,
            max_px_center_x + px_extent_x/2,
            min_px_center_y - px_extent_y/2,
            max_px_center_y + px_extent_y/2]


def ellipse_pts(x_center, y_center, r_x, r_y, angle, num_points=80):
    """return a list of (x,y) coordinates of points on an ellipse, in CCW order"""
    xy_list = np.empty(shape=(num_points,2))
    theta_list = np.linspace(0,2*pi,num=num_points, endpoint=False)
    for i,theta in enumerate(theta_list):
        dx0 = r_x * math.cos(theta)
        dy0 = r_y * math.sin(theta)
        xy_list[i,0] = x_center + dx0 * math.cos(angle) - dy0 * math.sin(angle)
        xy_list[i,1] = y_center + dx0 * math.sin(angle) + dy0 * math.cos(angle)
    if False:
        # test
        plt.figure()
        plt.plot(x_center,y_center,'r.')
        for x,y in xy_list:
            plt.plot(x,y,'k.')
        plt.gca().set_aspect('equal')
    return xy_list


def optimize(mygrating_start, target_wavelength, similar_to=None, how_similar=None,
             subfolder=None, numG=50):
    """optimize by varying parameters one after the other.
    If similar_to is provided, it's an xyrra_list describing a configuration
    that we need to resemble, and how_similar should be a factor like 0.02
    meaning the radii etc. should change by less than 2%.
    
    Target_wavelength is not what the simulation is run at (which is determined
    in grating.lua), but rather for determining what angle the light is coming
    in at (which is the same for all wavelengths, in a collimator).
    
    Return optimized Grating object (input object is not altered)."""
    assert validate(mygrating_start, print_details=True,
                    similar_to=similar_to, how_similar=how_similar)
    mygrating = mygrating_start.copy()
    xyrra_list = mygrating.xyrra_list
    fom_now = mygrating.run_lua(subfolder=subfolder,
                                target_wavelength=target_wavelength, numG=numG)
    print('fom now...', fom_now, flush=True)
    found_optimum = False
    things_to_try_changing = [(i,j) for i in range(xyrra_list.shape[0])
                                                for j in range(xyrra_list.shape[1])]
    while found_optimum is False:
        random.shuffle(things_to_try_changing)
        found_optimum = True
        for index in things_to_try_changing:
            # dont_bother... is if FOM improves by decreasing a parameter,
            # there's no point in trying to increase it right afterwards
            dont_bother_trying_opposite_change = False
            if index[1] == 4:
                changes = [-.3*degree, .3*degree]
            else:
                changes = [-1*nm, 1*nm]
            for change in changes:
                if dont_bother_trying_opposite_change is True:
                    continue
                for _ in range(10):
                    # when you find a good change, keep applying it over and over
                    # until it stops working. This speeds up the optimization.
                    # But I limited to 10 repeats so that we find the local
                    # optimum near the initial conditions.
                    xyrra_list[index] += change
                    if not validate(mygrating, similar_to=similar_to, how_similar=how_similar):
                        xyrra_list[index] -= change
                        break
                    fom_new = mygrating.run_lua(subfolder=subfolder,
                                                target_wavelength=target_wavelength,
                                                numG=numG)
                    if fom_new < fom_now:
                        xyrra_list[index] -= change
                        break
                    else:
                        mygrating.standardize()
                        assert validate(mygrating, similar_to=similar_to, how_similar=how_similar)
                        print('#New record! ', fom_new)
                        print('mygrating='+repr(mygrating), flush=True)
                        print('', flush=True)
                        fom_now = fom_new
                        found_optimum = False
                        dont_bother_trying_opposite_change = True
    return mygrating

def optimize2(mygrating_start, target_wavelength, attempts=inf, similar_to=None,
              how_similar=None, subfolder=None, numG=50):
    """vary parameters randomly.
    
    Target_wavelength is not what the simulation is run at (which is determined
    in grating.lua), but rather for determining what angle the light is coming
    in at (which is the same for all wavelengths, in a collimator).
    
    Return optimized Grating object (input object is not altered)."""
    assert validate(mygrating_start, print_details=True,
                    similar_to=similar_to, how_similar=how_similar)
    mygrating = mygrating_start.copy()
    xyrra_list = mygrating.xyrra_list
    fom_now = mygrating.run_lua(subfolder=subfolder,
                                target_wavelength=target_wavelength, numG=numG)
    print('fom now...', fom_now, flush=True)
    max_change_array = np.empty_like(xyrra_list)
    max_change_array[:, 0:4] = 1*nm
    max_change_array[:,4] = 0.1*degree
    max_change_array /= xyrra_list.size
    attempts_so_far = 0
    while attempts_so_far < attempts:
        attempts_so_far += 1
        xyrra_list_change = max_change_array*(2*np.random.random(size=xyrra_list.shape)-1)
        for _ in range(10):
            # when you find a good change, keep applying it over and over
            # until it stops working. This speeds up the optimization.
            # But I limited to 10 repeats so that we find the local
            # optimum near the initial conditions.
            xyrra_list += xyrra_list_change
            if not validate(mygrating, similar_to=similar_to, how_similar=how_similar):
                xyrra_list -= xyrra_list_change
                break
            fom_new = mygrating.run_lua(subfolder=subfolder,
                                        target_wavelength=target_wavelength,
                                        numG=numG)
            if fom_new < fom_now:
                #print('lower fom', fom_new)
                xyrra_list -=xyrra_list_change
                break
            else:
                mygrating.standardize()
                assert validate(mygrating, similar_to=similar_to,
                                how_similar=how_similar, print_details=True)
                print('#New record! ', fom_new)
                print('mygrating='+repr(mygrating), flush=True)
                print('', flush=True)
                fom_now = fom_new
    return mygrating

def plot_eps():
    a = np.genfromtxt(os.path.join(cwd_for_S4(), 'grating_eps.txt'))
    xs = np.unique(a[:,0])
    ys = np.unique(a[:,1])

    eps_matrix = np.zeros(shape=(len(xs),len(ys)),dtype=complex)
    for row in a:
        ix = np.nonzero(xs == row[0])[0]
        iy = np.nonzero(ys == row[1])[0]
        eps_matrix[ix,iy] = row[3] + 1j * row[4]

    plt.figure()
    plt.imshow((eps_matrix.real.T)**0.5,  origin='lower', aspect='equal',  extent=(min(xs), max(xs), min(ys), max(ys)))
    plt.title('index')
    plt.colorbar()
    return eps_matrix

def stretch_pattern(xyrra_list_start, x_scale, y_scale):
    xyrra_list = xyrra_list_start.copy()
    xyrra_list[:,[0,2]] *= x_scale
    xyrra_list[:,[1,3]] *= y_scale
    return xyrra_list

def vary_angle(start_grating=None, end_angle=None, lens_type=None,
               target_wavelength=None, start_grating_collection=None,
               subfolder=None, numG=50):
    """Repeatedly increase the grating_period by little steps, then tweak the
    pattern to have high efficiency. But don't change it too much so that it
    stays almost-periodic. lens_type should be 'cyl' or 'round'.
    
    For cyl lens, we work our way towards the center of the lens, increasing
    the grating_period. For round_lens, it's the reverse, we decrease
    grating_period while increasing lateral_period. That makes it most easy to
    keep xyrra_list the same to start the next step without violating the
    validate() geometric constraints. (lateral_period typically changes
    more quickly than grating_period).
        
    Don't optimize the starting pattern, assume it's already done.
    
    Subfolder runs the simulations in a different directory, so that you can
    have multiple iPython interpreters running this function simultaneously"""
    
    #Exactly one of these parameters must be provided
    assert (start_grating_collection is None) != (start_grating is None
                                                 and target_wavelength is None)
    
    ### Initialize all_gratings - the grating collection we are making
    if start_grating_collection is not None:
        all_gratings = start_grating_collection
    else:
        if lens_type == 'cyl':
            all_gratings = GratingCollection(target_wavelength=target_wavelength,
                                             lateral_period=start_grating.lateral_period,
                                             grating_list=[start_grating],
                                             lens_type='cyl')
        else:
            assert lens_type == 'round'
            # for round lenses, the lateral_period parameter is redefined as
            # "lateral_period at a certain point / tan(angle_in_air) at that point"
            angle_in_air = start_grating.get_angle_in_air(target_wavelength=target_wavelength)
            lateral_period = start_grating.lateral_period / math.tan(angle_in_air)
            all_gratings = GratingCollection(target_wavelength=target_wavelength,
                                             lateral_period=lateral_period,
                                             grating_list=[start_grating],
                                             lens_type='round')
    
    if all_gratings.lens_type == 'cyl':
        # change the grating_period by this fraction each step
        change_each_step = 1.01
        # how much we allow the pattern to change each step
        similarity_each_step = 0.03
    else:
        # change the lateral_period by this fraction each step
        change_each_step = 1.01
        similarity_each_step = 0.03
    

    
    while True:
        print('grating collection so far:')
        print(repr(all_gratings))

        # grating_list is sorted from lens-outside to lens-center
        # so for cylindrical lens, we put new entries after the last entry
        # whereas for round lens, we put new entries before the first entry
        if all_gratings.lens_type == 'cyl':
            grating_prev = all_gratings.grating_list[-1]
        else:
            grating_prev = all_gratings.grating_list[0]
        
        if all_gratings.lens_type == 'cyl':
            grating_new_start = all_gratings.get_one(
                 grating_period=grating_prev.grating_period * change_each_step)
        else:
            grating_new_start = all_gratings.get_one(
                 lateral_period=grating_prev.lateral_period * change_each_step)
        angle_in_air = grating_new_start.get_angle_in_air(
                                          target_wavelength=all_gratings.target_wavelength)
        if angle_in_air < end_angle and all_gratings.lens_type == 'cyl':
            break
        if angle_in_air > end_angle and all_gratings.lens_type == 'round':
            break
        
        #xyrra_list_prev = grating_prev.xyrra_list.copy()
        print('Optimizing for angle_in_air = ', angle_in_air/degree, 'degree')
        
        grating_new_start = resize(grating_prev, grating_new_start)
                
        grating_new = optimize(grating_new_start,
                               target_wavelength=all_gratings.target_wavelength,
                               similar_to=grating_new_start.xyrra_list,
                               how_similar=similarity_each_step, subfolder=subfolder,
                               numG=numG)
        grating_new = optimize2(grating_new, attempts=200,
                                target_wavelength=all_gratings.target_wavelength,
                                similar_to=grating_new_start.xyrra_list,
                                how_similar=similarity_each_step, subfolder=subfolder,
                                numG=numG)

        all_gratings.add_one(grating_new)       
        
    return all_gratings

class GratingCollection:
    """A smoothly-varying collection of Grating objects for different angles.
    Use lens_type='cyl' for cylindrical lens, i.e. lens that focuses only in
    one of the two lateral directions. Use lens_type='round' for normal lenses.
    
    For cylindrical lens design: lateral_period is constant
    For round lens design: lateral_period is short for
    "lateral_period over tan(angle_in_air)", which is constant for the whole
    collection. This does NOT take into account the fact that there has to be
    an integer number of repetitions going around the circle, because that's a
    negligible correction except for tiny tiny lenses.
    
    target_wavelength is critical for round lenses because it determines the
    relation between lateral_period and grating_period. For cylindrical lenses,
    it doesn't really matter, but I'm requiring it anyway for convenience and
    simplicity.
    """
    def __init__(self, target_wavelength, lateral_period,
                 lens_type='cyl', grating_list=None):
        """initialize a GratingCollection. grating_list (optional) is a python
        list of Grating objects"""
        self.target_wavelength = target_wavelength
        self.lateral_period = lateral_period
        self.target_kvac = 2*pi / target_wavelength
        self.lens_type = lens_type
        
        assert self.lens_type in ('cyl', 'round')
        
        if grating_list is None:
            self.grating_list = []
        else:
            self.grating_list = grating_list
            self.sort_grating_list()
            self.check_consistency()

    def check_consistency(self):
        """just make sure that all the gratings have consistent cyl_height,
        refractive index, lateral_period, etc."""
        assert len({g.cyl_height for g in self.grating_list}) <= 1
        assert len({g.n_glass for g in self.grating_list}) <= 1
        assert len({g.n_tio2 for g in self.grating_list}) <= 1
        if self.lens_type == 'cyl':
            assert all(self.lateral_period == g.lateral_period
                           for g in self.grating_list)
        else:
            assert self.lens_type == 'round'
            wl = self.target_wavelength
            g = [g.lateral_period / math.tan(g.get_angle_in_air(target_wavelength=wl))
                   for g in self.grating_list]
            assert (max(g) - min(g)) < 1e-7 * max(g)
    
    def sort_grating_list(self):
            self.grating_list.sort(key=lambda x:x.grating_period)
    
    def add_one(self, new_grating):
        """add a grating to this collection.
        Input EITHER the angle_in_air OR the grating_period"""
        self.grating_list.append(new_grating)
        self.grating_list.sort(key=lambda x:x.grating_period)
        self.check_consistency()
    
    def get_one(self, angle_in_air=None, grating_period=None, lateral_period=None):
        """get a sim_setup object from this collection.
        Input EITHER the angle_in_air OR the grating_period,
        OR (round lens only) the local lateral_period (not lateral_period/tan angle)
        Can be outside the range where we have xyrra_lists, in which case we
        return a Grating object with blank xyrra_list."""
        # first, calculate grating_period (if it wasn't supplied)
        if grating_period is not None:
            assert angle_in_air is None and lateral_period is None
        elif angle_in_air is not None:
            assert lateral_period is None
            grating_period = self.target_wavelength / math.sin(angle_in_air)
        else:
            # a lateral_period was supplied
            assert self.lens_type == 'round'
            # self.lateral_period is short for lateral_period / tan(angle_in_air)
            angle_in_air = math.atan(lateral_period / self.lateral_period)
            grating_period = self.target_wavelength / math.sin(angle_in_air)
        
        # calculate lateral_period
        if self.lens_type == 'cyl':
            lateral_period = self.lateral_period
        else:
            angle_in_air = math.asin(self.target_wavelength / grating_period)
            lateral_period = self.lateral_period * math.tan(angle_in_air)
        
        # Find xyrra_list
        self.sort_grating_list()
        
        # if grating_period is within the bounds of the list, or 1% beyond,
        # then interpolate to find xyrra_list. (The 1% beyond is for
        # convenience, in case you have GratingCollections that don't *quite*
        # overlap.) Otherwise, return an empty xyrra_list
        if (grating_period < self.grating_list[0].grating_period * 0.99
              or grating_period > self.grating_list[-1].grating_period * 1.01):
            xyrra_list_in_nm_deg = None
        elif grating_period > self.grating_list[-1].grating_period:
            xyrra_list_in_nm_deg = self.grating_list[-1].xyrra_list_in_nm_deg
        elif grating_period < self.grating_list[0].grating_period:
            xyrra_list_in_nm_deg = self.grating_list[0].xyrra_list_in_nm_deg
        elif any(g.grating_period == grating_period for g in self.grating_list):
            # perfect match is already in the collection
            i = [g.grating_period for g in self.grating_list].index(grating_period)
            xyrra_list_in_nm_deg = self.grating_list[i].xyrra_list_in_nm_deg
        else:
            # interpolate. We're looking between entry i-1 and i.
            i = next(j for j,g in enumerate(self.grating_list)
                                 if g.grating_period > grating_period)
            grating_before = self.grating_list[i-1]
            period_before = grating_before.grating_period
            grating_after = self.grating_list[i]
            period_after = grating_after.grating_period
            assert (period_before < grating_period < period_after)
            after_weight = (grating_period - period_before) / (period_after - period_before)
            before_weight = (period_after - grating_period) / (period_after - period_before)
            assert (0 < before_weight < 1) and (0 < after_weight < 1)
            assert before_weight + after_weight == 1
            
            xyrra_list_in_nm_deg = (before_weight * grating_before.xyrra_list_in_nm_deg
                            + after_weight * grating_after.xyrra_list_in_nm_deg)
            
        return Grating(lateral_period=lateral_period,
                       cyl_height=self.grating_list[0].cyl_height,
                       grating_period=grating_period,
                       n_glass=self.grating_list[0].n_glass,
                       n_tio2=self.grating_list[0].n_tio2,
                       xyrra_list_in_nm_deg=xyrra_list_in_nm_deg)

    def get_innermost(self):
        """return the Grating object for the closest-to-lens-center part of
        this GratingCollection"""
        return self.grating_list[-1]
        
    def get_outermost(self):
        """return the SimSetup object for the closest-to-lens-center part of
        this GratingCollection"""
        return self.grating_list[0]
            
    def show_efficiencies(self, numG=100):
        """Calculate the efficiencies of each grating in the collection"""
        angles_efficiencies_list = []
        process_list = []
        # Calculate efficiencies in parallel by spawning all the processes
        # before reading any of them
        subfolder=random_subfolder_name()
        for i,g in enumerate(self.grating_list):
            process_list.append(g.run_lua_initiate(target_wavelength=self.target_wavelength,
                                                   subfolder=subfolder + str(i),
                                                   numG=numG))
        for i,g in enumerate(self.grating_list):
            eff = g.run_lua_getresult(process_list[i])
            remove_subfolder(subfolder + str(i))
            angle = g.get_angle_in_air(self.target_wavelength)
            print('angle_in_air:', angle/degree, 'deg, effic:', eff)
            angles_efficiencies_list.append((angle, eff))
        
        plt.figure()
        plt.plot([x[0] / degree for x in angles_efficiencies_list],
                 [x[1] for x in angles_efficiencies_list])
        return angles_efficiencies_list
    
    def __repr__(self):
        """this is important - after we spend hours finding a nice
        GratingCollection, let's say "mycollection", we can just type
        "mycollection" or "repr(mycollection)" into IPython and copy the
        resulting text. That's the code to re-create that grating collection
        immediately, without re-calculating it."""
        return ('GratingCollection('
                + 'target_wavelength=' + repr(self.target_wavelength/nm) + '*nm'
                + ', lateral_period=' + repr(self.lateral_period/nm) + '*nm'
                + ', lens_type=' + repr(self.lens_type)
                + ', grating_list= ' + repr(self.grating_list)
                + ')')
    
    def show_graphs(self, with_efficiencies=False,
                    anim_filename='grating_collection_anim.gif', numG=100):
        max_grating_period = max(g.grating_period for g in self.grating_list)
        max_lateral_period = max(g.lateral_period for g in self.grating_list)
        filename_list = []
        
        # Calculate efficiencies in parallel by spawning all the processes
        # before reading any of them
        if with_efficiencies is True:
            subfolder=random_subfolder_name()
            process_list = []
            for i,g in enumerate(self.grating_list[::-1]):
                process_list.append(g.run_lua_initiate(target_wavelength=self.target_wavelength,
                                                       subfolder=subfolder+str(i), numG=numG))
        for i,g in enumerate(self.grating_list[::-1]):
            g.show_config()
            plt.xlim(-max_grating_period/nm, max_grating_period/nm)
            plt.ylim(-max_lateral_period/nm, max_lateral_period/nm)
            filename_list.append('grating_collection' + str(i) + '.png')
            angle = g.get_angle_in_air(self.target_wavelength) / degree
            if with_efficiencies is True:
                eff = g.run_lua_getresult(process_list[i])
                remove_subfolder(subfolder+str(i))
                plt.title('From angle: {:.1f}°, effic={:.2%}'.format(angle, eff))
            else:
                plt.title('From angle: {:.1f}°'.format(angle))
            plt.savefig(filename_list[-1])
            plt.close()
        seconds_per_frame = 0.3
        frame_delay = str(int(seconds_per_frame * 100))
        # Use the "convert" command (part of ImageMagick) to build the animation
        command_list = ['convert', '-delay', frame_delay, '-loop', '0'] + filename_list + [anim_filename]
        if True:
            #### WINDOWS ####
            subprocess.call(command_list, shell=True)
        else:
            #### LINUX ####
            subprocess.call(command_list)
        if True:
            for filename in filename_list:
                os.remove(filename)
                
    def export_to_lumerical(self, angle_in_air=None, grating_period=None, lateral_period=None):
        """Set up for running grating_lumerical.lsf for either one grating in
        this grating collection, or all of them.
        If no argument is supplied, run all of them. Or, input EITHER the
        angle_in_air OR the grating_period, OR (round lens only) the local
        lateral_period (not lateral_period/tan angle) to run just one"""
        if any(x is not None for x in (angle_in_air, grating_period, lateral_period)):
            mygrating = self.get_one(angle_in_air=angle_in_air,
                                     grating_period=grating_period,
                                     lateral_period=lateral_period)
            mygrating.run_lumerical()
            return
        i = 0
        for g in self.grating_list:
            i += 1
            angle_in_air = g.get_angle_in_air(self.target_wavelength)
            g.write(angle_in_air=angle_in_air, index=0, replicas=True)
        # delete the one after that so that lumerical knows to quit
        if os.path.isfile(xyrra_filename(index=i+1)):
            os.remove(xyrra_filename(index=i+1))
        if os.path.isfile(setup_filename(index=i+1)):
            os.remove(setup_filename(index=i+1))
    
    def characterize(self, wavelength, numG=100, u_steps=5, just_normal=False):
        """run g.characterize() for each g in the GratingCollection. This stores
        far-field amplitude information in the g object"""
        if just_normal:
            ux_min=ux_max=uy_min=uy_max=0.001
            u_steps=1
        else:
            target_ux_min = self.get_innermost().get_angle_in_air(self.target_wavelength)
            target_ux_max = self.get_outermost().get_angle_in_air(self.target_wavelength)
            ux_min = max(-0.99, target_ux_min - 0.25)
            ux_max = min(0.99, target_ux_max + 0.25)
            uy_min= -0.2
            uy_max=0.2
        subfolder=random_subfolder_name()
        process_list = []
        for i,g in enumerate(self.grating_list):
            process = g.run_lua_initiate(subfolder=subfolder + str(i),
                                         ux_min=ux_min, ux_max=ux_max,
                                         uy_min=uy_min, uy_max=uy_max,
                                         u_steps=u_steps, wavelength=wavelength,
                                         numG=numG)
            process_list.append(process)
        for i,g in enumerate(self.grating_list):
            g.characterize(process = process_list[i], just_normal=just_normal)
            remove_subfolder(subfolder+str(i))
    
    def build_interpolators(self):
        """after running self.characterize(), this creates interpolator objects
        
        If f = self.interpolators[(580,(1,2),'x','amprx')]
        then f([[0.3,0.1,1000*nm], [0.4,0.2,1010*nm]]) is [X,Y] where X is
        the amprx for light coming at (ux,uy=(0.3,0.1)) onto a 1000nm grating
        at 580nm, x-polarization (see S4conventions.py), (1,2) diffraction
        order, and Y is likewise for the next entry"""
        self.interpolators = {}
        ux_list = sorted({e['ux'] for g in self.grating_list for e in g.data})
        uy_list = sorted({e['uy'] for g in self.grating_list for e in g.data})
        grating_period_list = sorted({g.grating_period for g in self.grating_list})
        lookup_table = {(round(e['wavelength_in_nm']),e['ox'],e['oy'],e['x_or_y'],e['ux'],e['uy'],g.grating_period): e
                           for g in self.grating_list for e in g.data}
        
        for wavelength_in_nm in {round(e['wavelength_in_nm']) for g in self.grating_list for e in g.data}:
            for (ox,oy) in {(e['ox'], e['oy']) for g in self.grating_list for e in g.data}:
                for x_or_y in ('x','y'):
                    #for amp in ('ampfy','ampfx','ampry','amprx'):
                    for amp in ('ampfy','ampfx'):
                        # want grid_data[i,j,k] = amp(ux_list[i], uy_list[j], grating_period_list[k])
                        grid_data = np.zeros((len(ux_list), len(uy_list), len(grating_period_list)),
                                             dtype=complex)
                        for i,ux in enumerate(ux_list):
                            for j,uy in enumerate(uy_list):
                                for k, grating_period in enumerate(grating_period_list):
                                    entry=lookup_table.get((wavelength_in_nm,ox,oy,x_or_y,ux,uy,grating_period))
                                    if entry is not None:
                                        grid_data[i,j,k] = entry[amp]
                        # vary_angle stops a bit short of the target angle
                        # (TODO - fix that! Then delete this little workaround.)
                        # ...therefore we allow grating periods slightly outside
                        # the range by using the innermost or outermost grating.
                        grid_data_extended = np.zeros((len(ux_list),len(uy_list),2+len(grating_period_list)),
                                                      dtype=complex)
                        grid_data_extended[:, :, 1:-1] = grid_data
                        grid_data_extended[:, :, 0] = grid_data[:, :, 0]
                        grid_data_extended[:, :, -1] = grid_data[:, :, -1]
                        grating_period_list_extended = np.hstack((0.99 * min(grating_period_list),
                                                                  grating_period_list,
                                                                  1.01 * max(grating_period_list)))
                        interp_function = RegularGridInterpolator((ux_list,uy_list,grating_period_list_extended),
                                                                  grid_data_extended)
                        self.interpolators[(wavelength_in_nm, (ox,oy), x_or_y, amp)] = interp_function
        self.interpolator_bounds = (min(ux_list), max(ux_list), min(uy_list),
                                    max(uy_list), min(grating_period_list_extended),
                                    max(grating_period_list_extended))

#def read_lumerical_batch_analysis():
#    i = 0
#    while os.path.isfile(xyrra_filename(index=i)):
#        with open(xyrra_filename(index=i), 'r') as f:
#            pass
#        # TO DO
#        i += 1
            

def plot_round_lateral_period(f, reps_around_circumference, target_wavelength=580*nm):
    """quick investigation of how lateral_period and grating_period vary"""
    distances_from_center = np.linspace(100*nm, f*5, num=1000)
    angles = np.array([math.atan(d/f) for d in distances_from_center])
    lateral_periods = np.array([2*pi*d / reps_around_circumference for d in distances_from_center])
    grating_periods = np.array([target_wavelength / math.sin(angle) for angle in angles])

    plt.figure()
    plt.plot(lateral_periods / nm, grating_periods / nm)
    plt.xlabel('lateral period (nm)')
    plt.ylabel('grating period (nm)')
    plt.xlim(0,800)
    plt.ylim(0,2000)
    plt.grid()

    plt.figure()
    plt.plot(lateral_periods / nm, angles / degree)
    plt.xlabel('lateral period (nm)')
    plt.ylabel('angle (degree)')
    plt.xlim(0,800)
    plt.grid()

    plt.figure()
    plt.plot(angles[0:-1] / degree, [(lateral_periods[i+1] / lateral_periods[i] - 1) / (grating_periods[i] / grating_periods[i+1] - 1) for i in range(len(angles)-1)])
    plt.plot(angles[0:-1] / degree, [1 for i in range(len(angles)-1)])
    plt.xlabel('angle (degree)')
    plt.ylabel('(How fast lateral_period changes)/(How fast grating_period changes)')
    plt.grid()
#plot_round_lateral_period(150*um, 3427)
 

def n_glass(wavelength_in_nm):
    """for better data see refractive_index.py .. but this is what I'm using in
    lumerical and lua, and I want to be consistent sometimes"""
    data = {450: 1.466,
            500: 1.462,
            525: 1.461,
            550: 1.46,
            575: 1.459,
            580: 1.459,
            600: 1.458,
            625: 1.457,
            650: 1.457}
    if wavelength_in_nm not in data:
        raise ValueError('bad wavelength'+repr(wavelength_in_nm))
    return data[wavelength_in_nm]
