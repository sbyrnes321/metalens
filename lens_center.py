# coding=UTF-8
"""
(C) 2015 Steven Byrnes

This script defines a HexGridSet object which is used to store a set of
hexagonal grid cylinder patterns. Each unit cell in the center part of
the lens is chosen from this set to create the correct phase.

"""


import math, cmath
import numpy as np
import matplotlib.pyplot as plt
import grating
from numericalunits import nm
pi = math.pi
inf = float('inf')

degree = pi / 180

from scipy.interpolate import RegularGridInterpolator


class HexGridSet:
    """A HexGridSet is a set of geometries for the center of the lens"""
    def __init__(self, sep, cyl_height, n_glass=0, n_tio2=0, grating_list=None,
                 x_amp_list=None, num_entries=20):
        """sep is the cylinder nearest-neighbor center-to-center separation.
        grating_list is all the relevant configurations in the form of Grating
        objects. I know we're not really using them as gratings per se, but the
        Grating object still works."""
        self.sep = sep
        # nnn_sep is next-nearest-neighbor center-to-center separation
        self.nnn_sep = self.sep * 3**0.5
        self.cyl_height = cyl_height
        self.n_glass = n_glass
        self.n_tio2 = n_tio2
        if grating_list is not None:
            self.grating_list = grating_list
        else:
            self.grating_list = []
            for diam in np.linspace(100.01*nm, self.sep-100.01*nm, num=num_entries):
                # To do someday: Allow radius 0, i.e. leave out a cylinder from some cell
                r=diam/2
                xyrra_list_in_nm_deg = [[0, 0, r/nm, r/nm, 0],
                                        [self.nnn_sep/2/nm, self.sep/2/nm, r/nm, r/nm, 0]]
                g = grating.Grating(grating_period=self.nnn_sep,
                                    lateral_period=self.sep,
                                    n_glass=self.n_glass,
                                    n_tio2=self.n_tio2,
                                    cyl_height=self.cyl_height,
                                    xyrra_list_in_nm_deg=np.array(xyrra_list_in_nm_deg))
                assert grating.validate(g)
                self.grating_list.append(g)
        if x_amp_list is not None:
            self.x_amp_list = np.array(x_amp_list)
    
    def __repr__(self):
        """this is important - after we spend hours finding a nice
        GratingCollection, let's say "mycollection", we can just type
        "mycollection" or "repr(mycollection)" into IPython and copy the
        resulting text. That's the code to re-create that grating collection
        immediately, without re-calculating it."""
        if hasattr(self, 'x_amp_list'):
            x_amp_list_str = (np.array2string(self.x_amp_list, separator=',')
                                         .replace(' ', '').replace('\n',''))
        else:
            x_amp_list_str = 'None'

        return ('HexGridSet('
                + 'sep=' + repr(self.sep/nm) + '*nm'
                + ', cyl_height=' + repr(self.cyl_height/nm) + '*nm'
                + ', n_glass=' + repr(self.n_glass)
                + ', n_tio2=' + repr(self.n_tio2)
                + ', grating_list= ' + repr(self.grating_list)
                + ', x_amp_list=' + x_amp_list_str
                + ')')
                
    def characterize(self, wavelength=580*nm, numG=100, just_normal=True,
                     shortcut=False, u_steps=3):
        """run g.characterize() for each g in the HexGridSet. This stores
        far-field amplitude information in the g object. Then also compile
        a list of complex p amplitudes. If just_normal is False, use a range
        of input angles, not just normal incidence. If shortcut is True, use
        ux>=0,uy>=0, and fill in the negative values by symmetry"""
        subfolder=grating.random_subfolder_name()
        process_list = []
        if just_normal is True:
            u_args={'ux_min':0.001, 'ux_max':0.001, 'uy_min':0.001, 'uy_max':0.001,
                    'u_steps':1}
        elif shortcut is False:
            u_args={'ux_min':-0.499, 'ux_max':0.501, 'uy_min':-0.499, 'uy_max':0.501,
                    'u_steps':2*u_steps-1}
        else:
            u_args={'ux_min':0.001, 'ux_max':0.501, 'uy_min':0.001, 'uy_max':0.501,
                    'u_steps':u_steps}
        for i,g in enumerate(self.grating_list):
            process = g.run_lua_initiate(subfolder=subfolder + str(i),
                                         wavelength=wavelength, numG=numG,
                                         **u_args)
            process_list.append(process)
        for i,g in enumerate(self.grating_list):
            g.characterize(process = process_list[i], convert_to_xy=True,
                           just_normal=just_normal)
            grating.remove_subfolder(subfolder+str(i))
        
        if (just_normal is False) and (shortcut is True):
            assert False # until I double-check that this code is correct, esp. other grating orders
            for data in [g.data for g in self.grating_list]:
                for e in range(len(data)):
                    ux,uy,ox,oy,x_or_y = e['ux'],e['uy'],e['ox'],e['oy'],e['x_or_y']
                    ampfy,ampfx,ampry,amprx = e['ampfy'],e['ampfx'],e['ampry'],e['amprx']
                    assert ux > 0 and uy >= 0
                    if x_or_y == 'x':
                        data.append({'ux': -ux, 'uy': uy,
                                     'ox':-ox, 'oy':oy,
                                     'x_or_y': x_or_y, 'wavelength_in_nm':e['wavelength_in_nm'],
                                     'ampfx':ampfx, 'ampfy':ampfy,
                                     'amprx':amprx, 'ampry':ampry})
                        if uy > 0:
                            data.append({'ux': ux, 'uy': -uy,
                                         'ox':ox, 'oy':-oy,
                                         'x_or_y': x_or_y, 'wavelength_in_nm':e['wavelength_in_nm'],
                                         'ampfx':ampfx, 'ampfy':-ampfy,
                                         'amprx':amprx, 'ampry':-ampry})
                            data.append({'ux': -ux, 'uy':-uy,
                                         'ox':-ox, 'oy':-oy,
                                         'x_or_y': x_or_y, 'wavelength_in_nm':e['wavelength_in_nm'],
                                         'ampfx':ampfx, 'ampfy':-ampfy,
                                         'amprx':amprx, 'ampry':-ampry})
                    else: #x_or_y == 'y'
                        data.append({'ux': -ux, 'uy': uy,
                                     'ox':-ox, 'oy':oy,
                                     'x_or_y': x_or_y, 'wavelength_in_nm':e['wavelength_in_nm'],
                                     'ampfx':-ampfx, 'ampfy':ampfy,
                                     'amprx':-amprx, 'ampry':ampry})
                        if uy > 0:
                            data.append({'ux': ux, 'uy': -uy,
                                         'ox':ox, 'oy':-oy,
                                         'x_or_y': x_or_y, 'wavelength_in_nm':e['wavelength_in_nm'],
                                         'ampfx':ampfx, 'ampfy':ampfy,
                                         'amprx':amprx, 'ampry':ampry})
                            data.append({'ux': -ux, 'uy':-uy,
                                         'ox':-ox, 'oy':-oy,
                                         'x_or_y': x_or_y, 'wavelength_in_nm':e['wavelength_in_nm'],
                                         'ampfx':-ampfx, 'ampfy':ampfy,
                                         'amprx':-amprx, 'ampry':ampry})
        
        x_amp_list = []
        for g in self.grating_list:
            a = [e for e in g.data if e['x_or_y'] == 'x'
                   and e['ox']==e['oy']==0 and e['ux']==e['uy']==0.001]
            assert len(a) == 1
            x_amp_list.append(a[0]['ampfx'])
        self.x_amp_list = np.array(x_amp_list)
            
    def show_properties(self):
        d_list = np.array([2 * g.xyrra_list[0,2] for g in self.grating_list])
        x_amp_list = self.x_amp_list
        if self.grating_list[0].n_glass == 0:
            n_glass = grating.n_glass(self.grating_list[0].data[0]['wavelength_in_nm'])
        else:
            n_glass = self.grating_list[0].n_glass
        fig, ax1 = plt.subplots()
        Ts = abs(x_amp_list)**2 / n_glass
        phases = np.unwrap(np.angle(x_amp_list))
        ax1.plot(d_list/nm, Ts, 'b')
        ax1.set_ylim(0,1)
        plt.title('T and phase at normal incidence')
        plt.xlabel('diameter')
        ax2 = ax1.twinx()
        ax2.plot(d_list/nm, phases, 'g')
   
    def pick_from_phase(self, target_phase):
        """given a target phase, find the best Grating object for
        achieving this phase. Return its index in self.grating_list"""
        if not hasattr(self, 'x_amp_list'):
            raise ValueError('Need to run characterize() first')
        # I know that I have the right phase convention here (relative to the
        # periphery of the lens) because I ran build_nearfield() and plotted
        # the complex phase of the nearfield Ex and it was the same in the
        # center as the periphery.
        fom_list = (self.x_amp_list * np.exp(-1j * target_phase)).imag
        best_index = np.argmax(fom_list)
        return best_index
        
    def build_interpolators(self):        
        """after running self.characterize(), this creates interpolator objects
        
        If f = self.interpolators[(580,(1,2),'x','ampfy')]
        then f([[0.3,0.1,6], [0.4,0.2,3]]) is [X,Y] where X is
        the ampfy for light coming at (ux,uy=(0.3,0.1)) onto self.grating_list[6]
        at 580nm, x-polarization, (1,2) diffraction order, and Y is likewise
        for the next entry"""
        if not hasattr(self, 'x_amp_list'):
            raise ValueError('Need to run characterize() first')
        
        self.interpolators = {}
        ux_list = sorted({e['ux'] for g in self.grating_list for e in g.data})
        uy_list = sorted({e['uy'] for g in self.grating_list for e in g.data})
        grating_index_list = np.arange(len(self.grating_list))
        for wavelength_in_nm in {round(e['wavelength_in_nm']) for g in self.grating_list for e in g.data}:
            for (ox,oy) in {(e['ox'], e['oy']) for g in self.grating_list for e in g.data}:
                for x_or_y in ('x','y'):
                    for amp in ('ampfy','ampfx','ampry','amprx'):
                        # want grid_data[i,j,k] = amp(ux_list[i], uy_list[j], grating_period_list[k])
                        grid_data = np.zeros((len(ux_list), len(uy_list), len(grating_index_list)),
                                             dtype=complex)
                        for i,ux in enumerate(ux_list):
                            for j,uy in enumerate(uy_list):
                                for k, grating_index in enumerate(grating_index_list):
                                    grating_now = self.grating_list[grating_index]
                                    matches = [e for e in grating_now.data
                                                   if e['wavelength_in_nm'] == wavelength_in_nm
                                                       and (e['ox'], e['oy']) == (ox, oy)
                                                       and e['x_or_y'] == x_or_y
                                                       and (e['ux'], e['uy']) == (ux,uy)]
                                    assert len(matches) <= 1
                                    if len(matches) == 1:
                                        grid_data[i,j,k] = matches[0][amp]
                        interp_function = RegularGridInterpolator((ux_list,uy_list,grating_index_list), grid_data)
                        self.interpolators[(wavelength_in_nm, (ox,oy), x_or_y, amp)] = interp_function
        self.interpolator_bounds = (min(ux_list), max(ux_list), min(uy_list),
                                    max(uy_list), min(grating_index_list),
                                    max(grating_index_list))

