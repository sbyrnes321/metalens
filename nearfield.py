# -*- coding: utf-8 -*-
"""
(C) 2015 Steven Byrnes

Calculate the near-field of a grating-based metasurface lens
"""

from __future__ import division, print_function
import math
from math import pi
degree = pi / 180
import numpy as np
#http://pythonhosted.org/numericalunits/
import numericalunits as nu
from numericalunits import um, nm
# http://pythonhosted.org/dxfwrite/
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
inf = float('inf')


import design_collimator
import grating
from numpy.fft import fft2, fftshift

#import loop_post_analysis
#import grating


def good_fft_number(goal):
    """pick a number >= goal that has only factors of 2,3,5. FFT will be much
    faster if I use such a number"""
    assert goal < 1e5
    choices = [2**a * 3**b * 5**c for a in range(17) for b in range(11)
                                  for c in range(8)]
    return min(x for x in choices if x >= goal)

"""
       * lens_periphery_summary, a dictionary:
          {'r_center_list': array([...]),
              -- the radius at the center of each subsequent grating

           'r_min_list': array([...]),
             -- the radius at the inner boundary of this grating

           'grating_period_list': array([...]),
              -- the period of the corresponding grating. Note that
              r_center_list[i] + 0.5 * grating_period_list[i]
                 + 0.5 * grating_period_list[i+1] == r_center_list[i+1]

           'gratingcollection_list': [...]
             -- GratingCollection objects from inside to out
                (list will look like [gc0, gc1, gc2, ...])

           'gratingcollection_index_here_list': [...]
             -- For each ring of gratings, what is the applicable
                gratingcollection object (indexed from the list above)?
                (list will look like [0,0,0,...,1,1,1,...2,2,2...])

           'num_around_circle_list': [...]
             -- how many copies are there around 2pi, for each entry of the above
                 (list will look like [n1, n1, n1, ... , n2, n2, n2, ...])
           }
                 """

def build_nearfield(source_x, source_y, source_z, source_pol, wavelength,
                    lens_periphery_summary, lens_center_summary, hexgridset,
                    x_pts=None, y_pts=None, dipole_moment=1e-30 * nu.C * nu.m):
    """To get an isotropic source, we can take an incoherent sum of an x-, y-,
    and z-polarized dipole source. Then to get a Lambertian source we scale
    the field by cos(theta). So source_pol should be 'x' or 'y' or 'z'. I don't
    think there's any way to do it with just two incoherent runs ... you can't
    pick two orthogonal polarizations smoothly everywhere.

    Note: I am not worrying about how much RAM this function uses. If you run
    out of RAM just use build_nearfield_big() below instead.

    dipole_moment is arbitrary, it turns into a scale factor for E and H. But
    use real (numericalunits) units, and the result will also be in real units
    
    If source_z = -inf, do a normally-incident plane wave. Use dipole_moment
    as the magnitude of the electric field.
    """
    assert source_z < 0
    assert source_pol in ('x','y','z')
    wavelength_in_nm = int(round(wavelength/nm))
    r_min_list = lens_periphery_summary['r_min_list']
    r_max_list = lens_periphery_summary['r_max_list']
    r_center_list = lens_periphery_summary['r_center_list']
    gratingcollection_index_here_list = lens_periphery_summary['gratingcollection_index_here_list']
    num_around_circle_list = lens_periphery_summary['num_around_circle_list']
    grating_period_list = lens_periphery_summary['grating_period_list']
    gratingcollection_list = lens_periphery_summary['gratingcollection_list']
    lens_max_r = r_max_list[-1]
    if x_pts is None:
        num_x = good_fft_number(2 * lens_max_r / (wavelength / 2.2))
        x_pts = np.linspace(-lens_max_r, lens_max_r, num=num_x)
    else:
        num_x = len(x_pts)
    if y_pts is None:
        num_y = good_fft_number(2 * lens_max_r / (wavelength / 2.2))
        y_pts = np.linspace(-lens_max_r, lens_max_r, num=num_y)
    else:
        num_y = len(y_pts)

    for l in [x_pts,y_pts]:
        diffs = [l[i+1] - l[i] for i in range(len(l)-1)]
        assert 0 < diffs[0] < wavelength/2
        assert max(diffs) - min(diffs) <= 1e-9 * max(abs(d) for d in diffs)

    n_glass = gratingcollection_list[0].grating_list[0].n_glass
    if n_glass == 0:
        n_glass = grating.n_glass(wavelength_in_nm)
    k_glass = 2*pi*n_glass/wavelength
    kvac = 2*pi/wavelength

    x_meshgrid,y_meshgrid = np.meshgrid(x_pts, y_pts, indexing='ij')
    lens_r = (x_meshgrid**2 + y_meshgrid**2)**0.5
    lens_phi = np.arctan2(y_meshgrid,x_meshgrid)

    # which_ring is the index for what ring of gratings each thing is, or -1
    # means N/A (in the center or outside the lens). in_center is specifically
    # points in the center

    ring_boundary_list = np.hstack((r_min_list, lens_max_r))
    which_ring = np.searchsorted(ring_boundary_list, lens_r) - 1
    in_center = (which_ring == -1)
    which_ring[which_ring == len(r_min_list)] = -1
    
    if which_ring.max() == -1 and in_center.max() == 0:
        # no points in the lens, shortcut to the end
        Ex = Ey = Hx = Hy = np.zeros_like(which_ring, dtype=complex)
        power_passing_through_lens = 0
        return Ex, Ey, Hx, Hy, x_pts, y_pts, power_passing_through_lens, n_glass

#    #### test the which_ring code
#    for i,x in enumerate(x_pts):
#        for j,y in enumerate(y_pts):
#            n = which_ring[i,j]
#            r = (x**2 + y**2)**0.5
#            if n == -1:
#                assert r <= r_min_list[0] or r >= lens_max_r
#            else:
#                assert r_min_list[n] <= r <= r_max_list[n]

    # which_gratingcollection is -1 if the point is not in the periphery, or i
    # if it falls in the domain of gratingcollection_list[i]
    which_gratingcollection = gratingcollection_index_here_list[which_ring]
    which_gratingcollection[which_ring == -1] = -1


    # grating_period is the length of this grating unit cell in the radial
    # direction
    grating_period = grating_period_list[which_ring]
    # Note: The command a = blah[which_ring] will set a[i,j] = blah[-1] when
    # i,j is outside the lens periphery. I will not be using the data at these
    # points for any output results so it generally doesn't matter what they're
    # set to. (Except which_ring and which_gratingcollection; these are used to
    # see what's in the lens periphery.)

    # angle_per_grating the angle that you need to rotate about the lens
    # center to get to the next copy of this grating
    angle_per_grating = 2*pi/num_around_circle_list[which_ring]
    r_center = r_center_list[which_ring]
    # lateral_period is the length of this grating unit cell in the azimuthal
    # direction
    lateral_period = r_center * angle_per_grating
    # grating_rotation is the CCW rotation of this grating relative to the x axis
    grating_rotation = (lens_phi / angle_per_grating).round() * angle_per_grating
    gratingcenter_x = r_center * np.cos(grating_rotation)
    gratingcenter_y = r_center * np.sin(grating_rotation)
    dx = x_meshgrid - source_x
    dy = y_meshgrid - source_y
    dz = 0 - source_z
    distance = (dx**2 + dy**2 + dz**2)**0.5
    # (ux,uy,uz) is the unit vector that the incoming light is traveling.
    if source_z == -inf:
        ux = np.zeros_like(x_meshgrid)
        uy = np.zeros_like(x_meshgrid)
        uz = np.ones_like(x_meshgrid)
    else:
        ux = dx / distance
        uy = dy / distance
        uz = dz / distance

    # xp,yp,z (short for xprime, yprime,z) coordinates are a coordinate system
    # where (xp,yp)=(0,0) is the center of the grating that this point is on,
    # increasing xp moves away from the lens center, and increasing yp move
    # CCW around the lens center.

    # (uxp,uyp,uz) is the primed coordinates version of (ux,uy,uz), i.e. the
    # unit vector that the incoming light is travelgin
    # Checking signs: If (ux,uy)=(1,0) (light heading rightward)
    # and grating_rotation = +10degrees (first quadrant) then uyp is negative
    uxp = ux * np.cos(grating_rotation) + uy * np.sin(grating_rotation)
    uyp = -ux * np.sin(grating_rotation) + uy * np.cos(grating_rotation)
    # Checking signs: If (x,y) ~ (cos(grating_rotation),sin(grating_rotation))
    # then we expect yp = 0
    # The following two options are exactly identical (I checked)
    xp = x_meshgrid * np.cos(grating_rotation) + y_meshgrid * np.sin(grating_rotation) - r_center
    yp = -x_meshgrid * np.sin(grating_rotation) + y_meshgrid * np.cos(grating_rotation)
#    xp = ((x_meshgrid-gratingcenter_x) * np.cos(grating_rotation)
#               + (y_meshgrid-gratingcenter_y) * np.sin(grating_rotation))
#    yp = (-(x_meshgrid-gratingcenter_x) * np.sin(grating_rotation)
#               + (y_meshgrid-gratingcenter_y) * np.cos(grating_rotation))


    # dipole field: We are calculating the actual field in real units, except
    # for the e^ikr phase factor
    # lambert cosine law: intensity goes as cos(angle_from_normal), so I should
    # scale fields by the square-root of that, i.e. uz**0.5
    # Jackson (9.19): H = ck^2/4pi * (n x p) * e^ikr/r   ;  E = Z0 H x n
    H_coef = nu.c0 * (2*pi / wavelength)**2 * dipole_moment / (4*pi)

    pol_vector = {'x':[1,0,0], 'y':[0,1,0], 'z':[0,0,1]}[source_pol]
    if source_z > -inf:
        dipole_field_Hx = (uy * pol_vector[2] - uz * pol_vector[1]) * H_coef * uz**0.5 / distance
        dipole_field_Hy = (uz * pol_vector[0] - ux * pol_vector[2]) * H_coef * uz**0.5 / distance
        dipole_field_Hz = (ux * pol_vector[1] - uy * pol_vector[0]) * H_coef * uz**0.5 / distance
        # then E is proportional to H cross rhat
        dipole_field_Ex = (dipole_field_Hy * uz - dipole_field_Hz * uy) * nu.Z0
        dipole_field_Ey = (dipole_field_Hz * ux - dipole_field_Hx * uz) * nu.Z0
    else:
        assert source_pol != 'z'
        dipole_field_Ex = pol_vector[0] * dipole_moment * np.ones((num_x,num_y))
        dipole_field_Ey = pol_vector[1] * dipole_moment * np.ones((num_x,num_y))
        dipole_field_Hx = -pol_vector[1] * dipole_moment / nu.Z0 * np.ones((num_x,num_y))
        dipole_field_Hy = pol_vector[0] * dipole_moment / nu.Z0 * np.ones((num_x,num_y))
        
    # switch to primed coordinates
    dipole_field_Hxp = (dipole_field_Hx * np.cos(grating_rotation)
                       + dipole_field_Hy * np.sin(grating_rotation))
    dipole_field_Hyp = (-dipole_field_Hx * np.sin(grating_rotation)
                       + dipole_field_Hy * np.cos(grating_rotation))
    
    
    # Our grating.characterize() data has results of a simulation with unit
    # amplitude x-polarized incoming light, and a simulation with y-polarized
    # (see S4conventions.py for definitions). We want to write our incoming
    # dipole_field as
    # x_weight * (x simulation incoming field) + y_weight * (y incoming field)
    # and then we know that the output is similarly a sum of the two simulation
    # outputs.
    # Note that this is the weight for H. H_weight * Z0 == E_weight, because
    # Z0=1 in S4 units (Z0 is impedance of free space)
    H_xp_weight = dipole_field_Hyp
    H_yp_weight = dipole_field_Hxp

    # electric and magnetic fields in primed coordinates at each point
    # There is a z component too but it doesn't enter far-field calculation
    Exp = np.zeros((num_x,num_y), dtype=complex)
    Eyp = np.zeros((num_x,num_y), dtype=complex)
    Hxp = np.zeros((num_x,num_y), dtype=complex)
    Hyp = np.zeros((num_x,num_y), dtype=complex)

    # This does the interpolation. Note that we are evaluating each
    # interpolating function only once, in a vectorized way, otherwise it is
    # super slow.
    # make cache to store kxp, kyp, kxp**2+kyp**2 for each grating order
    kxp_cache = {}
    kyp_cache = {}
    kxp2_plus_kyp2_cache = {}
    for gc_index, gc in enumerate(gratingcollection_list):
        all_orders = {(e['ox'],e['oy']) for g in gc.grating_list for e in g.data}
        for ox,oy in all_orders:
            # uxp,uyp is propagation direction in air. So use kvac here, not kglass
            if (ox,oy) not in kxp_cache:
                kxp = kvac * uxp + ox * 2*pi/grating_period
                kyp = kvac * uyp + oy * 2*pi/lateral_period
                kxp2_plus_kyp2 = kxp**2 + kyp**2
                kxp_cache[(ox,oy)] = kxp
                kyp_cache[(ox,oy)] = kyp
                kxp2_plus_kyp2_cache[(ox,oy)] = kxp2_plus_kyp2
            else:
                kxp = kxp_cache[(ox,oy)]
                kyp = kyp_cache[(ox,oy)]
                kxp2_plus_kyp2 = kxp2_plus_kyp2_cache[(ox,oy)]

            entries = np.logical_and((kxp2_plus_kyp2 <= kvac**2),
                                    (which_gratingcollection==gc_index))
            if entries.sum() == 0:
                continue
            print('diffraction order', (ox,oy), 'of gc', gc_index,
                  '; applies at', entries.sum(), 'points', flush=True)
            kxp = kxp[entries]
            kyp = kyp[entries]
            kzp = (k_glass**2-kxp**2-kyp**2)**0.5
            # S4 references phases to the pillar-glass interface, center of the
            # grating unit cell. Because we want the field at a different point,
            # we need a phase propagation factor
            phase_from_offcenter = np.exp(1j * (kxp * xp[entries] + kyp * yp[entries]))

            points_to_interpolate_at = np.vstack((uxp[entries], uyp[entries], grating_period[entries])).T
            if uxp[entries].min() < gc.interpolator_bounds[0]:
                raise ValueError('need to calculate at smaller ux!', uxp[entries].min(), gc.interpolator_bounds[0])
            if uxp[entries].max() > gc.interpolator_bounds[1]:
                raise ValueError('need to calculate at bigger ux!', uxp[entries].max(), gc.interpolator_bounds[1])
            if uyp[entries].min() < gc.interpolator_bounds[2]:
                raise ValueError('need to calculate at smaller uy!', uyp[entries].min(), gc.interpolator_bounds[2])
            if uyp[entries].max() > gc.interpolator_bounds[3]:
                raise ValueError('need to calculate at bigger uy!', uyp[entries].max(), gc.interpolator_bounds[3])
            if grating_period[entries].min() < gc.interpolator_bounds[4]:
                raise ValueError('need to calculate at smaller grating_period!', grating_period[entries].min()/nm, gc.interpolator_bounds[4]/nm)
            if grating_period[entries].max() > gc.interpolator_bounds[5]:
                raise ValueError('need to calculate at bigger grating_period!', grating_period[entries].max()/nm, gc.interpolator_bounds[5]/nm)
            for x_or_y in ('x', 'y'):
                H_weight = H_xp_weight[entries] if x_or_y == 'x' else H_yp_weight[entries]
                E_weight = H_weight * nu.Z0
                for which_amp in ('ampfy', 'ampfx'):
                    f = gc.interpolators[(wavelength_in_nm, (ox,oy), x_or_y, which_amp)]
                    amps = f(points_to_interpolate_at)
                    if which_amp == 'ampfy':
                        Exp[entries] += (E_weight * amps
                                         * kxp * kyp / (k_glass * kzp) / n_glass
                                         * phase_from_offcenter)
                        Eyp[entries] += (E_weight * amps
                                         * (-kxp**2 - kzp**2) / (k_glass * kzp) / n_glass
                                         * phase_from_offcenter)
                        Hxp[entries] += H_weight * amps * phase_from_offcenter
                    else:
                        Exp[entries] += (E_weight * amps
                                         * (kyp**2 + kzp**2) / (k_glass * kzp) / n_glass
                                         * phase_from_offcenter)
                        Eyp[entries] += (E_weight * amps
                                         * -kxp*kyp / (k_glass * kzp) / n_glass
                                         * phase_from_offcenter)
                        Hyp[entries] += H_weight * amps * phase_from_offcenter




    
    # note that the S4 individual grating simulations assume the light has
    # phase 0 at (x,y)=grating_center, z=air-pillar interface.
    # Note also that S4 propagates using e^{+ikr}
    # Remember, in dipole_field_Hx etc., we included everything but e^ikr
    if source_z > -inf:
        air_propagation_distance = ((gratingcenter_x - source_x)**2
                                    + (gratingcenter_y - source_y)**2
                                    + source_z**2)**0.5
        eikr = np.exp(1j * kvac * air_propagation_distance)
        Exp *= eikr
        Eyp *= eikr
        #Ez *= eikr
        Hxp *= eikr
        Hyp *= eikr
        #Hz *= eikr

    # double-check signs: If grating_rotation=10deg (first quadrant) and
    # Exp=1, Eyp=0 (E points outward), then Ex>0,Ey>0
    Ex = Exp * np.cos(grating_rotation) - Eyp * np.sin(grating_rotation)
    Ey = Exp * np.sin(grating_rotation) + Eyp * np.cos(grating_rotation)
    Hx = Hxp * np.cos(grating_rotation) - Hyp * np.sin(grating_rotation)
    Hy = Hxp * np.sin(grating_rotation) + Hyp * np.cos(grating_rotation)
    # Note E=H=0 outside lens periphery
    
    ############    Next, the center part of the lens! ###############
    
    x = x_meshgrid[in_center]
    y = y_meshgrid[in_center]
    # closest_indices[j] is the index of the entry in lens_center_summary
    # that is closest to (x[j],y[j])
    mytree = cKDTree(lens_center_summary[:,0:2])
    closest_indices = mytree.query(np.vstack((x,y)).T)[1]
    cell_center_x = lens_center_summary[closest_indices, 0]
    cell_center_y = lens_center_summary[closest_indices, 1]
    which_grating = lens_center_summary[closest_indices, 2].astype(int)
    
    Ex_centerpoints = np.zeros_like(x, dtype=complex)
    Ey_centerpoints = np.zeros_like(x, dtype=complex)
    Hx_centerpoints = np.zeros_like(x, dtype=complex)
    Hy_centerpoints = np.zeros_like(x, dtype=complex)
    
    # how much to weight the results with x-polarized and y-polarized input
    H_x_weight = dipole_field_Hy
    H_y_weight = dipole_field_Hx
    
    if source_z > -inf:
        dx = x - source_x
        dy = y - source_y
        dz = 0 - source_z
        distance = (dx**2 + dy**2 + dz**2)**0.5
        # (ux,uy,uz) is the unit vector that the incoming light is traveling.
        ux = dx / distance
        uy = dy / distance
        uz = dz / distance
    else:
        ux = uy = np.zeros_like(x)
        uz = np.ones_like(x)
    all_orders = {(e['ox'],e['oy']) for g in hexgridset.grating_list for e in g.data}
    x_period = hexgridset.grating_list[0].grating_period
    y_period = hexgridset.grating_list[0].lateral_period
    for ox,oy in all_orders:
        # ux,uy is propagation direction in air. So use kvac here, not kglass
        kx = kvac * ux + ox * 2*pi/x_period
        ky = kvac * uy + oy * 2*pi/y_period
        
        entries = (kx**2 + ky**2 <= kvac**2)
        if entries.sum() == 0:
            continue
        print('diffraction order', (ox,oy), 'of center; applies at', entries.sum(), 'points', flush=True)
        kx = kx[entries]
        ky = ky[entries]
        kz = (k_glass**2-kx**2-ky**2)**0.5
        # S4 references phases to the pillar-glass interface, center of the
        # grating unit cell. Because we want the field at a different point,
        # we need a phase propagation factor
        phase_from_offcenter = np.exp(1j * (kx * (x[entries] - cell_center_x[entries])
                                          + ky * (y[entries] - cell_center_y[entries])))

        points_to_interpolate_at = np.vstack((ux[entries], uy[entries], which_grating[entries])).T
        if ux[entries].min() < hexgridset.interpolator_bounds[0]:
            raise ValueError('need to calculate at smaller ux!', ux[entries].min(), hexgridset.interpolator_bounds[0])
        if ux[entries].max() > hexgridset.interpolator_bounds[1]:
            raise ValueError('need to calculate at bigger ux!', ux[entries].max(), hexgridset.interpolator_bounds[1])
        if uy[entries].min() < hexgridset.interpolator_bounds[2]:
            raise ValueError('need to calculate at smaller uy!', uy[entries].min(), hexgridset.interpolator_bounds[2])
        if uy[entries].max() > hexgridset.interpolator_bounds[3]:
            raise ValueError('need to calculate at bigger uy!', uy[entries].max(), hexgridset.interpolator_bounds[3])
        for x_or_y in ('x', 'y'):
            H_weight = H_x_weight[in_center][entries] if x_or_y == 'x' else H_y_weight[in_center][entries]
            E_weight = H_weight * nu.Z0
            for which_amp in ('ampfy', 'ampfx'):
                f = hexgridset.interpolators[(wavelength_in_nm, (ox,oy), x_or_y, which_amp)]
                amps = f(points_to_interpolate_at)
                if which_amp == 'ampfy':
                    Ex_centerpoints[entries] += (E_weight * amps
                                                 * kx * ky / (k_glass * kz) / n_glass
                                                 * phase_from_offcenter)
                    Ey_centerpoints[entries] += (E_weight * amps
                                                 * (-kx**2 - kz**2) / (k_glass * kz) / n_glass
                                                 * phase_from_offcenter)
                    Hx_centerpoints[entries] += H_weight * amps * phase_from_offcenter
                else:
                    Ex_centerpoints[entries] += (E_weight * amps
                                                 * (ky**2 + kz**2) / (k_glass * kz) / n_glass
                                                 * phase_from_offcenter)
                    Ey_centerpoints[entries] += (E_weight * amps
                                                 * -kx*ky / (k_glass * kz) / n_glass
                                                 * phase_from_offcenter)
                    Hy_centerpoints[entries] += H_weight * amps * phase_from_offcenter
#                temp = x_meshgrid*0
#                temp2 = temp[in_center]
#                temp2[entries] += amps
#                temp[in_center] += temp2
#                #temp[in_center][entries] = E_weight
#                plt.figure()
#                plt.imshow(temp.T)
#                plt.title(s_or_p + '  ' + which_amp + '  ' + str((ox,oy)))
#                plt.colorbar()
#

    if source_z > -inf:
        air_propagation_distance = ((cell_center_x - source_x)**2
                                    + (cell_center_y - source_y)**2
                                    + source_z**2)**0.5
        eikr = np.exp(1j * kvac * air_propagation_distance)
        Ex_centerpoints *= eikr
        Ey_centerpoints *= eikr
        Hx_centerpoints *= eikr
        Hy_centerpoints *= eikr

    Ex[in_center] += Ex_centerpoints
    Ey[in_center] += Ey_centerpoints
    Hx[in_center] += Hx_centerpoints
    Hy[in_center] += Hy_centerpoints
    
#    a = Ex / (dipole_field_Ex*np.exp(1j * kvac * ((x_meshgrid-source_x)**2 + (y_meshgrid-source_y)**2 + source_z**2)**0.5))
#    plt.figure()
#    plt.imshow(a.real.T)
#    plt.colorbar()
    
    # TODO - Check for Possible factor-of-2 error??
    local_power_z = dipole_field_Ex * dipole_field_Hy - dipole_field_Ey * dipole_field_Hx
    entries = np.logical_or((which_gratingcollection != -1), in_center)
    power_passing_through_lens = (local_power_z[entries].sum()
                                     * (x_pts[1]-x_pts[0]) * (y_pts[1]-y_pts[0]))


    return Ex, Ey, Hx, Hy, x_pts, y_pts, power_passing_through_lens, n_glass

def build_nearfield_big(source_x, source_y, source_z, source_pol, wavelength,
                        lens_periphery_summary, lens_center_summary, hexgridset,
                        x_pts=None, y_pts=None, dipole_moment=1e-30 * nu.C * nu.m):
    """build_nearfield() uses a lot of temporary storage. With lots of
    near-field points, this function avoids running out of memory by filling in
    a subset of the points at a time"""
    pts_at_a_time = 1e7
    y_pts_at_a_time = int(pts_at_a_time / x_pts.size)
    
    Ex = np.zeros(shape=(x_pts.size,y_pts.size), dtype=complex)
    Ey = np.zeros(shape=(x_pts.size,y_pts.size), dtype=complex)
    Hx = np.zeros(shape=(x_pts.size,y_pts.size), dtype=complex)
    Hy = np.zeros(shape=(x_pts.size,y_pts.size), dtype=complex)
    power_passing_through_lens=0
    
    start = 0
    end = min(start+y_pts_at_a_time, y_pts.size)
    while start < y_pts.size:
        print('running y-index', start, 'to', end, 'out of', y_pts.size, flush=True)
        y_pts_now = y_pts[start:end]
        Ex_now,Ey_now,Hx_now,Hy_now,_,_,P_now,n_glass = build_nearfield(
                        source_x=source_x, source_y=source_y, source_z=source_z,
                        source_pol=source_pol, wavelength=wavelength,
                        lens_periphery_summary=lens_periphery_summary, 
                        lens_center_summary=lens_center_summary, hexgridset=hexgridset,
                        x_pts=x_pts, y_pts=y_pts_now, dipole_moment=dipole_moment)
        Ex[:, start:end] = Ex_now
        Ey[:, start:end] = Ey_now
        Hx[:, start:end] = Hx_now
        Hy[:, start:end] = Hy_now
        power_passing_through_lens += P_now
        start = end
        end = min(start+y_pts_at_a_time, y_pts.size)
        
    return Ex, Ey, Hx, Hy, x_pts, y_pts, power_passing_through_lens, n_glass
    
