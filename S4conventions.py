# -*- coding: utf-8 -*-
"""
(c) 2015 Steven Byrnes

This script was used to help write and debug the software, it serves no
external purpose. Specifically, S4 software returns complex amplitudes but
doesn't explain well the meaning of the amplitudes. This file is figuring it
out, by checking against the electric and magnetic fields.

I posted some of what I learned at
https://github.com/victorliu/S4/pull/25/files

"""
import math
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import cmath
import grating, gratingdata
from random import random



# my units package
# https://pypi.python.org/pypi/numericalunits
# just run the command "pip install numericalunits"
import numericalunits as nu
from numericalunits import m, nm, um

pi = math.pi
inf = float('inf')
degree = pi / 180

def array_almost_equal(a,b):
    a,b = np.array(a), np.array(b)
    return (abs(a-b)).max() <= 1e-9 * (abs(a) + abs(b)).max()

def sp_polarization(kx, ky, kz, n):
    """If an amplitude in "s" or "p" polarization is 1, find the E and H field
    that S4 would output. (...which has H in arbitrary units, E in the same units
    but with a factor of Z0 (impedance of free space)).
    n is the index."""
    assert n==1 # Haven't worked out the n>1 case
    if kx == ky == 0:
        # warning: program treats -0.0 differently from 0.0...
        Ep = [1,0,0]
        Es = [0,1,0]
        Hp = [0,1,0]
        Hs = [-1,0,0]
        return [np.array(v) for v in (Es,Ep,Hs,Hp)]
    k = (kx**2 + ky**2 + kz**2)**0.5
    Es = [-ky/(kx**2+ky**2)**0.5, kx/(kx**2+ky**2)**0.5, 0]
    Ep = [kx*kz/(k * (kx**2+ky**2)**0.5),
          ky*kz/(k * (kx**2+ky**2)**0.5),
          -(kx**2+ky**2)**0.5 / k]
    Es=np.array(Es) ; Ep = np.array(Ep)
    
    # alternative / derivation
    Es_unnorm = np.cross([0,0,1], [kx,ky,kz])
    Ep_unnorm = np.cross(Es_unnorm, [kx,ky,kz])
    Es_alt = Es_unnorm / sum(Es_unnorm**2)**0.5
    Ep_alt = Ep_unnorm / sum(Ep_unnorm**2)**0.5
    assert array_almost_equal(Es, Es_alt)
    assert array_almost_equal(Ep, Ep_alt)
    
    Hp = Es
    Hs = -Ep
    return Es,Ep,Hs,Hp

def xy_polarization(kx,ky,kz, n):
    """For output amplitudes, S4 uses a different polarization scheme, which
    we'll call 'x' and 'y' polarization.
    
    There are a couple funny things about it: (1) The unit-amplitude fields are
    actually not normalized, (2) The x and y are not orthogonal (except if
    kx=0 or ky=0). But it's actually kinda nice in other ways: Mainly, the fact
    that it has no discontinuous changes, unlike s and p near normal. So I'm
    actually using it myself by choice for incoming waves
    (see grating.characterize()).
    
    Anyway, if an x-polarized or y-polarized amplitude is 1, find the E and H
    field that S4 would output.
    
    n is the index of refraction"""
    if kx == ky == 0:
        signkz = math.copysign(1,kz)
        E_xpol = [signkz/n,0,0]
        E_ypol = [0,-signkz/n,0]
        H_xpol = [0,1,0]
        H_ypol = [1,0,0]
        return [np.array(v) for v in (E_xpol,E_ypol,H_xpol,H_ypol)]
        
    k = (kx**2 + ky**2 + kz**2)**0.5
    H_xpol = [0,1,-ky/kz]
    E_xpol = [(ky**2+kz**2)/(k*kz*n), -kx*ky/(k*kz*n), -kx/(k*n)]
    H_ypol = [1,0,-kx/kz]
    E_ypol = [kx*ky/(k*kz*n), (-kx**2-kz**2)/(k*kz*n), ky/(k*n)]
    # alternative / derivation
    E_xpol_alt = np.cross(H_xpol, [kx/k, ky/k, kz/k]) / n
    E_ypol_alt = np.cross(H_ypol, [kx/k, ky/k, kz/k]) / n
    assert array_almost_equal(E_xpol, E_xpol_alt)
    assert array_almost_equal(E_ypol, E_ypol_alt)
    return [np.array(v) for v in (E_xpol,E_ypol,H_xpol,H_ypol)]

def x_from_sp(kx,ky,kz,n):
    """how to create incoming x polarization from combining s and p. Just
    checking my math here"""
    assert n==1
    Es,Ep,Hs,Hp = sp_polarization(kx,ky,kz,n=n)
    E_xpol,E_ypol,H_xpol,H_ypol = xy_polarization(kx,ky,kz,n=n)
    k = (kx**2 + ky**2 + kz**2)**0.5
    p_coef = kx/(kx**2+ky**2)**0.5
    s_coef = -ky*k/(kz*(kx**2+ky**2)**0.5)
    assert array_almost_equal(p_coef*Hp + s_coef*Hs, H_xpol)
    assert array_almost_equal(p_coef*Ep + s_coef*Es, E_xpol)

def y_from_sp(kx,ky,kz,n):
    """how to create incoming y polarization by combining s and p. Just
    checking my math here"""
    assert n==1
    Es,Ep,Hs,Hp = sp_polarization(kx,ky,kz,n=n)
    E_xpol,E_ypol,H_xpol,H_ypol = xy_polarization(kx,ky,kz,n=n)
    k = (kx**2 + ky**2 + kz**2)**0.5
    p_coef = -ky/(kx**2+ky**2)**0.5
    s_coef = -kx*k/(kz*(kx**2+ky**2)**0.5)
    assert array_almost_equal(p_coef*Hp + s_coef*Hs, H_ypol)
    assert array_almost_equal(p_coef*Ep + s_coef*Es, E_ypol)

def arbitrary_from_xy(Hx,Hy,kx,ky,kz,n):
    """how to recreate an arbitrary field by combing x and y polarizations.
    Just checking my math here. Oh, actually this is obvious."""
    E_xpol,E_ypol,H_xpol,H_ypol = xy_polarization(kx,ky,kz,n=n)
    x_coef = Hy
    y_coef = Hx
    assert array_almost_equal((x_coef*H_xpol + y_coef*H_ypol)[0:2], [Hx,Hy])

x_from_sp(random(),random(),random(),n=1)
y_from_sp(random(),random(),random(),n=1)
arbitrary_from_xy(random(),random(),random(), random(), random(),n=random())


"""The rest of the file is for making S4 output (1) Complex amplitudes of
propagating diffraction orders, and (2) Actual E and H fields at particular
points. Then find the formula for calculating (2) given (1)."""



def read_fields(mygrating, target_wavelength=580*nm):
    """When grating.lua is set up to output the complex amplitudes and then
    real-space fields, this function is for parsing that output.
    Again, this function will not work if you don't uncomment certain lines
    in grating.lua."""
    # Assuming lua is set up to spit out near-field...
    ux = math.sin(mygrating.get_angle_in_air(580*nm)) - 0.1
    uy = 0.123
    process = mygrating.run_lua_initiate(ux_min=ux, ux_max=ux, uy_min=uy,
                                         uy_max=uy, u_steps=1, wavelength=600*nm)
    output, error = process.communicate()
    output = output.decode()
    
    grating_amplitude_data = []
    lines = output.split('\r\n')
    for i,line in enumerate(lines):
        if line[0:6] == 'Fields':
            break
        split = line.split()
        #print(split)
        if len(split) > 0:
            grating_amplitude_data.append({'wavelength': float(split[0]),
                             's_or_p': split[1],
                             'ux':float(split[2]),
                             'uy':float(split[3]),
                             'ox':int(split[4]),
                             'oy':int(split[5]),
                             'ampfy':float(split[6]) + 1j * float(split[7]),
                             'ampfx':float(split[8]) + 1j * float(split[9]),
                             'ampry':float(split[10]) + 1j * float(split[11]),
                             'amprx':float(split[12]) + 1j * float(split[13])})
    field_data = []
    for line in lines[i+1:]:
        if line != '':
            field_data.append([float(x) for x in line.split('\t')])
    field_data = array(field_data)
    x_list = np.unique(field_data[:,0]) * um
    y_list = np.unique(field_data[:,1]) * um
    z = field_data[0,2] * um
    assert (field_data[:,2] * um == z).all()
    # E[ix, iy, 2] is Ez at the point x_list[ix],y_list[iy]
    E = np.empty(shape=(len(x_list), len(y_list),3), dtype=complex)
    H = np.empty(shape=(len(x_list), len(y_list),3), dtype=complex)
    for row in field_data:
        ix = np.argmin(abs(x_list/um - row[0]))
        iy = np.argmin(abs(y_list/um - row[1]))
        E[ix,iy,0] = row[3] + 1j * row[9]
        E[ix,iy,1] = row[4] + 1j * row[10]
        E[ix,iy,2] = row[5] + 1j * row[11]
        H[ix,iy,0] = row[6] + 1j * row[12]
        H[ix,iy,1] = row[7] + 1j * row[13]
        H[ix,iy,2] = row[8] + 1j * row[14]
    
    return E,H,x_list,y_list,z,grating_amplitude_data
    
    
def E_from_amplitudes(x, y, z, grating_amplitude_list, mygrating):
    """z is relative to air-cylinder interface, i.e. the start of the first S4
    layer.
    
    Note, you need to set pol by hand in the first line to agree with
    grating.lua"""
    
    pol = 's' # TODO - read from lua. For now, set it by hand.
    assert len({x['wavelength'] for x in grating_amplitude_list}) == 1
    wavelength_in_nm = grating_amplitude_list[0]['wavelength']
    output_amplitude_list = [e for e in grating_amplitude_list if e['s_or_p'] == pol]
    #num_orders = len(output_amplitude_list)
    z_above_cyl = z - mygrating.cyl_height
    # not interested in evanescent crap
    assert z_above_cyl > 3*um or z < -3*um
    kvac = 2*pi / (wavelength_in_nm * nm)
    n_glass = mygrating.n_glass if mygrating.n_glass > 0 else grating.n_glass(wavelength_in_nm)
    kglass = kvac * n_glass
        
    E = np.array([0,0,0], dtype=complex)
    H = np.array([0,0,0], dtype=complex)
    
    ktotal = kglass if z > 0 else kvac
    kz_sign = +1 if z > 0 else -1
    for d in output_amplitude_list:
        ux,uy,ox,oy,ampfy,ampfx,ampry,amprx = d['ux'],d['uy'],d['ox'],d['oy'],d['ampfy'],d['ampfx'],d['ampry'],d['amprx']
        kx_incoming = ux * kvac
        ky_incoming = uy * kvac
        kx = kx_incoming + mygrating.grating_kx * ox
        ky = ky_incoming + 2*pi/mygrating.lateral_period * oy
        kz = kz_sign * (ktotal**2 - kx**2 - ky**2)**0.5
        if kz.imag != 0:
            #the TIR-type orders are propagaing when z>0 but evanescent in air
            assert z<0
            continue
        if z > 0:
            E_fx,E_fy,H_fx,H_fy = xy_polarization(kx,ky,kz,n_glass)
            
            E += ((ampfy * E_fy
                   + ampfx * E_fx)
                   * cmath.exp(1j * kx * x)
                   * cmath.exp(1j * ky * y)
                   * cmath.exp(1j * kz * z_above_cyl))
            H += ((ampfy * H_fy
                   + ampfx * H_fx)
                   * cmath.exp(1j * kx * x)
                   * cmath.exp(1j * ky * y)
                   * cmath.exp(1j * kz * z_above_cyl))
        else:
            E_rx,E_ry,H_rx,H_ry = xy_polarization(kx,ky,kz,1)
            
            E += ((ampry * E_ry
                   + amprx * E_rx)
                   * cmath.exp(1j * kx * x)
                   * cmath.exp(1j * ky * y)
                   * cmath.exp(1j * kz * z))
            H += ((ampry * H_ry
                   + amprx * H_rx)
                   * cmath.exp(1j * kx * x)
                   * cmath.exp(1j * ky * y)
                   * cmath.exp(1j * kz * z))
    if z<0:
        kx = kx_incoming
        ky = ky_incoming
        kz = (ktotal**2 - kx**2 - ky**2)**0.5
        assert kz.imag == 0
        Es,Ep,Hs,Hp = sp_polarization(kx,ky,kz,1)
        if pol == 's':
            amplitude_s = 1
            amplitude_p = 0
        else:
            amplitude_s = 0
            amplitude_p = 1
        
        E += ((amplitude_s * Es
               + amplitude_p * Ep)
               * cmath.exp(1j * kx * x)
               * cmath.exp(1j * ky * y)
               * cmath.exp(1j * kz * z))
        H += ((amplitude_s * Hs
               + amplitude_p * Hp)
               * cmath.exp(1j * kx * x)
               * cmath.exp(1j * ky * y)
               * cmath.exp(1j * kz * z))
            
    
    return E,H


########## Test #######

mygrating = gratingdata.mygrating31a
#mygrating = gratingdata.mygrating45n
mygrating.lateral_period *= 2.7
#mygrating.n_glass = 1.2
E,H,x_list,y_list,z,grating_amplitude_data = read_fields(mygrating, target_wavelength=600*nm)
print('z/um', z/um)

ix = 16
iy = 12
x = x_list[ix]
y = y_list[iy]

E_from_amps, H_from_amps = E_from_amplitudes(x,y,z,grating_amplitude_data,mygrating)
print('Hopefully all of the following are equal to 1.0...')
print('E ratio x', E_from_amps[0] / E[ix,iy,0])
print('E ratio y', E_from_amps[1] / E[ix,iy,1])
print('E ratio z', E_from_amps[2] / E[ix,iy,2])
print('H ratio x', H_from_amps[0] / H[ix,iy,0])
print('H ratio y', H_from_amps[1] / H[ix,iy,1])
print('H ratio z', H_from_amps[2] / H[ix,iy,2])


