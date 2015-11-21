# coding=UTF-8
"""
(C) 2015 Steven Byrnes

Nearfield-farfield transform
"""
import numpy as np
from numpy.fft import fft2, fftshift
from math import pi
import numericalunits as nu
import matplotlib.pyplot as plt


def farfield_from_nearfield(fftEx, fftEy, fftHx, fftHy, xp_list, yp_list, wavelength, n_glass):
    """following Taflove 1995.
    xp, yp (really x', y', p stands for "prime") are coordinates on the lens,
    with xp=yp=0 at the lens center
    fftEx is fft2(fftshift(Ex)) and ditto with the others. I'm doing it this
    way rather than FFTing inside the function so that the real-space data can
    be more easily deleted to reclaim precious RAM"""
    
    dxp = xp_list[1] - xp_list[0]
    dyp = yp_list[1] - yp_list[0]
    num_x = len(xp_list)
    num_y = len(yp_list)
    assert fftEx.shape == fftEy.shape == fftHx.shape == fftHy.shape == (num_x, num_y)
    for l in [xp_list,yp_list]:
        diffs = [l[i+1] - l[i] for i in range(len(l)-1)]
        assert 0 < diffs[0] < wavelength/2
        assert max(diffs) - min(diffs) <= 1e-9 * max(abs(d) for d in diffs)
    
    # these formulas are justified in farfield_from_nearfield_helper()
    # Note that ux,uy are the x- and y- direction cosines (components of the
    # unit vector in the propagation direction) ... in glass not air
    ux_list = np.arange(num_x) * (wavelength/n_glass) / (dxp * num_x)
    uy_list = np.arange(num_y) * (wavelength/n_glass) / (dyp * num_y)
    # pick the correctly aliased version
    ux_list[ux_list > ux_list.max()/2] -= (wavelength/n_glass) / dxp
    uy_list[uy_list > uy_list.max()/2] -= (wavelength/n_glass) / dyp
    
    # To conserve RAM we run the rest of the calculation in chunks, using
    # farfield_from_nearfield_helper(), so that intermediate results can be
    # discarded
    
    pts_at_a_time = 1e7
    uy_pts_at_a_time = int(pts_at_a_time / ux_list.size)
    
    P_here_times_r2_over_uz = np.zeros(shape=(ux_list.size,uy_list.size), dtype=float)
    
    start = 0
    end = min(start+uy_pts_at_a_time, uy_list.size)
    while start < uy_list.size:
        print('running uy-index', start, 'to', end, 'out of', uy_list.size)
        uy_list_now = uy_list[start:end]
        
        temp = farfield_from_nearfield_helper(fftEx=fftEx[:,start:end],
                                              fftEy=fftEy[:,start:end],
                                              fftHx=fftHx[:,start:end],
                                              fftHy=fftHy[:,start:end],
                                              ux_list=ux_list, uy_list=uy_list_now,
                                              dxp=dxp, dyp=dyp, wavelength=wavelength,
                                              n_glass=n_glass)
        P_here_times_r2_over_uz[:, start:end] = temp
        
        start = end
        end = min(start+uy_pts_at_a_time, uy_list.size)
    
    P_here_times_r2_over_uz = fftshift(P_here_times_r2_over_uz)
    ux_list = fftshift(ux_list)
    uy_list = fftshift(uy_list)
    dux = ux_list[1] - ux_list[0]
    duy = uy_list[1] - uy_list[0]
    ux,uy = np.meshgrid(ux_list, uy_list, indexing='ij', sparse=True)
    total_P = (P_here_times_r2_over_uz * dux * duy)[np.isfinite(P_here_times_r2_over_uz)].sum()
    return P_here_times_r2_over_uz, total_P, ux, uy, dux, duy

def farfield_from_nearfield_helper(fftEx, fftEy, fftHx, fftHy, ux_list,
                                       uy_list, dxp, dyp, wavelength, n_glass):
    """Don't run directly, this is called by farfield_from_nearfield().
    For some entries of the FFT of Ex, Ey, Hx, Hy, with the corresponding
    direction cosines (in glass) ux_list, uy_list, calculate the outgoing power
    
    We analyze only a subset of the entries at a time so that we can do
    calculations without worrying that we'll run out of RAM.
    
    Note that fftEx[i,j] must correspond to the direction cosine
    (ux_list[i], uy_list[j]). But the lists are not sorted - we didn't run
    fftshift on them.
    """
    assert fftEx.shape == fftEy.shape == fftHx.shape == fftHy.shape == (ux_list.size, uy_list.size)
    ux,uy = np.meshgrid(ux_list, uy_list, indexing='ij', sparse=True)

    """
    (8.15): J=n×H, M=-n×E. n is the outward-pointing normal, i.e. +zhat. So
    Jy = Hx , Jx = -Hy , My = -Ex , Mx = Ey

    (8.17): N = integral J e^{-ikr' cos psi}, L = integral M e^{-ikr' cos psi}
    Note j=-i because I'm using e^+ikx but Taflove is using e^-jkx
    r'=(x',y',0) = position on the lens, psi = angle between r' and 
    r = (ux,uy,uz) * infinity. We calculate r' cos psi = x'*ux + y'*uy, so
    now N = (integral over x',y') J e^{-i*k*(x'*ux + y'*uy)}
    Rewrite this as
    N(ux,uy) = Δx' * Δy' * (sum over x',y') J(x',y') e^{-i*k*(x'*ux + y'*uy)}
    Now I need to shift J...
    J[m1,m2] = J(x'min + m1 * Δx', y'min + m2 * Δy')
    Jshift[m1,m2] = J(m1 * Δx', m2 * Δy')
    This is what np.fft.fftshift() does.Of course this formula only really
    makes sense if J is periodic, which amounts to discretizing the frequency,
    but we were going to do that anyway.
    
    Now we have
    N(ux,uy) = Δx' * Δy' * (sum over m1,m2) Jshift[m1,m2] e^{-i*k*(m1 * Δx' * ux + m2 * Δy' * uy)}
    Next, let Jhat be the FFT of Jshift. The numpy conventions page says:
    http://docs.scipy.org/doc/numpy/reference/routines.fft.html
    Jhat[i1,i2] = (sum over m1,m2) J[m1,m2] e^{-2*pi*i*(i1*m1/num_x' + i2*m2/num_y')}
    I want this to match with . . . . . . . e^{-i*k*(m1 * Δx' * ux + m2 * Δy' * uy)}
    so we get the following relation between ux,uy and i1,i2:
        ux = (lambda/Δx') * i1 / num_x' , uy = (lambda/Δy') * i2 / num_y'
    With that,
    N(ux,uy) = Δx' * Δy' * Jhat[i1,i2]
    
    Note that the FFT won't distinguish between ux and (ux + lambda/Δx').
    So if Δx' > lambda/2 then I cannot cover -1 < ux < +1 without
    aliasing different parts of that range together. This is what I expect.
    When you sample at exactly half the wavelength, you can't tell apart waves
    traveling at (+k,0,0) from (-k,0,0), i.e. perfectly glancing.
    
    From  N = integral J e^{-ikr' cos psi} it seems intuitively clear that I
    should take refractive index into account by using k in the medium, i.e.
    replace wavelength with (wavelength/n_glass). I'm planning to allow some
    aliasing because I don't expect there to be much light with k_inplane > kvac
    because I'm excluding those modes. But I should double-check with proper sampling
    """

    Nx = -fftHy * dxp * dyp
    Ny = fftHx * dxp * dyp
    Lx = fftEy * dxp * dyp
    Ly = -fftEx * dxp * dyp
    
#    uz = (1 - ux**2 - uy**2)**0.5
#    # note: uz is nan sometimes
#    Z = nu.Z0 / n_glass
#    Efar_sq_over_r2 = (abs(Ly * uz + Nx * Z)**2 + abs(-Lx * uz + Ny * Z)**2
#                            + abs(Lx * uy - Ly * ux)**2) * (2*pi*n_glass/wavelength / (4*pi))**2
#    print('unittest', Efar_sq_over_r2[70,70] / ((nu.V/nu.m)**2 / nu.m**2))
#    return
    
    """(8.23-4). Note that cos θ = uz,  sin θ cos φ = ux,  sin θ sin φ = uy
    Therefore cos θ cos φ = ux * sqrt(ux^2+uy^2) / uz
    ux / sin θ = ux / sqrt(1-uz^2)"""
    #uz = (1 - ux**2 - uy**2)**0.5
    # actually, I like avoiding the numpy RunTimeWarning
    uz = (1 - ux**2 - uy**2)
    uz[uz<0] = np.nan
    uz = uz**0.5
    sintheta = (ux**2 + uy**2)**0.5
    # add 1e-9 to avoid divide-by-zero RunTimeWarning, I'll fix it in a minute
    Ntheta = Nx * ux * uz / (sintheta + 1e-9) + Ny * uy * uz / (sintheta + 1e-9)
    Nphi = -Nx * uy / (sintheta + 1e-9) + Ny * ux / (sintheta + 1e-9)
    # for ux=uy=0, take the limit where uy=0, ux is positive infinitesimal
    i = np.where(ux==0)[0]
    j = np.where(uy==0)[0]
    Ntheta[i,j] = Nx[i,j]
    Nphi[i,j] = Ny[i,j]
    del Nx, Ny
    Ltheta = Lx * ux * uz / (sintheta + 1e-9) + Ly * uy * uz / (sintheta + 1e-9)
    Lphi = -Lx * uy / (sintheta + 1e-9) + Ly * ux / (sintheta + 1e-9)
    Ltheta[i,j] = Lx[i,j]
    Lphi[i,j] = Ly[i,j]
    del Lx, Ly, sintheta
    

    """(8.25)"""
    # We imagine a hemispherical surface of radius R, with local power P(ux,uy).
    # The total power flux is (integral over surface) P d(surface area)
    # with d(surface area) = (dux * duy / uz) * r^2.
    # Proof: Imagine you're looking overhead at a hemisphere. You draw a square
    #(from your POV). The area on the hemisphere that the square covers is
    # bigger than its projection in your POV, because of the slope. The ratio
    # is uz.
    # So all in all, (integral over surface) (P*r^2 / uz) dux duy
    
    Z = nu.Z0 / n_glass
    P_here_times_r2_over_uz = ((2*pi*n_glass/wavelength)**2 / (32*pi**2*Z)
                      * (abs(Lphi + Z * Ntheta)**2 + abs(Ltheta - Z * Nphi)**2)) / (uz+1e-5)
    del uz
                      
    # mystery factor - empty aperture should be 100% transmissive etc
    P_here_times_r2_over_uz *= 2
    
    return P_here_times_r2_over_uz


#############
# This is an alternative implementation, following a different
# reference. The results are exactly the same. This one is a bit slower though.

#def farfield_from_nearfield2(Ex, Ey, Hx, Hy, xp_list, yp_list, wavelength, n_glass):
#    """following "Understanding the Finite-Difference Time-Domain Method",
#    John B. Schneider, www.eecs.wsu.edu/~schneidj/ufdtd
#    
#    xp, yp (really x', y', p stands for "prime") are coordinates on the lens,
#    with xp=yp=0 at the lens center
#    dxp (really Δx', p stands for "prime") is the spacing between sample points
#    on the lens
#    """
#    dxp = xp_list[1] - xp_list[0]
#    dyp = yp_list[1] - yp_list[0]
#    num_x = len(xp_list)
#    num_y = len(yp_list)
#    assert Ex.shape == Ey.shape == Hx.shape == Hy.shape == (num_x, num_y)
#    for l in [xp_list,yp_list]:
#        diffs = [l[i+1] - l[i] for i in range(len(l)-1)]
#        assert 0 < diffs[0] < wavelength/2
#        assert max(diffs) - min(diffs) <= 1e-9 * max(abs(d) for d in diffs)
#    """
#    (14.3-4): J=n×H, M=-n×E. n is the outward-pointing normal, i.e. +zhat
#    """
#    Jy = Hx
#    Jx = -Hy
#    My = -Ex
#    Mx = Ey
#    """
#    (8.17): N = integral J e^{-ikr' cos psi}, L = integral M e^{-ikr' cos psi}
#    Note j=-i because I'm using e^+ikx but Taflove is using e^-jkx
#    r'=(x',y',0) = position on the lens, psi = angle between r' and 
#    r = (ux,uy,uz) * infinity. We calculate r' cos psi = x'*ux + y'*uy, so
#    now N = (integral over x',y') J e^{-i*k*(x'*ux + y'*uy)}
#    Rewrite this as
#    N(ux,uy) = Δx' * Δy' * (sum over x',y') J(x',y') e^{-i*k*(x'*ux + y'*uy)}
#    Now I need to shift J...
#    J[m1,m2] = J(x'min + m1 * Δx', y'min + m2 * Δy')
#    Jshift[m1,m2] = J(m1 * Δx', m2 * Δy')
#    This is what np.fft.fftshift() does.Of course this formula only really
#    makes sense if J is periodic, which amounts to discretizing the frequency,
#    but we were going to do that anyway.
#    
#    Now we have
#    N(ux,uy) = Δx' * Δy' * (sum over m1,m2) Jshift[m1,m2] e^{-i*k*(m1 * Δx' * ux + m2 * Δy' * uy)}
#    Next, let Jhat be the FFT of Jshift. The numpy conventions page says:
#    http://docs.scipy.org/doc/numpy/reference/routines.fft.html
#    Jhat[i1,i2] = (sum over m1,m2) J[m1,m2] e^{-2*pi*i*(i1*m1/num_x' + i2*m2/num_y')}
#    I want this to match with . . . . . . . e^{-i*k*(m1 * Δx' * ux + m2 * Δy' * uy)}
#    so we get the following relation between ux,uy and i1,i2:
#        ux = (lambda/Δx') * i1 / num_x' , uy = (lambda/Δy') * i2 / num_y'
#    With that,
#    N(ux,uy) = Δx' * Δy' * Jhat[i1,i2]
#    
#    Note that the FFT won't distinguish between ux and (ux + lambda/Δx').
#    So if Δx' > lambda/2 then I cannot cover -1 < ux < +1 without
#    aliasing different parts of that range together. This is what I expect.
#    When you sample at exactly half the wavelength, you can't tell apart waves
#    traveling at (+k,0,0) from (-k,0,0), i.e. perfectly glancing.
#    
#    From  N = integral J e^{-ikr' cos psi} it seems intuitively clear that I
#    should take refractive index into account by using k in the medium, i.e.
#    replace wavelength with (wavelength/n_glass). I'm planning to allow some
#    aliasing because I don't expect there to be much light with k_inplane > kvac
#    because I'm excluding those modes. But I should double-check with proper sampling
#    """
#    ux,uy = np.meshgrid(np.arange(num_x) * (wavelength/n_glass) / (dxp * num_x),
#                        np.arange(num_y) * (wavelength/n_glass) / (dyp * num_y), indexing='ij')
#    # little test
#    if True:
#        i1,i2 = 1,3
#        assert ux[i1,i2] == (wavelength/n_glass) * i1 / (dxp * num_x)
#        assert uy[i1,i2] == (wavelength/n_glass) * i2 / (dyp * num_y)
#    ux[ux > ux.max()/2] -= (wavelength/n_glass) / dxp
#    uy[uy > uy.max()/2] -= (wavelength/n_glass) / dyp
#    
#    Nx = fft2(fftshift(Jx)) * dxp * dyp
#    Ny = fft2(fftshift(Jy)) * dxp * dyp
#    Lx = fft2(fftshift(Mx)) * dxp * dyp
#    Ly = fft2(fftshift(My)) * dxp * dyp
#    
#    # We have Nx[i,j] corresponding to the direction (ux[i,j], uy[i,j]). That's
#    # fine. But it's nice to have everything sorted, for imshow plots etc.
#    ux = fftshift(ux)
#    uy = fftshift(uy)
#    Nx = fftshift(Nx)
#    Ny = fftshift(Ny)
#    Lx = fftshift(Lx)
#    Ly = fftshift(Ly)
#    
#    """
#    Above we said
#    
#    N = integral J e^{-ikr' cos psi}, L = integral M e^{-ikr' cos psi}
#    
#    Compare to (14.32) (remember that we are switching the sign of i to use
#    e^+ikr, but also e^-ik|r-r'| turns into e^+ikr'sin(psi). So the sign is
#    right).
#    We get A(r) = (mu0/4pi) N e^ikr/r
#           F(r) = (eps/4pi) L e^ikr/r
#    
#    Then (14.52) says
#    E = +iw [A - khat(khat dot A)] - 1j/eps * k cross F
#    """
#    uz = (1 - ux**2 - uy**2)**0.5
#    eps = nu.eps0 * n_glass**2
#    Ax_over_eikrr = Nx * nu.mu0 / (4*pi)
#    Ay_over_eikrr = Ny * nu.mu0 / (4*pi)
#    Fx_over_eikrr = Lx * eps / (4*pi)
#    Fy_over_eikrr = Ly * eps / (4*pi)
#    u_dot_A_over_eikrr = Ax_over_eikrr * ux + Ay_over_eikrr * uy
#    kx = (2*pi*n_glass / wavelength) * ux
#    ky = (2*pi*n_glass / wavelength) * uy
#    kz = (2*pi*n_glass / wavelength) * uz
#    
#    omega = nu.c0 * 2*pi/wavelength
#    Ex_over_eikrr = (1j * omega * (Ax_over_eikrr - ux * u_dot_A_over_eikrr)
#                      - 1j/eps * (-kz * Fy_over_eikrr))
#    Ey_over_eikrr = (1j * omega * (Ay_over_eikrr - uy * u_dot_A_over_eikrr)
#                      - 1j/eps * (kz * Fx_over_eikrr))
#    Ez_over_eikrr = (1j * omega * (     0        - uz * u_dot_A_over_eikrr)
#                      - 1j/eps * (kx * Fy_over_eikrr - ky * Fx_over_eikrr))
#    absE2_times_r2 = abs(Ex_over_eikrr)**2 + abs(Ey_over_eikrr)**2 + abs(Ez_over_eikrr)**2
#                      
#    #print('unittest', absE2_over_r2[70,70] / ((nu.V/nu.m)**2 * nu.m**2))
#    
#    Z = nu.Z0 / n_glass
#    
#    #Power flow per area in the direction normal to the direction it's flowing
#    P_here_times_r2 = absE2_times_r2 / Z
#    
#    # We imagine a hemispherical surface of radius R, with local power P(ux,uy).
#    # The total power flux is (integral over surface) P dsurface
#    # d(surface area) = (dux * duy / uz) * r^2.
#    # So all in all, (integral over surface) (P*r^2 / uz) dux duy
#    
#    dux = ux[1,0] - ux[0,0]
#    duy = uy[0,1] - uy[0,0]
#    P_here_times_r2_over_uz = P_here_times_r2 / uz
#    #P_here_times_r2_over_uz[1-np.isfinite(P_here_times_r2_over_uz)] = 0
#    #P_here_times_r2_over_uz[np.isnan(P_here_times_r2_over_uz)] = 0
#    total_P = (P_here_times_r2_over_uz * dux * duy)[np.isfinite(P_here_times_r2_over_uz)].sum()
#    return P_here_times_r2_over_uz, total_P, ux, uy, dux, duy
#    
#
