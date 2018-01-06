# -*- coding: utf-8 -*-
"""
(C) 2015 Steven Byrnes

After you create the pieces for a meta-lens (GratingCollection's from
grating.py for the lens periphery, and HexGridSet's from lens_center.py for the
lens center), the functions here glue them together into a lens, and export
the final design in summarized form, or as an explicit list of nano-pillars,
or as a DXF or SVG file.
"""

from __future__ import division, print_function
import math
from math import pi
degree = pi / 180
import numpy as np
import matplotlib.pyplot as plt
from numericalunits import um, nm
# http://pythonhosted.org/dxfwrite/
from dxfwrite import DXFEngine as dxf
#https://pypi.python.org/pypi/ezdxf/
import ezdxf
# http://svgwrite.readthedocs.org/en/latest/
import svgwrite

import os
inf = float('inf')

#import loop_post_analysis
import grating
import lens_center

# pitch is cylinder center-to-center separation, which is also the lateral period
pitch = 320*nm
period = pitch * math.sqrt(3) # in meters
cyl_height = 550*nm
n_glass = 0
n_tio2 = 0

FOLDER_NAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp')

xyrra_filename = os.path.join(FOLDER_NAME, 'collimator_xyrra_list.txt')
setup_filename = os.path.join(FOLDER_NAME, 'collimator_setup.txt')

#############################################################################

# radius of collimator
#radius = 410*um

#source_distance = 150*um
wavelength = 580*nm # in vacuum
# refractive index in the medium filling the space between the source and the lens
#refractive_index = 1.458 
refractive_index = 1
refractive_index_in_glass = n_glass

def target_phase(x, source_distance):
    """ x is the distance away from the center of the lens """
    k = 2 * pi * refractive_index / wavelength
    return (-k * (math.sqrt(source_distance**2 + x**2) - source_distance)) % (2*pi)

def target_phase_zeros(radius, source_distance):
    ans = []
    order=0
    k = 2 * pi * refractive_index / wavelength
    while len(ans)==0 or ans[-1] < radius:
        x = (((2*pi*order)/k + source_distance)**2 - source_distance**2)**0.5
        ans.append(x)
        order += 1
    return ans



def hexagonal_grid(n, radius, fourfold_symmetry=True):
    """calculate a list of (x,y) coordinates on a hexagonal grid with
    given nearest-neighbor separation n, only with x^2 + y^2 < radius^2.
    If fourfold_symmetry is True, restrict the design to x,y >= 0, with
    mirror-flips to get to the other quadrants. (Note that x=0 or y=0 points
    will get duplicated.)"""
    # basis vectors are
    # v1 = (0,neighbor_separation)
    # v2 = (neighbor_separation*sqrt(3)/2 , neighbor_separation/2)
    # if (x,y) = n1*v1 + n2*v2, then
    # y = n*(n1 + n2/2)  ;  x = n*n2*sqrt(3)/2
    # n1 = y/n - n2/2  ;  n2 = 2x/(n*sqrt(3))
    
    # calculate (n1,n2) at a few points to figure out bounds on how big they
    # could possibly be (we'll still test the points individually within that
    # range)
    if fourfold_symmetry is True:
        xy_corner_list = [(0,0), (radius,0), (0,radius), (radius,radius)]
    else:
        xy_corner_list = [(radius,radius), (radius, -radius), (-radius, radius), (-radius, -radius)]
    n1n2_corner_list = [(y/n - x/(n * 3**0.5), 2*x/(n * 3**0.5)) for x,y in xy_corner_list]
    xy_list = []
    min_n1 = int(min(n1 for n1,n2 in n1n2_corner_list)) - 2
    max_n1 = int(max(n1 for n1,n2 in n1n2_corner_list)) + 2
    min_n2 = int(min(n2 for n1,n2 in n1n2_corner_list)) - 2
    max_n2 = int(max(n2 for n1,n2 in n1n2_corner_list)) + 2
    
    for n2 in range(min_n2, max_n2+1):
        x = n * n2 * 3**0.5/2
        for n1 in range(min_n1, max_n1+1):
            y = n * (n1+n2/2)
            if fourfold_symmetry is True:
                if x >= 0 and y >= 0 and x**2 + y**2 < radius**2:
                    xy_list.append([x,y])
            else:
                if x**2 + y**2 < radius**2:
                    xy_list.append([x,y])
    if False:
        #display the points
        plt.figure()
        for x,y in xy_list:
            plt.plot(x,y, 'k.')
            plt.gca().set_aspect('equal')

    return np.array(xy_list)

def design_center(hgs, source_distance, radius):
    """design a metasurface lens following the "traditional" approach of laying
    out cylinders with centers on a hexagonal grid. hgs is a HexGridSet.
    
    Return a lens_center_summary [[xcenter, ycenter, index in hgs] in arbitrary
    order"""
    assert isinstance(hgs, lens_center.HexGridSet)
    xy_list = hexagonal_grid(pitch, radius, fourfold_symmetry=False)
    xyi_list = []
    for x,y in xy_list:
        target_phase_here = target_phase((x**2+y**2)**0.5, source_distance)
        # TODO
        # adding pi seems necessary to make the grating part and center part
        # add in phase (based on inspecting the design, I haven't worked
        # through all the conventions used)
        target_phase_here += pi
        xyi_list.append([x,y,hgs.pick_from_phase(target_phase_here)])
    return np.array(xyi_list)
    
def make_center_xyrra_list(hgs, lens_center_summary):
    """HexGridSet, lens_center_summary"""
    assert isinstance(hgs, lens_center.HexGridSet)
    xyrra_list = []
    for x,y,i in lens_center_summary:
        r = hgs.grating_list[int(i)].xyrra_list[0,2]
        xyrra_list.append([x,y,r,r,0])
    return np.array(xyrra_list)

def design_periphery(collections, source_distance, radius):
    """collections is a list
       [[(phi_start0,phi_end0), gratingcollection0],
        [(phi_start1,phi_end1), gratingcollection1],
        ...]
        where phi_end0 == phi_start1, etc.
       Returns lens_periphery_summary, a dictionary containing:
         * 'r_center_list' (np.array) the radius at the center of each
             subsequent grating
         * 'r_min_list' (np.array) the radius at the inner boundary of this
             grating
         * 'grating_period_list' (np.array), the period of the corresponding
             grating. Note that
              r_center_list[i] + 0.5 * grating_period_list[i]
                 + 0.5 * grating_period_list[i+1] == r_center_list[i+1]
         * 'gratingcollection_list' (list), GratingCollection objects from
             inside to out (list will look like [gc0, gc1, gc2, ...])
         * 'gratingcollection_index_here_list' (np.array), the index of the
             applicable gratingcollection object for each ring of gratings
             (indexed from the list above). Looks like [0,0,...,1,1,...2,2...].
         * 'num_around_circle_list' (np.array), how many copies are there
             around 2pi, for each entry of the above. Looks like
             [n1,n1 ...,n2,n2,...].
    """
    for i in range(len(collections)-1):
        assert collections[i][0][1] == collections[i+1][0][0]
    assert all(x[0][0] < x[0][1] for x in collections)
    assert len(collections) > 0
    def num_around_circle(gc):
        """The patterns go in a wedge of angle 2*pi/num_around_circle
        radians. This calculates num_around_circle for a given
        GratingCollection gc.
        
        For round lenses, the gratingcollection.lateral_period parameter is
        redefined as
        "lateral_period at a certain point / tan(angle_in_air) at that point"
        Turns out: 2*pi*source_distance / (lateral period at x / tan(angle_in_air at x))
           == (2*pi*x / lateral_period) == num_around_circle"""
        return int(round(2*pi*source_distance / gc.lateral_period))
    
    r_center_list = []
    grating_period_list = []
    gratingcollection_index_here_list = []
    num_around_circle_list = []
    collection_index = 0
    angle_for_switch = collections[0][0][0]
    phase_zeros = [x for x in target_phase_zeros(radius+2*um, source_distance) if x > source_distance * math.tan(angle_for_switch)]
    if len(phase_zeros) <= 1:
        raise ValueError('Periphery is too small for even one ring')
    phase_zero_index = 0

    while True:
        r_outer = phase_zeros[phase_zero_index+1]
        r_inner = phase_zeros[phase_zero_index]
        r_center = (r_outer + r_inner) / 2
        angle_in_air = math.atan(r_center / source_distance)
        if collections[collection_index][0][1] < angle_in_air:
            collection_index += 1
            if collection_index >= len(collections):
                raise ValueError('radius is too big for provided collections')
            continue
        collection = collections[collection_index][1]
        assert isinstance(collection, grating.GratingCollection)
        num_around_circle_here = num_around_circle(collection)
        num_around_circle_list.append(num_around_circle_here)
        r_center_list.append(r_center)
        grating_period_list.append(r_outer-r_inner)
        gratingcollection_index_here_list.append(collection_index)
        if r_outer > radius:
            break
        phase_zero_index += 1
    r_center_list = np.array(r_center_list)
    grating_period_list = np.array(grating_period_list)
    lens_periphery_summary = {'gratingcollection_list': [i[1] for i in collections],
                              'r_center_list': r_center_list,
                              'r_min_list' : r_center_list - 0.5 * grating_period_list,
                              'r_max_list' : r_center_list + 0.5 * grating_period_list,
                              'grating_period_list': grating_period_list,
                              'gratingcollection_index_here_list': np.array(gratingcollection_index_here_list),
                              'num_around_circle_list': np.array(num_around_circle_list)}
    return lens_periphery_summary

def make_periphery_xyrra_list(lens_periphery_summary):
    """lens_periphery_summary is as outputted by design_periphery()"""
    num_around_circle_list = lens_periphery_summary['num_around_circle_list']
    gratingcollection_list = lens_periphery_summary['gratingcollection_list']
    gratingcollection_index_here_list = lens_periphery_summary['gratingcollection_index_here_list']
    grating_period_list = lens_periphery_summary['grating_period_list']
    r_center_list = lens_periphery_summary['r_center_list']
    xyrra_list = []
    num_rings = len(num_around_circle_list)
    for i in range(num_rings):
        num_around_circle_here = num_around_circle_list[i]
        gc_here = gratingcollection_list[gratingcollection_index_here_list[i]]
        assert isinstance(gc_here, grating.GratingCollection)
        grating_period = grating_period_list[i]
        xyrra_list_here = gc_here.get_one(grating_period=grating_period).xyrra_list
        if i != 0 and gratingcollection_index_here_list[i] == gratingcollection_index_here_list[i-1]:
            # check for pillars passing through the periodic boundary
            # TODO - test this routine
            xyrra_list_prev = gc_here.get_one(grating_period=grating_period_list[i-1]).xyrra_list
            assert xyrra_list_prev.shape == xyrra_list_here.shape
            for j in range(xyrra_list_here.shape[0]):
                if (xyrra_list_prev[j,0] > 0.8 * grating_period
                   and xyrra_list_here[j,0] < 0.2 * grating_period):
                    # want to avoid drawing this ellipse twice
                    xyrra_list_here = np.delete(xyrra_list_here, j, axis=0)
                    # break because the lists are not aligned anymore and I
                    # doubt this will happen with two ellipses simultaneously
                    break
                if (xyrra_list_prev[j,0] < 0.2 * grating_period
                   and xyrra_list_here[j,0] > 0.8 * grating_period):
                    # want to avoid omitting this ellipse altogether
                    xyrra_list_here = np.vstack((xyrra_list_here, [xyrra_list_prev[j,:]]))
                    # break because the lists are not aligned anymore and I
                    # doubt this will happen with two ellipses simultaneously
                    break
        for angle in np.linspace(0, 2*pi, num=num_around_circle_here, endpoint=False):
            for x,y,rx,ry,a in xyrra_list_here:
                x += r_center_list[i]
                xyrra_list.append([x * math.cos(angle) - y * math.sin(angle),
                                   x * math.sin(angle) + y * math.cos(angle),
                                   rx, ry, angle+a])
    return np.array(xyrra_list)

def make_design(collections, source_distance, radius, hgs, make_xyrra_list=False):
    """calculate the design for a full round lens, including both the
    center and the periphery. hgs is the HexGridSet used for the center (see
    lens_center.py).
    collections is of the form [[(15*degree, 20*degree), my_grating_collection_A],
                                [(20*degree, 33.2*degree), my_grating_collection_B],
                                [(33.2*degree, 45*degree), my_grating_collection_C],
                                 ...]
    If make_xyrra_list is True, also return the full xyrra_list
    specifying the center (x,y), radii (rx,ry), and rotation angle a for every
    nano-pillar in the whole lens -- input for make_dxf() etc."""
    if len(collections) > 0:
        n_tio2 = hgs.n_tio2
        n_glass = hgs.n_glass
        cyl_height = hgs.cyl_height
        for _,gc in collections:
            assert gc.lens_type == 'round'
            for g in gc.grating_list:
                assert g.n_tio2 == n_tio2
                assert g.n_glass == n_glass
                assert g.cyl_height == cyl_height
        lens_periphery_summary = design_periphery(collections, source_distance, radius)
        if make_xyrra_list:
            periphery_xyrra_list = make_periphery_xyrra_list(lens_periphery_summary)
        r_for_switch = lens_periphery_summary['r_min_list'][0]
        assert r_for_switch < radius
    else:
        r_for_switch = radius
        periphery_xyrra_list = None
        lens_periphery_summary = None
    
    lens_center_summary=design_center(hgs, source_distance, r_for_switch-300*nm)
    
    if make_xyrra_list:
        center_xyrra_list = make_center_xyrra_list(hgs, lens_center_summary)
        if periphery_xyrra_list is not None:
            xyrra_list = np.vstack((center_xyrra_list, periphery_xyrra_list))
        else:
            xyrra_list = center_xyrra_list
        return lens_periphery_summary, lens_center_summary, r_for_switch, xyrra_list
    return lens_periphery_summary, lens_center_summary, r_for_switch



def make_dxf(xyrra_list):
    """turn an xyrra_list (xcenter, ycenter, radius_x, radius_y, angle of rotation)
    into a dxf file """
    directory_now = os.path.dirname(os.path.realpath(__file__))
    drawing = dxf.drawing(os.path.join(directory_now, 'test.dxf'))
    for i in range(xyrra_list.shape[0]):
        if i % 10000 == 0:
            print(xyrra_list.shape[0] - i, 'ellipses remaining in dxf creation...', flush=True)
        x,y,rx,ry,a = xyrra_list[i,:]
        if rx == ry:
            circ = dxf.circle(radius=rx / um, center=(x/um,y/um))
            drawing.add(circ)
        else:
            ellipse = dxf.ellipse((x/um,y/um), rx/um, ry/um, rotation=a/degree, segments=16)
            drawing.add(ellipse)
    print('saving dxf...', flush=True)
    drawing.save()

def make_dxf2(xyrra_list):
    """turn an xyrra_list (xcenter, ycenter, radius_x, radius_y, angle of rotation)
    into a dxf file. Should be identical to make_dxf(). Seems to run 3X faster.
    
    Note: When I tried this, it looked good except for 2 random polylines near
    the center that I had to delete by hand. Weird! Always check your files."""
    directory_now = os.path.dirname(os.path.realpath(__file__))

    dwg = ezdxf.new('AC1015')
    msp = dwg.modelspace()
    
    points = [(0, 0), (3, 0), (6, 3), (6, 6)]
    msp.add_lwpolyline(points)

    for i in range(xyrra_list.shape[0]):
        if i % 10000 == 0:
            print(xyrra_list.shape[0] - i, 'ellipses remaining in dxf creation...', flush=True)
        x,y,rx,ry,a = xyrra_list[i,:]
        if rx == ry:
            msp.add_circle((x/um,y/um), rx / um)
        else:
            ellipse_pts = grating.ellipse_pts(x/um,y/um, rx/um, ry/um, a, num_points=16)
            # repeat first point twice
            ellipse_pts = np.vstack((ellipse_pts, ellipse_pts[0,:]))
            msp.add_lwpolyline(ellipse_pts)
    print('saving dxf...', flush=True)
    dwg.saveas(os.path.join(directory_now, "test2.dxf"))

def make_svg(xyrra_list):
    """like make_dxf but it's an svg file instead. This is twice as slow as
    making the dxf, but 10X smaller file size, and easy fast viewing with
    Inkscape or other programs."""
    directory_now = os.path.dirname(os.path.realpath(__file__))
    dwg = svgwrite.Drawing(filename=os.path.join(directory_now, 'test.svg'))
    for i in range(xyrra_list.shape[0]):
        if i % 10000 == 0:
            print(xyrra_list.shape[0] - i, 'ellipses remaining in svg creation...')
        x,y,rx,ry,a = xyrra_list[i,:]
        if rx == ry:
            circ = svgwrite.shapes.Circle(center=(x/um,y/um), r=rx / um)
            dwg.add(circ)
        else:
            ellipse = svgwrite.shapes.Ellipse(center=(x/um,y/um), r=(rx/um, ry/um))
            ellipse.rotate(angle=a/degree, center=(x/um,y/um))
            dwg.add(ellipse)
    print('saving svg...', flush=True)
    dwg.save()
    
