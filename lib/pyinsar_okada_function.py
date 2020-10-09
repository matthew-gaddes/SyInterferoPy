#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# The MIT License (MIT)
# Copyright (c) 2018 Massachusetts Institute of Technology
#
# Author: Guillaume Rongier
# This software has been created in projects supported by the US National
# Science Foundation and NASA (PI: Pankratius)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import math
import numpy as np
#from numba import jit, prange                                              # MEG: remove as numba functions have been removed.  

################################################################################
# Okada's surface displacement
################################################################################

def I1(xi, eta, q, delta, nu, R, X, d_tild):
    '''
    Compute the component I1 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I1
    '''
    if math.cos(delta) > 10E-8:
        return (1 - 2*nu)*(-xi/(math.cos(delta)*(R + d_tild))) - I5(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)/math.cos(delta)
    else:
        return -((1 - 2*nu)/2.)*(xi*q/((R + d_tild)**2))

def I2(xi, eta, q, delta, nu, R, y_tild, d_tild):
    '''
    Compute the component I2 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I2
    '''
    return (1 - 2*nu)*(-np.log(R + eta)) - I3(xi, eta, q, delta, nu, R, y_tild, d_tild)

def I3(xi, eta, q, delta, nu, R, y_tild, d_tild):
    '''
    Compute the component I3 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I3
    '''
    if math.cos(delta) > 10E-8:
        return (1 - 2*nu)*(y_tild/(math.cos(delta)*(R + d_tild)) - np.log(R + eta)) + I4(xi, eta, q, delta, nu, R, d_tild)*math.sin(delta)/math.cos(delta)
    else:
        return ((1 - 2*nu)/2.)*(eta/(R + d_tild) + y_tild*q/((R + d_tild)**2) - np.log(R + eta))
    
def I4(xi, eta, q, delta, nu, R, d_tild):
    '''
    Compute the component I4 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I4
    '''
    if math.cos(delta) > 10E-8:
        return (1 - 2*nu)*(np.log(R + d_tild) - math.sin(delta)*np.log(R + eta))/math.cos(delta)
    else:
        return -(1 - 2*nu)*q/(R + d_tild)
    
def I5(xi, eta, q, delta, nu, R, X, d_tild):
    '''
    Compute the component I5 of the model (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return I5
    '''
    if math.cos(delta) > 10E-8:
        return (1 - 2*nu)*2*np.arctan((eta*(X + q*math.cos(delta)) + X*(R + X)*math.sin(delta))/(xi*(R + X)*math.cos(delta)))/math.cos(delta)
    else:
        return -(1 - 2*nu)*xi*math.sin(delta)/(R + d_tild)

def f_x_strike(xi, eta, q, delta, nu):
    '''
    Fault strike component along the x axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return xi*q/(R*(R + eta)) + np.arctan(xi*eta/(q*R)) + I1(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)
def f_x_dip(xi, eta, q, delta, nu):
    '''
    Fault dip component along the x axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    y_tild = eta*math.cos(delta) + q*math.sin(delta)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return q/R - I3(xi, eta, q, delta, nu, R, y_tild, d_tild)*math.sin(delta)*math.cos(delta)
def f_x_tensile(xi, eta, q, delta, nu):
    '''
    Fault tensile component along the x axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    y_tild = eta*math.cos(delta) + q*math.sin(delta)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return (q**2)/(R*(R + eta)) - I3(xi, eta, q, delta, nu, R, y_tild, d_tild)*math.sin(delta)**2

def f_y_strike(xi, eta, q, delta, nu):
    '''
    Fault strike component along the y axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    y_tild = eta*math.cos(delta) + q*math.sin(delta)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return y_tild*q/(R*(R + eta)) + q*math.cos(delta)/(R + eta) + I2(xi, eta, q, delta, nu, R, y_tild, d_tild)*math.sin(delta)
def f_y_dip(xi, eta, q, delta, nu):
    '''
    Fault dip component along the y axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    y_tild = eta*math.cos(delta) + q*math.sin(delta)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return y_tild*q/(R*(R + xi)) + math.cos(delta)*np.arctan(xi*eta/(q*R)) - I1(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)*math.cos(delta)
def f_y_tensile(xi, eta, q, delta, nu):
    '''
    Fault tensile component along the y axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return -d_tild*q/(R*(R + xi)) - math.sin(delta)*(xi*q/(R*(R + eta)) - np.arctan(xi*eta/(q*R))) - I1(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)**2

def f_z_strike(xi, eta, q, delta, nu):
    '''
    Fault strike component along the z axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return d_tild*q/(R*(R + eta)) + q*math.sin(delta)/(R + eta) + I4(xi, eta, q, delta, nu, R, d_tild)*math.sin(delta)
def f_z_dip(xi, eta, q, delta, nu):
    '''
    Fault dip component along the z axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return d_tild*q/(R*(R + xi)) + math.sin(delta)*np.arctan(xi*eta/(q*R)) - I5(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)*math.cos(delta)
def f_z_tensile(xi, eta, q, delta, nu):
    '''
    Fault tensile component along the z axis (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The associated component
    '''
    R = np.sqrt(xi**2 + eta**2 + q**2)
    X = np.sqrt(xi**2 + q**2)
    y_tild = eta*math.cos(delta) + q*math.sin(delta)
    d_tild = eta*math.sin(delta) - q*math.cos(delta)
    return y_tild*q/(R*(R + xi)) + math.cos(delta)*(xi*q/(R*(R + eta)) - np.arctan(xi*eta/(q*R))) - I5(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)**2

def chinnerys_notation(f, x, p, q, L, W, delta, nu):
    '''
    Formula to add the different fault components (for more information, see 
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
    @return The combined components
    '''
    return f(x, p, q, delta, nu)\
           - f(x, p - W, q, delta, nu)\
           - f(x - L, p, q, delta, nu)\
           + f(x - L, p - W, q, delta, nu)
            
def compute_okada_displacement(fault_centroid_x,
                               fault_centroid_y,
                               fault_centroid_depth,
                               fault_strike,
                               fault_dip,
                               fault_length,
                               fault_width,
                               fault_rake,
                               fault_slip,
                               fault_open,
                               poisson_ratio,
                               x_array,
                               y_array):
    '''
    Compute the surface displacements for a rectangular fault, based on
    Okada's model. For more information, see:
    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154
    
    @param fault_centroid_x: x cooordinate for the fault's centroid
    @param fault_centroid_y: y cooordinate for the fault's centroid
    @param fault_centroid_depth: depth of the fault's centroid
    @param fault_strike: strike of the fault ([0 - 2pi], in radian)
    @param fault_dip: dip of the fault ([0 - pi/2], in radian)
    @param fault_length: length of the fault (same unit as x and y)
    @param fault_width: width of the fault (same unit as x and y)
    @param fault_rake: rake of the fault ([-pi - pi], in radian)
    @param fault_slip: slipe of the fault (same unit as x and y)
    @param fault_open: opening of the fault (same unit as x and y)
    @param poisson_ratio: Poisson's ratio
    @param x_array: x cooordinate for the domain within a NumPy array
    @param y_array: y cooordinate for the domain within a NumPy array
    
    @return The surface displacement field
    '''
    U1 = math.cos(fault_rake)*fault_slip
    U2 = math.sin(fault_rake)*fault_slip

    east_component = x_array - fault_centroid_x + math.cos(fault_strike)*math.cos(fault_dip)*fault_width/2.
    north_component = y_array - fault_centroid_y - math.sin(fault_strike)*math.cos(fault_dip)*fault_width/2.
    okada_x_array = math.cos(fault_strike)*north_component + math.sin(fault_strike)*east_component + fault_length/2.
    okada_y_array = math.sin(fault_strike)*north_component - math.cos(fault_strike)*east_component + math.cos(fault_dip)*fault_width
    
    d = fault_centroid_depth + math.sin(fault_dip)*fault_width/2.
    p = okada_y_array*math.cos(fault_dip) + d*math.sin(fault_dip)
    q = okada_y_array*math.sin(fault_dip) - d*math.cos(fault_dip)

    displacement_shape = [3] + list(x_array.shape)
    okada_displacement_array = np.zeros(displacement_shape)
    
    okada_displacement_array[0] = -U1*chinnerys_notation(f_x_strike, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)\
                                  - U2*chinnerys_notation(f_x_dip, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)\
                                  + fault_open*chinnerys_notation(f_x_tensile, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)
    okada_displacement_array[1] = -U1*chinnerys_notation(f_y_strike, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)\
                                  - U2*chinnerys_notation(f_y_dip, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)\
                                  + fault_open*chinnerys_notation(f_y_tensile, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)
    okada_displacement_array[2] = -U1*chinnerys_notation(f_z_strike, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)\
                                  - U2*chinnerys_notation(f_z_dip, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)\
                                  + fault_open*chinnerys_notation(f_z_tensile, okada_x_array, p, q, fault_length, fault_width, fault_dip, poisson_ratio)/(2*np.pi)

    displacement_array = np.zeros(displacement_shape)

    displacement_array[0] = math.sin(fault_strike)*okada_displacement_array[0] - math.cos(fault_strike)*okada_displacement_array[1]
    displacement_array[1] = math.cos(fault_strike)*okada_displacement_array[0] + math.sin(fault_strike)*okada_displacement_array[1]
    displacement_array[2] = okada_displacement_array[2]
            
    return displacement_array

def compute_fault_projection_corners(fault_centroid_x,
                                     fault_centroid_y,
                                     fault_strike,
                                     fault_dip,
                                     fault_length,
                                     fault_width):
    '''
    Compute the corners of the projection of a fault at the surface
    
    @param fault_centroid_x: x cooordinate for the fault's centroid
    @param fault_centroid_y: y cooordinate for the fault's centroid
    @param fault_centroid_depth: depth of the fault's centroid
    @param fault_strike: strike of the fault ([0 - 2pi], in radian)
    @param fault_dip: dip of the fault ([0 - pi/2], in radian)
    @param fault_length: length of the fault (same unit as x and y)
    @param fault_width: width of the fault (same unit as x and y)

    @return A list of corners' coordinates
    '''
    okada_x_1 = fault_length/2.
    okada_x_2 = fault_length/2.
    okada_x_3 = -fault_length/2.
    okada_x_4 = -fault_length/2.
    okada_y_1 = math.cos(fault_dip)*fault_width/2.
    okada_y_2 = -math.cos(fault_dip)*fault_width/2.
    okada_y_3 = -math.cos(fault_dip)*fault_width/2.
    okada_y_4 = math.cos(fault_dip)*fault_width/2.
    
    fault_x_1 = math.sin(fault_strike)*okada_x_1 - math.cos(fault_strike)*okada_y_1
    fault_x_2 = math.sin(fault_strike)*okada_x_2 - math.cos(fault_strike)*okada_y_2
    fault_x_3 = math.sin(fault_strike)*okada_x_3 - math.cos(fault_strike)*okada_y_3
    fault_x_4 = math.sin(fault_strike)*okada_x_4 - math.cos(fault_strike)*okada_y_4
    fault_y_1 = math.cos(fault_strike)*okada_x_1 + math.sin(fault_strike)*okada_y_1
    fault_y_2 = math.cos(fault_strike)*okada_x_2 + math.sin(fault_strike)*okada_y_2
    fault_y_3 = math.cos(fault_strike)*okada_x_3 + math.sin(fault_strike)*okada_y_3
    fault_y_4 = math.cos(fault_strike)*okada_x_4 + math.sin(fault_strike)*okada_y_4
    
    return [[fault_centroid_x + fault_x_1, fault_centroid_y + fault_y_1],
            [fault_centroid_x + fault_x_2, fault_centroid_y + fault_y_2,],
            [fault_centroid_x + fault_x_3, fault_centroid_y + fault_y_3],
            [fault_centroid_x + fault_x_4, fault_centroid_y + fault_y_4]]


# ################################################################################
# # Okada's surface displacement with Numba
# ################################################################################

# @jit(nopython = True)
# def I1_nb(xi, eta, q, delta, nu, R, X, d_tild):
#     '''
#     Compute the component I1 of the model (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return I1
#     '''
#     if math.cos(delta) > 10E-8:
#         return (1 - 2*nu)*(-xi/(math.cos(delta)*(R + d_tild))) - I5_nb(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)/math.cos(delta)
#     else:
#         return -((1 - 2*nu)/2.)*(xi*q/((R + d_tild)**2))

# @jit(nopython = True)
# def I2_nb(xi, eta, q, delta, nu, R, y_tild, d_tild):
#     '''
#     Compute the component I2 of the model (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return I2
#     '''
#     return (1 - 2*nu)*(-math.log(R + eta)) - I3_nb(xi, eta, q, delta, nu, R, y_tild, d_tild)

# @jit(nopython = True)
# def I3_nb(xi, eta, q, delta, nu, R, y_tild, d_tild):
#     '''
#     Compute the component I3 of the model (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return I3
#     '''
#     if math.cos(delta) > 10E-8:
#         return (1 - 2*nu)*(y_tild/(math.cos(delta)*(R + d_tild)) - math.log(R + eta)) + I4_nb(xi, eta, q, delta, nu, R, d_tild)*math.sin(delta)/math.cos(delta)
#     else:
#         return ((1 - 2*nu)/2.)*(eta/(R + d_tild) + y_tild*q/((R + d_tild)**2) - math.log(R + eta))

# @jit(nopython = True)
# def I4_nb(xi, eta, q, delta, nu, R, d_tild):
#     '''
#     Compute the component I4 of the model (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return I4
#     '''
#     if math.cos(delta) > 10E-8:
#         return (1 - 2*nu)*(math.log(R + d_tild) - math.sin(delta)*math.log(R + eta))/math.cos(delta)
#     else:
#         return -(1 - 2*nu)*q/(R + d_tild)

# @jit(nopython = True)
# def I5_nb(xi, eta, q, delta, nu, R, X, d_tild):
#     '''
#     Compute the component I5 of the model (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return I5
#     '''
#     if math.cos(delta) > 10E-8:
#         return (1 - 2*nu)*2*math.atan((eta*(X + q*math.cos(delta)) + X*(R + X)*math.sin(delta))/(xi*(R + X)*math.cos(delta)))/math.cos(delta)
#     else:
#         return -(1 - 2*nu)*xi*math.sin(delta)/(R + d_tild)

# @jit(nopython = True)
# def f_x_strike_nb(xi, eta, q, delta, nu):
#     '''
#     Fault strike component along the x axis (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return The associated component
#     '''
#     R = math.sqrt(xi**2 + eta**2 + q**2)
#     X = math.sqrt(xi**2 + q**2)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     return xi*q/(R*(R + eta)) + math.atan(xi*eta/(q*R)) + I1_nb(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)
# @jit(nopython = True)
# def f_x_dip_nb(xi, eta, q, delta, nu):
#     '''
#     Fault dip component along the x axis (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return The associated component
#     '''
#     R = math.sqrt(xi**2 + eta**2 + q**2)
#     y_tild = eta*math.cos(delta) + q*math.sin(delta)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     return q/R - I3_nb(xi, eta, q, delta, nu, R, y_tild, d_tild)*math.sin(delta)*math.cos(delta)
# @jit(nopython = True)
# def f_x_tensile_nb(xi, eta, q, delta, nu):
#     '''
#     Fault tensile component along the x axis (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return The associated component
#     '''
#     R = math.sqrt(xi**2 + eta**2 + q**2)
#     y_tild = eta*math.cos(delta) + q*math.sin(delta)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     return (q**2)/(R*(R + eta)) - I3_nb(xi, eta, q, delta, nu, R, y_tild, d_tild)*math.sin(delta)**2

# @jit(nopython = True)
# def f_y_strike_nb(xi, eta, q, delta, nu):
#     '''
#     Fault strike component along the y axis (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return The associated component
#     '''
#     R = math.sqrt(xi**2 + eta**2 + q**2)
#     y_tild = eta*math.cos(delta) + q*math.sin(delta)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     return y_tild*q/(R*(R + eta)) + q*math.cos(delta)/(R + eta) + I2_nb(xi, eta, q, delta, nu, R, y_tild, d_tild)*math.sin(delta)
# @jit(nopython = True)
# def f_y_dip_nb(xi, eta, q, delta, nu):
#     '''
#     Fault dip component along the y axis (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return The associated component
#     '''
#     R = math.sqrt(xi**2 + eta**2 + q**2)
#     X = math.sqrt(xi**2 + q**2)
#     y_tild = eta*math.cos(delta) + q*math.sin(delta)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     return y_tild*q/(R*(R + xi)) + math.cos(delta)*math.atan(xi*eta/(q*R)) - I1_nb(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)*math.cos(delta)
# @jit(nopython = True)
# def f_y_tensile_nb(xi, eta, q, delta, nu):
#     '''
#     Fault tensile component along the y axis (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return The associated component
#     '''
#     R = math.sqrt(xi**2 + eta**2 + q**2)
#     X = math.sqrt(xi**2 + q**2)
#     y_tild = eta*math.cos(delta) + q*math.sin(delta)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     return -d_tild*q/(R*(R + xi)) - math.sin(delta)*(xi*q/(R*(R + eta)) - math.atan(xi*eta/(q*R))) - I1_nb(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)**2

# @jit(nopython = True)
# def f_z_strike_nb(xi, eta, q, delta, nu):
#     '''
#     Fault strike component along the z axis (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return The associated component
#     '''
#     R = math.sqrt(xi**2 + eta**2 + q**2)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     return d_tild*q/(R*(R + eta)) + q*math.sin(delta)/(R + eta) + I4_nb(xi, eta, q, delta, nu, R, d_tild)*math.sin(delta)
# @jit(nopython = True)
# def f_z_dip_nb(xi, eta, q, delta, nu):
#     '''
#     Fault dip component along the z axis (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return The associated component
#     '''
#     R = math.sqrt(xi**2 + eta**2 + q**2)
#     X = math.sqrt(xi**2 + q**2)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     return d_tild*q/(R*(R + xi)) + math.sin(delta)*math.atan(xi*eta/(q*R)) - I5_nb(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)*math.cos(delta)
# @jit(nopython = True)
# def f_z_tensile_nb(xi, eta, q, delta, nu):
#     '''
#     Fault tensile component along the z axis (for more information, see 
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
    
#     @return The associated component
#     '''
#     R = math.sqrt(xi**2 + eta**2 + q**2)
#     X = math.sqrt(xi**2 + q**2)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     y_tild = eta*math.cos(delta) + q*math.sin(delta)
#     return y_tild*q/(R*(R + xi)) + math.cos(delta)*(xi*q/(R*(R + eta)) - math.atan(xi*eta/(q*R))) - I5_nb(xi, eta, q, delta, nu, R, X, d_tild)*math.sin(delta)**2

# @jit(nopython = True)
# def compute_okada_displacement_nb(fault_centroid_x,
#                                   fault_centroid_y,
#                                   fault_centroid_depth,
#                                   fault_strike,
#                                   fault_dip,
#                                   fault_length,
#                                   fault_width,
#                                   fault_rake,
#                                   fault_slip,
#                                   fault_open,
#                                   poisson_ratio,
#                                   x_array,
#                                   y_array):
#     '''
#     Compute the surface displacements for a rectangular fault, based on
#     Okada's model. For more information, see:
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154
    
#     @param fault_centroid_x: x cooordinate for the fault's centroid
#     @param fault_centroid_y: y cooordinate for the fault's centroid
#     @param fault_centroid_depth: depth of the fault's centroid
#     @param fault_strike: strike of the fault ([0 - 2pi], in radian)
#     @param fault_dip: dip of the fault ([0 - pi/2], in radian)
#     @param fault_length: length of the fault (same unit as x and y)
#     @param fault_width: width of the fault (same unit as x and y)
#     @param fault_rake: rake of the fault ([-pi - pi], in radian)
#     @param fault_slip: slipe of the fault (same unit as x and y)
#     @param fault_open: opening of the fault (same unit as x and y)
#     @param poisson_ratio: Poisson's ratio
#     @param x_array: x cooordinate for the domain within a NumPy array
#     @param y_array: y cooordinate for the domain within a NumPy array
    
#     @return The surface displacement field
#     '''
#     U1 = math.cos(fault_rake)*fault_slip
#     U2 = math.sin(fault_rake)*fault_slip
    
#     d = fault_centroid_depth + math.sin(fault_dip)*fault_width/2.

#     displacement_array = np.empty((3, x_array.shape[0]))
    
#     for i, _ in np.ndenumerate(x_array):

#         east_component = x_array[i] - fault_centroid_x + math.cos(fault_strike)*math.cos(fault_dip)*fault_width/2.
#         north_component = y_array[i] - fault_centroid_y - math.sin(fault_strike)*math.cos(fault_dip)*fault_width/2.
#         okada_x = math.cos(fault_strike)*north_component + math.sin(fault_strike)*east_component + fault_length/2.
#         okada_y = math.sin(fault_strike)*north_component - math.cos(fault_strike)*east_component + math.cos(fault_dip)*fault_width

#         p = okada_y*math.cos(fault_dip) + d*math.sin(fault_dip)
#         q = okada_y*math.sin(fault_dip) - d*math.cos(fault_dip)

#         okada_displacement_x = -U1*(f_x_strike_nb(okada_x, p, q, fault_dip, poisson_ratio)
#                                     - f_x_strike_nb(okada_x - fault_length, p, q, fault_dip, poisson_ratio)
#                                     - f_x_strike_nb(okada_x, p - fault_width, q, fault_dip, poisson_ratio)
#                                     + f_x_strike_nb(okada_x - fault_length, p - fault_width, q, fault_dip, poisson_ratio))/(2*np.pi)\
#                                - U2*(f_x_dip_nb(okada_x, p, q, fault_dip, poisson_ratio)
#                                      - f_x_dip_nb(okada_x - fault_length, p, q, fault_dip, poisson_ratio)
#                                      - f_x_dip_nb(okada_x, p - fault_width, q, fault_dip, poisson_ratio)
#                                      + f_x_dip_nb(okada_x - fault_length, p - fault_width, q, fault_dip, poisson_ratio))/(2*np.pi)\
#                                + fault_open*(f_x_tensile_nb(okada_x, p, q, fault_dip, poisson_ratio)
#                                              - f_x_tensile_nb(okada_x - fault_length, p, q, fault_dip, poisson_ratio)
#                                              - f_x_tensile_nb(okada_x, p - fault_width, q, fault_dip, poisson_ratio)
#                                              + f_x_tensile_nb(okada_x - fault_length, p - fault_width, q, fault_dip, poisson_ratio))/(2*np.pi)
#         okada_displacement_y = -U1*(f_y_strike_nb(okada_x, p, q, fault_dip, poisson_ratio)
#                                     - f_y_strike_nb(okada_x - fault_length, p, q, fault_dip, poisson_ratio)
#                                     - f_y_strike_nb(okada_x, p - fault_width, q, fault_dip, poisson_ratio)
#                                     + f_y_strike_nb(okada_x - fault_length, p - fault_width, q, fault_dip, poisson_ratio))/(2*np.pi)\
#                                - U2*(f_y_dip_nb(okada_x, p, q, fault_dip, poisson_ratio)
#                                      - f_y_dip_nb(okada_x - fault_length, p, q, fault_dip, poisson_ratio)
#                                      - f_y_dip_nb(okada_x, p - fault_width, q, fault_dip, poisson_ratio)
#                                      + f_y_dip_nb(okada_x - fault_length, p - fault_width, q, fault_dip, poisson_ratio))/(2*np.pi)\
#                                + fault_open*(f_y_tensile_nb(okada_x, p, q, fault_dip, poisson_ratio)
#                                              - f_y_tensile_nb(okada_x - fault_length, p, q, fault_dip, poisson_ratio)
#                                              - f_y_tensile_nb(okada_x, p - fault_width, q, fault_dip, poisson_ratio)
#                                              + f_y_tensile_nb(okada_x - fault_length, p - fault_width, q, fault_dip, poisson_ratio))/(2*np.pi)
#         displacement_array[0][i] = math.sin(fault_strike)*okada_displacement_x - math.cos(fault_strike)*okada_displacement_y
#         displacement_array[1][i] = math.cos(fault_strike)*okada_displacement_x + math.sin(fault_strike)*okada_displacement_y

#         displacement_array[2][i] = -U1*(f_z_strike_nb(okada_x, p, q, fault_dip, poisson_ratio)
#                                         - f_z_strike_nb(okada_x - fault_length, p, q, fault_dip, poisson_ratio)
#                                         - f_z_strike_nb(okada_x, p - fault_width, q, fault_dip, poisson_ratio)
#                                         + f_z_strike_nb(okada_x - fault_length, p - fault_width, q, fault_dip, poisson_ratio))/(2*np.pi)\
#                                   - U2*(f_z_dip_nb(okada_x, p, q, fault_dip, poisson_ratio)
#                                         - f_z_dip_nb(okada_x - fault_length, p, q, fault_dip, poisson_ratio)
#                                         - f_z_dip_nb(okada_x, p - fault_width, q, fault_dip, poisson_ratio)
#                                         + f_z_dip_nb(okada_x - fault_length, p - fault_width, q, fault_dip, poisson_ratio))/(2*np.pi)\
#                                   + fault_open*(f_z_tensile_nb(okada_x, p, q, fault_dip, poisson_ratio)
#                                                 - f_z_tensile_nb(okada_x - fault_length, p, q, fault_dip, poisson_ratio)
#                                                 - f_z_tensile_nb(okada_x, p - fault_width, q, fault_dip, poisson_ratio)
#                                                 + f_z_tensile_nb(okada_x - fault_length, p - fault_width, q, fault_dip, poisson_ratio))/(2*np.pi)
            
#     return displacement_array

# @jit(nopython = True, nogil = True, parallel = True)
# def compute_okada_segments_displacement_nb(segment_parameters, x, y):
#     '''
#     Compute the surface displacements for several rectangular segments, based on
#     Okada's model. For more information, see:
#     Okada, Surface deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154
    
#     @param segment_parameters: a 2D NumPy array or list containing the parameters
#                                of the function compute_okada_displacement_nb for
#                                each segment (except x_array and y_array)
#     @param x_array: x cooordinate for the domain within a NumPy array
#     @param y_array: y cooordinate for the domain within a NumPy array
    
#     @return The surface displacement field
#     '''
#     displacements = np.empty((segment_parameters.shape[0], 3, x.shape[0]))
    
#     for i in prange(segment_parameters.shape[0]):
#         displacements[i] = compute_okada_displacement_nb(segment_parameters[i, 0],
#                                                          segment_parameters[i, 1],
#                                                          segment_parameters[i, 2],
#                                                          segment_parameters[i, 3],
#                                                          segment_parameters[i, 4],
#                                                          segment_parameters[i, 5],
#                                                          segment_parameters[i, 6],
#                                                          segment_parameters[i, 7],
#                                                          segment_parameters[i, 8],
#                                                          segment_parameters[i, 9],
#                                                          segment_parameters[i, 10],
#                                                          x,
#                                                          y)
        
#     displacement = np.zeros((3, x.shape[0]))
#     for i in range(segment_parameters.shape[0]):
#         displacement += displacements[i]
        
#     return displacement

# ################################################################################
# # Okada's internal displacement
# ################################################################################

# def I1_int(xi, eta, z, y, delta, c, d, q, R):
#     '''
#     Compute the component I1 of the model (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return I1
#     '''
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     return -math.cos(delta)*xi/(R + d_tild) - I4_int(xi, eta, z, y, delta, c, d, q, R)*math.sin(delta)

# def I2_int(xi, eta, z, y, delta, c, d, q, R):
#     '''
#     Compute the component I2 of the model (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return I2
#     '''
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     return np.log(R + d_tild) + I3_int(xi, eta, z, y, delta, c, d, q, R)*math.sin(delta)

# def I3_int(xi, eta, z, y, delta, c, d, q, R):
#     '''
#     Compute the component I3 of the model (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return I3
#     '''
#     y_tild = eta*math.cos(delta) + q*math.sin(delta)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     if math.cos(delta) < 10E-8:
#         return (1/2.)*(eta/(R + d_tild) + y_tild*q/((R + d_tild)**2) - np.log(R + eta))
#     else:
#         return y_tild/(math.cos(delta)*(R + d_tild)) - (np.log(R + eta) - math.sin(delta)*np.log(R + d_tild))/(math.cos(delta)**2)

# def I4_int(xi, eta, z, y, delta, c, d, q, R):
#     '''
#     Compute the component I4 of the model (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return I4
#     '''
#     y_tild = eta*math.cos(delta) + q*math.sin(delta)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     X = np.sqrt(xi**2 + q**2)
#     if math.cos(delta) < 10E-8:
#         return xi*y_tild/(2.*(R + d_tild)**2)
#     else:
#         temp = np.arctan((eta*(X + q*math.cos(delta)) + X*(R + X)*math.sin(delta))/(xi*(R + X)*math.cos(delta)))
#         return math.sin(delta)*xi/(math.cos(delta)*(R + d_tild)) + 2.*temp/(math.cos(delta)**2)

# def fA_1_strike(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault strike component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     phi = np.arctan(xi*eta/(q*R))
#     Y11 = 1/(R*(R + eta))
#     return phi/2. + alpha*xi*q*Y11/2.
# def fA_2_strike(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault strike component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     return alpha*q/(2.*R)
# def fA_3_strike(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault strike component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     Y11 = 1/(R*(R + eta))
#     return np.log(R + eta)*(1 - alpha)/2. - alpha*(q**2)*Y11/2.
# def fB_1_strike(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault strike component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     phi = np.arctan(xi*eta/(q*R))
#     Y11 = 1/(R*(R + eta))
#     return -xi*q*Y11 - phi - (1 - alpha)*I1_int(xi, eta, z, y, delta, c, d, q, R)*math.sin(delta)/alpha
# def fB_2_strike(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault strike component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     y_tild = eta*math.cos(delta) + q*math.sin(delta)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     return -q/R + (1 - alpha)*y_tild*math.sin(delta)/(alpha*(R + d_tild))
# def fB_3_strike(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault strike component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     Y11 = 1/(R*(R + eta))
#     return Y11*q**2 - (1 - alpha)*I2_int(xi, eta, z, y, delta, c, d, q, R)*math.sin(delta)/alpha
# def fC_1_strike(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault strike component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     Y11 = 1/(R*(R + eta))
#     Y32 = (2*R + eta)/((R**3)*((R + eta)**2))
#     h = q*math.cos(delta) - z
#     Z32 = math.sin(delta)/(R**3) - h*Y32
#     return (1 - alpha)*xi*Y11*math.cos(delta) - alpha*xi*q*Z32
# def fC_2_strike(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault strike component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     c_tild = d_tild + z
#     Y11 = 1/(R*(R + eta))
#     return (1 - alpha)*(math.cos(delta)/R + 2*q*Y11*math.sin(delta)) - alpha*c_tild*q/(R**3)
# def fC_3_strike(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault strike component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     c_tild = d_tild + z
#     Y11 = 1/(R*(R + eta))
#     Y32 = (2*R + eta)/((R**3)*((R + eta)**2))
#     h = q*math.cos(delta) - z
#     Z32 = math.sin(delta)/(R**3) - h*Y32
#     return (1 - alpha)*q*Y11*math.cos(delta) - alpha*(c_tild*eta/R**3 - z*Y11 + Z32*xi**2)

# def fA_1_dip(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault dip component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     return alpha*q/(2.*R)
# def fA_2_dip(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault dip component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     phi = np.arctan(xi*eta/(q*R))
#     X11 = 1/(R*(R + xi))
#     return phi/2. + alpha*eta*q*X11/2.
# def fA_3_dip(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault dip component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     X11 = 1/(R*(R + xi))
#     return np.log(R + xi)*(1 - alpha)/2. - alpha*(q**2)*X11/2.
# def fB_1_dip(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault dip component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     return -q/R + (1 - alpha)*I3_int(xi, eta, z, y, delta, c, d, q, R)*math.sin(delta)*math.cos(delta)/alpha
# def fB_2_dip(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault dip component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     phi = np.arctan(xi*eta/(q*R))
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     X11 = 1/(R*(R + xi))
#     return -eta*q*X11 - phi - (1 - alpha)*xi*math.sin(delta)*math.cos(delta)/(alpha*(R + d_tild))
# def fB_3_dip(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault dip component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     X11 = 1/(R*(R + xi))
#     return X11*q**2 + (1 - alpha)*I4_int(xi, eta, z, y, delta, c, d, q, R)*math.sin(delta)*math.cos(delta)/alpha
# def fC_1_dip(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault dip component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     c_tild = d_tild + z
#     Y11 = 1/(R*(R + eta))
#     return (1 - alpha)*math.cos(delta)/R - q*Y11*math.sin(delta) - alpha*c_tild*q/(R**3)
# def fC_2_dip(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault dip component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     c_tild = d_tild + z
#     y_tild = eta*math.cos(delta) + q*math.sin(delta)
#     X11 = 1/(R*(R + xi))
#     X32 = (2*R + xi)/((R**3)*((R + xi)**2))
#     return (1 - alpha)*y_tild*X11 - alpha*c_tild*eta*q*X32
# def fC_3_dip(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault dip component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     c_tild = d_tild + z
#     X11 = 1/(R*(R + xi))
#     X32 = (2*R + xi)/((R**3)*((R + xi)**2))
#     Y11 = 1/(R*(R + eta))
#     return -d_tild*X11 - xi*Y11*math.sin(delta) - alpha*c_tild*(X11 - X32*q**2)

# def fA_1_tensile(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault tensile component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     Y11 = 1/(R*(R + eta))
#     return -np.log(R + eta)*(1 - alpha)/2. - alpha*(q**2)*Y11/2.
# def fA_2_tensile(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault tensile component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     X11 = 1/(R*(R + xi))
#     return -np.log(R + xi)*(1 - alpha)/2. - alpha*(q**2)*X11/2.
# def fA_3_tensile(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault tensile component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     phi = np.arctan(xi*eta/(q*R))
#     X11 = 1/(R*(R + xi))
#     Y11 = 1/(R*(R + eta))
#     return phi/2. - alpha*q*(eta*X11 + xi*Y11)/2.
# def fB_1_tensile(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault tensile component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     Y11 = 1/(R*(R + eta))
#     return Y11*q**2 - (1 - alpha)*I3_int(xi, eta, z, y, delta, c, d, q, R)*(math.sin(delta)**2)/alpha
# def fB_2_tensile(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault tensile component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     X11 = 1/(R*(R + xi))
#     return X11*q**2 + (1 - alpha)*(xi/(R + d_tild))*(math.sin(delta)**2)/alpha
# def fB_3_tensile(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault tensile component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     phi = np.arctan(xi*eta/(q*R))
#     X11 = 1/(R*(R + xi))
#     Y11 = 1/(R*(R + eta))
#     return q*(eta*X11 + xi*Y11) - phi - (1 - alpha)*I4_int(xi, eta, z, y, delta, c, d, q, R)*(math.sin(delta)**2)/alpha
# def fC_1_tensile(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault tensile component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     Y11 = 1/(R*(R + eta))
#     Y32 = (2*R + eta)/((R**3)*((R + eta)**2))
#     h = q*math.cos(delta) - z
#     Z32 = math.sin(delta)/(R**3) - h*Y32
#     return -(1 - alpha)*(math.sin(delta)/R + q*Y11*math.cos(delta)) - alpha*(z*Y11 - Z32*q**2)
# def fC_2_tensile(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault tensile component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     c_tild = d_tild + z
#     X11 = 1/(R*(R + xi))
#     Y11 = 1/(R*(R + eta))
#     X32 = (2*R + xi)/((R**3)*(R + xi)**2)
#     return (1 - alpha)*2*xi*Y11*math.sin(delta) + d_tild*X11 - alpha*c_tild*(X11 - X32*q**2)
# def fC_3_tensile(xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute part of the fault tensile component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated part
#     '''
#     d = c - z
#     q = y*math.sin(delta) - d*math.cos(delta)
#     R = np.sqrt(xi**2 + eta**2 + q**2)
#     d_tild = eta*math.sin(delta) - q*math.cos(delta)
#     c_tild = d_tild + z
#     y_tild = eta*math.cos(delta) + q*math.sin(delta)
#     X11 = 1/(R*(R + xi))
#     Y11 = 1/(R*(R + eta))
#     X32 = (2*R + xi)/((R**3)*((R + xi)**2))
#     Y32 = (2*R + eta)/((R**3)*(R + eta)**2)
#     h = q*math.cos(delta) - z
#     Z32 = math.sin(delta)/(R**3) - h*Y32
#     return (1 - alpha)*(y_tild*X11 + xi*Y11*math.cos(delta)) + alpha*q*(c_tild*eta*X32 + xi*Z32)
    
# def fA_1(displacement_type, xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute a fault component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated component
#     '''
#     assert displacement_type == 'strike' or displacement_type == 'dip' or displacement_type == 'tensile',\
#         "Unknow displacement type: %r" % displacement_type
#     if displacement_type == 'strike':
#         return fA_1_strike(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'dip':
#         return fA_1_dip(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'tensile':
#         return fA_1_tensile(xi, eta, z, y, delta, c, alpha)
# def fA_2(displacement_type, xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute a fault component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated component
#     '''
#     assert displacement_type == 'strike' or displacement_type == 'dip' or displacement_type == 'tensile',\
#         "Unknow displacement type: %r" % displacement_type
#     if displacement_type == 'strike':
#         return fA_2_strike(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'dip':
#         return fA_2_dip(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'tensile':
#         return fA_2_tensile(xi, eta, z, y, delta, c, alpha)
# def fA_3(displacement_type, xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute a fault component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated component
#     '''
#     assert displacement_type == 'strike' or displacement_type == 'dip' or displacement_type == 'tensile',\
#         "Unknow displacement type: %r" % displacement_type
#     if displacement_type == 'strike':
#         return fA_3_strike(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'dip':
#         return fA_3_dip(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'tensile':
#         return fA_3_tensile(xi, eta, z, y, delta, c, alpha)
# def fB_1(displacement_type, xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute a fault component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated component
#     '''
#     assert displacement_type == 'strike' or displacement_type == 'dip' or displacement_type == 'tensile',\
#         "Unknow displacement type: %r" % displacement_type
#     if displacement_type == 'strike':
#         return fB_1_strike(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'dip':
#         return fB_1_dip(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'tensile':
#         return fB_1_tensile(xi, eta, z, y, delta, c, alpha)
# def fB_2(displacement_type, xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute a fault component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated component
#     '''
#     assert displacement_type == 'strike' or displacement_type == 'dip' or displacement_type == 'tensile',\
#         "Unknow displacement type: %r" % displacement_type
#     if displacement_type == 'strike':
#         return fB_2_strike(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'dip':
#         return fB_2_dip(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'tensile':
#         return fB_2_tensile(xi, eta, z, y, delta, c, alpha)
# def fB_3(displacement_type, xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute a fault component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated component
#     '''
#     assert displacement_type == 'strike' or displacement_type == 'dip' or displacement_type == 'tensile',\
#         "Unknow displacement type: %r" % displacement_type
#     if displacement_type == 'strike':
#         return fB_3_strike(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'dip':
#         return fB_3_dip(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'tensile':
#         return fB_3_tensile(xi, eta, z, y, delta, c, alpha)
# def fC_1(displacement_type, xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute a fault component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated component
#     '''
#     assert displacement_type == 'strike' or displacement_type == 'dip' or displacement_type == 'tensile',\
#         "Unknow displacement type: %r" % displacement_type
#     if displacement_type == 'strike':
#         return fC_1_strike(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'dip':
#         return fC_1_dip(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'tensile':
#         return fC_1_tensile(xi, eta, z, y, delta, c, alpha)
# def fC_2(displacement_type, xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute a fault component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated component
#     '''
#     assert displacement_type == 'strike' or displacement_type == 'dip' or displacement_type == 'tensile',\
#         "Unknow displacement type: %r" % displacement_type
#     if displacement_type == 'strike':
#         return fC_2_strike(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'dip':
#         return fC_2_dip(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'tensile':
#         return fC_2_tensile(xi, eta, z, y, delta, c, alpha)
# def fC_3(displacement_type, xi, eta, z, y, delta, c, alpha):
#     '''
#     Compute a fault component (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The associated component
#     '''
#     assert displacement_type == 'strike' or displacement_type == 'dip' or displacement_type == 'tensile',\
#         "Unknow displacement type: %r" % displacement_type
#     if displacement_type == 'strike':
#         return fC_3_strike(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'dip':
#         return fC_3_dip(xi, eta, z, y, delta, c, alpha)
#     elif displacement_type == 'tensile':
#         return fC_3_tensile(xi, eta, z, y, delta, c, alpha)
    
# def chinnerys_notation_int(f, displacement_type, x, y, z, L, W, delta, c, alpha):
#     '''
#     Formula to add the different fault components (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The combined components
#     '''
#     d = c - z
#     p = y*math.cos(delta) + d*math.sin(delta)
#     return f(displacement_type, x, p, z, y, delta, c, alpha)\
#            - f(displacement_type, x, p - W, z, y, delta, c, alpha)\
#            - f(displacement_type, x - L, p, z, y, delta, c, alpha)\
#            + f(displacement_type, x - L, p - W, z, y, delta, c, alpha)
    
# def compute_fault_internal_displacement_type(displacement_type,
#                                              c,
#                                              L,
#                                              W,
#                                              delta, 
#                                              U, 
#                                              alpha, 
#                                              xxx_array, 
#                                              yyy_array, 
#                                              zzz_array):
#     '''
#     Formula to compute a given type of displacement (for more information, see 
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @return The displacement
#     '''
#     displacement_shape = [3] + list(xxx_array.shape)
#     displacement_array = np.zeros(displacement_shape)
    
#     displacement_array[0] = U*(chinnerys_notation_int(fA_1, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha)
#                                - chinnerys_notation_int(fA_1, displacement_type, xxx_array, yyy_array, -zzz_array, L, W, delta, c, alpha)
#                                + chinnerys_notation_int(fB_1, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha)
#                                + zzz_array*chinnerys_notation_int(fC_1, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha))/(2*np.pi)
#     displacement_array[1] = U*((chinnerys_notation_int(fA_2, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha)
#                                 - chinnerys_notation_int(fA_2, displacement_type, xxx_array, yyy_array, -zzz_array, L, W, delta, c, alpha)
#                                 + chinnerys_notation_int(fB_2, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha)
#                                 + zzz_array*chinnerys_notation_int(fC_2, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha))*math.cos(delta)
#                                - (chinnerys_notation_int(fA_3, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha)
#                                   - chinnerys_notation_int(fA_3, displacement_type, xxx_array, yyy_array, -zzz_array, L, W, delta, c, alpha)
#                                   + chinnerys_notation_int(fB_3, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha)
#                                   + zzz_array*chinnerys_notation_int(fC_3, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha))*math.sin(delta))/(2*np.pi)
#     displacement_array[2] = U*((chinnerys_notation_int(fA_2, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha)
#                                 - chinnerys_notation_int(fA_2, displacement_type, xxx_array, yyy_array, -zzz_array, L, W, delta, c, alpha)
#                                 + chinnerys_notation_int(fB_2, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha)
#                                 - zzz_array*chinnerys_notation_int(fC_2, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha))*math.sin(delta)
#                                + (chinnerys_notation_int(fA_3, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha)
#                                   - chinnerys_notation_int(fA_3, displacement_type, xxx_array, yyy_array, -zzz_array, L, W, delta, c, alpha)
#                                   + chinnerys_notation_int(fB_3, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha)
#                                   - zzz_array*chinnerys_notation_int(fC_3, displacement_type, xxx_array, yyy_array, zzz_array, L, W, delta, c, alpha))*math.cos(delta))/(2*np.pi)

#     return displacement_array

# def compute_okada_internal_displacement(fault_centroid_x,
#                                         fault_centroid_y,
#                                         fault_centroid_depth,
#                                         fault_strike,
#                                         fault_dip,
#                                         fault_length,
#                                         fault_width,
#                                         fault_rake,
#                                         fault_slip,
#                                         fault_open,
#                                         poisson_ratio,
#                                         xxx_array,
#                                         yyy_array,
#                                         depth_array):
#     '''
#     Compute the subsurface displacements for a rectangular fault, based on
#     Okada's model. For more information, see:
#     Okada, Internal deformation due to shear and tensile faults in a half-space,
#     Bulletin of the Seismological Society of America (1992) 82 (2): 1018-1040
    
#     @param fault_centroid_x: x cooordinate for the fault's centroid
#     @param fault_centroid_y: y cooordinate for the fault's centroid
#     @param fault_centroid_depth: depth of the fault's centroid
#     @param fault_strike: strike of the fault ([0 - 2pi], in radian)
#     @param fault_dip: dip of the fault ([0 - pi/2], in radian)
#     @param fault_length: length of the fault (same unit as x and y)
#     @param fault_width: width of the fault (same unit as x and y)
#     @param fault_rake: rake of the fault ([-pi - pi], in radian)
#     @param fault_slip: slipe of the fault (same unit as x and y)
#     @param fault_open: opening of the fault (same unit as x and y)
#     @param poisson_ratio: Poisson's ratio
#     @param xxx_array: x cooordinate for the domain within a 3D array
#     @param yyy_array: y cooordinate for the domain within a 3D array
#     @param depth_array: depth cooordinate for the domain within a 3D array
#                         (i.e., vertical axis oriented downward)
    
#     @return The internal displacement field
#     '''
#     U1 = math.cos(fault_rake)*fault_slip
#     U2 = math.sin(fault_rake)*fault_slip

#     east_component = xxx_array - fault_centroid_x + math.cos(fault_strike)*math.cos(fault_dip)*fault_width/2.
#     north_component = yyy_array - fault_centroid_y - math.sin(fault_strike)*math.cos(fault_dip)*fault_width/2.
#     okada_xxx_array = math.cos(fault_strike)*north_component + math.sin(fault_strike)*east_component + fault_length/2.
#     okada_yyy_array = math.sin(fault_strike)*north_component - math.cos(fault_strike)*east_component + math.cos(fault_dip)*fault_width

#     zzz_array = -depth_array

#     alpha = 1./(2*(1 - poisson_ratio))
#     c = fault_centroid_depth + math.sin(fault_dip)*fault_width/2.

#     okada_displacement_array = compute_fault_internal_displacement_type('strike',
#                                                                         c,
#                                                                         fault_length,
#                                                                         fault_width,
#                                                                         fault_dip,
#                                                                         U1,
#                                                                         alpha,
#                                                                         okada_xxx_array,
#                                                                         okada_yyy_array,
#                                                                         zzz_array)
#     okada_displacement_array += compute_fault_internal_displacement_type('dip',
#                                                                          c,
#                                                                          fault_length,
#                                                                          fault_width,
#                                                                          fault_dip,
#                                                                          U2,
#                                                                          alpha,
#                                                                          okada_xxx_array,
#                                                                          okada_yyy_array,
#                                                                          zzz_array)
#     okada_displacement_array += compute_fault_internal_displacement_type('tensile',
#                                                                          c,
#                                                                          fault_length,
#                                                                          fault_width,
#                                                                          fault_dip,
#                                                                          fault_open,
#                                                                          alpha,
#                                                                          okada_xxx_array,
#                                                                          okada_yyy_array,
#                                                                          zzz_array)

#     displacement_array = np.zeros(okada_displacement_array.shape)

#     displacement_array[0] = math.sin(fault_strike)*okada_displacement_array[0] - math.cos(fault_strike)*okada_displacement_array[1]
#     displacement_array[1] = math.cos(fault_strike)*okada_displacement_array[0] + math.sin(fault_strike)*okada_displacement_array[1]
#     displacement_array[2] = okada_displacement_array[2]
    
#     return displacement_array