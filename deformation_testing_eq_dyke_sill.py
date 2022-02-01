#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 16:54:59 2021

@author: matthew
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/matthew/university_work/python_stuff/python_scripts")                    # personal imports.  
from small_plot_functions import matrix_show

sys.path.append('./lib/')
from syinterferopy_functions import deformation_eq_dyke_sill


los_vector = np.array([[ 0.38213591],
                       [-0.08150437],
                       [ 0.92050485]])

nx = ny = 400                                                                                                         # 18000m in each direction with 90m pixels
X, Y = np.meshgrid(90 * np.arange(0, nx),90 * np.arange(0,ny))                                                        # make a meshgrid
Y = np.flipud(Y)                                                                                                      # change 0 y cordiante from matrix style (top left) to axes style (bottom left)
ij = np.vstack((np.ravel(X)[np.newaxis], np.ravel(Y)[np.newaxis]))                                                    # pairs of coordinates of everywhere we have data   
ijk = np.vstack((ij, np.zeros((1, ij.shape[1]))))   

kwargs = {'strike': 0, 'dip': 70, 'length': 5000, 'rake': -90, 'slip': 1, 'top_depth': 4000, 'bottom_depth': 8000}

import pdb; pdb.set_trace()

U = deformation_eq_dyke_sill('quake', (np.max(X)/3, np.max(Y)/2), ijk, **kwargs)

x_grid = np.reshape(U[0,], (X.shape[0], X.shape[1]))
y_grid = np.reshape(U[1,], (X.shape[0], X.shape[1]))
z_grid = np.reshape(U[2,], (X.shape[0], X.shape[1]))
los_grid = x_grid*los_vector[0,0] + y_grid*los_vector[1,0] + z_grid*los_vector[2,0]

f, axes = plt.subplots(1,4)
f.suptitle('Python')
matrix_show(x_grid, ax = axes[0], fig = f)
matrix_show(y_grid, ax = axes[1], fig = f)
matrix_show(z_grid, ax = axes[2], fig = f)
matrix_show(los_grid, ax = axes[3], fig = f)

