#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:53:45 2020

@author: matthew
"""



#%% simple case, using model_generator.m



los_grid, x_grid, y_grid, z_grid = deformation_wrapper(dem_crop, ll_extent_crop, deformation_ll, source = 'mogi', **mogi_kwargs)
matrix_show(los_grid)

deformation_ll = (14.17, 40.88,)                                                     # deformation lon lat

dyke_kwargs = {'strike' : 0,
        'top_depth' : 1000,
        'bottom_depth' : 3000,
        'length' : 5000,
        'dip' : 80,
        'opening' : 0.5}
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(dem_crop, ll_extent_crop, deformation_ll, source = 'dyke', **dyke_kwargs)
matrix_show(los_grid)
