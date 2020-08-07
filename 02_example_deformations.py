#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:53:45 2020

@author: matthew
"""



#%% simple case, using model_generator.m



dyke = {'strike' : 0,
        'top_depth' : 1000,
        'bottom_depth' : 3000,
        'length' : 5000,
        'dip' : 80,
        'opening' : 0.5}

xgrid, ygrid, zgrid, los_grid = deformation_eq_dyke_sill_('dyke', **dyke) 
                                                          


import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.imshow(los_grid)

