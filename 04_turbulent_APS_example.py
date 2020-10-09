#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:35:11 2020

@author: matthew

"""

import numpy as np    
import matplotlib.pyplot as plt
import sys
sys.path.append('./lib/')

from syinterferopy_functions import atmosphere_turb

#%% Things to set

pixel_sizes_degs = [1/200, 1/1200, 1/3600]                              # ~ 0.5km, 90m, 30m
Lcs = [100, 500, 1000, 2000, 5000, 10000]                                      # these are in metres.  
    
#%%

f, axes = plt.subplots(len(pixel_sizes_degs),len(Lcs))

for row_n, pixel_size_degs in enumerate(pixel_sizes_degs):
    lons = np.arange(14.0, 14.4, pixel_size_degs)                                                        # create a simple grid of lons and latitudes at the correct size
    lats = np.arange(55.0, 55.2, pixel_size_degs)
    lons_mg = np.repeat(lons[np.newaxis,:], len(lats), axis = 0)
    lats_mg = np.repeat(lats[::-1, np.newaxis], len(lons), axis = 1)

    for col_n, Lc in enumerate(Lcs):
        ph_turb = atmosphere_turb(1, lons_mg, lats_mg, Lc = Lc, verbose=True, interpolate_threshold = 10000, mean_m = 0.02)
        
        im = axes[row_n, col_n].imshow(ph_turb[0,])
        
        if col_n == 0:
            axes[row_n, col_n].set_ylabel(f"Pixel size: {int(111e3 * pixel_size_degs)} m ")                             # take 1 deg of latitude as 111km (111000m)
        if row_n == 0:
            axes[row_n, col_n].set_title(f"Correlation length (m): {Lc}")
        
   