#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:53:45 2020

@author: matthew
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from syinterferopy_functions import deformation_wrapper
from auxiliary_functions import dem_wrapper, griddata_plot                        # smaller functions


#%%

## Campi Flegrei
dem_ll_width = [(14.14, 40.84), 20]                                                 # lon lat scene width(km) for area covered by interferogram
deformation_ll = (14.14, 40.84,)                                                     # deformation lon lat
mogi_kwargs = {'volume_change' : 1e6,                           
                'depth'         : 2000}                                                 # both in metres

dem_settings = {"srtm_dem_tools_bin" : '/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/' ,       # The python package SRTM-DEM-tools is required for making DEMs (that are used to synthesise topographically correlated signals).  
                                                                                                            # it can be downloaded from https://github.com/matthew-gaddes/SRTM-DEM-tools
                "download_dems"      : False,                                                               # if don't need to download anymore, faster to set to false
                "void_fill"          : False,                                                               # Best to leave as False here, dems can have voids, which it can be worth filling (but can be slow and requires scipy)
                "path_tiles"         : './SRTM_3_tiles/',                                                   # folder to keep SRTM3 tiles in 
                "m_in_pix"           : 92.6,                                                                # pixels in m.  92.6m for a SRTM3 pixel on equator       
                "water_mask_resolution"    : 'f'}                                                           # resolution of water mask.  c (crude), l (low), i (intermediate), h (high), f (full)



#%%  First, make a DEM (digtal elevation model)

dem, dem_crop, ijk_m, ll_extent, ll_extent_crop = dem_wrapper(dem_ll_width, **dem_settings)             # make a dem (both original and cropped ones are returned by this function)

griddata_plot(dem_crop, ll_extent_crop, "01 DEM Cropped to area of interest, with water masked (as white)")


#%% With a dem, we can now generate geocoded deformation patterns.  

# 1: Mogi
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(dem_crop, ll_extent_crop, deformation_ll, source = 'mogi', **mogi_kwargs)
griddata_plot(los_grid, ll_extent_crop, '02: Point (Mogi) source', dem_mode = False)


# 2: Dyke
dyke_kwargs = {'strike' : 0,
        'top_depth' : 1000,
        'bottom_depth' : 3000,
        'length' : 5000,
        'dip' : 80,
        'opening' : 0.5}
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(dem_crop, ll_extent_crop, deformation_ll, source = 'dyke', **dyke_kwargs)
griddata_plot(los_grid, ll_extent_crop, '03: Opening Dyke', dem_mode = False)


#3: Sill
sill_kwargs = {'strike' : 0,
               'depth' : 3000,
               'width' : 5000,
               'length' : 5000,
               'dip' : 1,
               'opening' : 0.5}
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(dem_crop, ll_extent_crop, deformation_ll, source = 'sill', **sill_kwargs)
griddata_plot(los_grid, ll_extent_crop, '04: Inflating Sill', dem_mode = False)


#4: SS EQ
quake_ss_kwargs  = {'strike' : 0,
                    'dip' : 80,
                    'length' : 5000,
                    'rake' : 0,
                    'slip' : 1,
                    'top_depth' : 4000,
                    'bottom_depth' : 8000}
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(dem_crop, ll_extent_crop, deformation_ll, source = 'quake', **quake_ss_kwargs)
griddata_plot(los_grid, ll_extent_crop, '05: SS fault earthquake', dem_mode = False)

#5: Normal EQ
quake_normal_kwargs  = {'strike' : 0,
                        'dip' : 70,
                        'length' : 5000,
                        'rake' : -90,
                        'slip' : 1,
                        'top_depth' : 4000,
                        'bottom_depth' : 8000}
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(dem_crop, ll_extent_crop, deformation_ll, source = 'quake', **quake_normal_kwargs)
griddata_plot(los_grid, ll_extent_crop, '06: Normal fault earthquake', dem_mode = False)


#6: Thrust EQ
quake_thrust_kwargs = {'strike' : 0,
                       'dip' : 30,
                       'length' : 5000,
                       'rake' : 90,
                       'slip' : 1,
                       'top_depth' : 4000,
                       'bottom_depth' : 8000}
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(dem_crop, ll_extent_crop, deformation_ll, source = 'quake', **quake_thrust_kwargs)
griddata_plot(los_grid, ll_extent_crop, '07: Thurst fault earthquake', dem_mode = False)        
        
        

        

        





